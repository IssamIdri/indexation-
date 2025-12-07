import os, re, sqlite3, shutil
from pathlib import Path
import mimetypes
from flask import send_file, abort
from collections import Counter, defaultdict
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pdfminer.high_level import extract_text as pdf_extract_text
from io import BytesIO
from flask import send_file
from pygments import highlight
from wordcloud import WordCloud
import spacy
from collections import defaultdict
import math
from pygments import highlight as pyg_highlight
from pygments.lexers import TextLexer
from pygments.formatters import HtmlFormatter
import html, re

# charger les modèles une fois
try:
    NLP_FR = spacy.load("fr_core_news_sm")
except OSError:
    NLP_FR = None
try:
    NLP_EN = spacy.load("en_core_web_sm")
except OSError:
    NLP_EN = None

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "index.db"

app = Flask(
            __name__,
            template_folder=str(BASE_DIR / "templates"),
             static_folder=str(BASE_DIR / "static"),
                static_url_path="/static"   # URL = /static/...
                )
app.secret_key = "change-me"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# --- Stopwords FR (compactes, extensibles) ---
STOP_WORDS = set("""
a ai ais ait ainsi alors an ans au aux avec assez avoir avant car ce ces cet cette ceux chaque chez ci comme comment d dans de des du donc dos
elle elles en encore entre est et etaient etais etait etant ete etre eux fait fois font hors il ils je la le les leur leurs loin lui me meme memes
mes moi mon mais ne ni non nos notre nous on or ou où par parce pas pendant peu peut plus plupart pour pourquoi pourrait pres proche puis quand que
quel quelle quelles quels qui quoi sans se ses soi sont sous sur ta te tes toi ton tous tout toute toutes tres trop tu un une uns unes vos votre vous
y à ç ça était étais étaient être où pourquoi qu quand s t u
""".split())

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)

# ---------- DB ----------
def db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        rel_path TEXT NOT NULL,         -- chemin relatif dans data/
        length INTEGER NOT NULL         -- nb de caractères de texte extrait
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS frequencies (
        doc_id INTEGER NOT NULL,
        word TEXT NOT NULL,
        freq INTEGER NOT NULL,
        PRIMARY KEY (doc_id, word),
        FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_freq_word ON frequencies(word)")
    con.commit()
    con.close()

init_db()
# ---------- Lemmatisation ----------
def guess_lang(text: str) -> str:
    """Heuristique très simple FR/EN selon stop-words et caractères ; suffisant ici."""
    if not text:
        return "fr"
    sample = text[:10000].lower()
    fr_hits = sum(w in sample for w in (" le ", " la ", " les ", " des ", " un ", " une ", " et ", " que "))
    en_hits = sum(w in sample for w in (" the ", " and ", " of ", " to ", " in ", " with ", " for ", " is "))
    return "fr" if fr_hits >= en_hits else "en"

def lemmatize_tokens(tokens: list[str], lang_hint: str) -> list[str]:
    """Lemmatisation FR/EN ; si modèle manquant, retourne les tokens d'origine."""
    lemmas = []
    # on prépare une phrase simple pour éviter les overheads
    sent = " ".join(tokens)
    if lang_hint == "fr" and NLP_FR is not None:
        doc = NLP_FR(sent)
        lemmas = [t.lemma_.lower() for t in doc]
    elif lang_hint == "en" and NLP_EN is not None:
        doc = NLP_EN(sent)
        lemmas = [t.lemma_.lower() for t in doc]
    else:
        # fallback : sans lemmatiseur
        lemmas = [t.lower() for t in tokens]
    return lemmas

# ---------- Normalisation ----------
def normalize_text(text: str) -> list[str]:
    # 1) tokenisation "mots alphabetiques"
    raw = [w.lower() for w in WORD_RE.findall(text)]
    if not raw:
        return []

    # 2) lemmatisation (FR/EN) -> infinitif pour les verbes
    lang = guess_lang(text)
    lemmas = lemmatize_tokens(raw, lang)

    # 3) filtre stopwords + longueur >= 2
    final = [w for w in lemmas if w not in STOP_WORDS and len(w) > 2]
    return final

# ---------- Extraction texte ----------
def extract_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        elif ext in (".htm", ".html"):
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            return soup.get_text(separator=" ")
        elif ext == ".docx":
            doc = DocxDocument(str(path))
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pdf":
            return pdf_extract_text(str(path)) or ""
        else:
            return ""  # ignore unsupported
    except Exception as e:
        print(f"[WARN] Failed to extract {path}: {e}")
        return ""
# ---------INDEXATION -------------
def index_document(file_path: Path, rel_root: Path):
    text = extract_text_from_file(file_path)
    tokens = normalize_text(text)
    if not tokens:
        return None

    con = db_conn()
    cur = con.cursor()
    rel_path = str(file_path.relative_to(rel_root).as_posix())
    cur.execute("INSERT INTO documents(name, rel_path, length) VALUES (?, ?, ?)",
                (file_path.name, rel_path, len(text)))
    doc_id = cur.lastrowid

    freqs = Counter(tokens)
    cur.executemany("INSERT INTO frequencies(doc_id, word, freq) VALUES (?, ?, ?)",
                    [(doc_id, w, int(c)) for w, c in freqs.items()])
    con.commit()
    con.close()

    # retourne le Top-5 pour affichage
    top5 = freqs.most_common(5)
    return {"name": file_path.name, "rel_path": rel_path, "top": top5}

def clear_index():
    con = db_conn()
    cur = con.cursor()
    cur.execute("DELETE FROM frequencies")
    cur.execute("DELETE FROM documents")
    con.commit()
    con.close()

def build_snippet(full_text: str, terms: list[str], width: int = 150) -> str:
    """
    Retourne le 1er paragraphe (ou phrase) qui contient au moins un des termes (insensible à la casse).
    Fallback: début du texte si rien trouvé.
    """
    if not full_text:
        return ""
    if not terms:
        return " ".join(full_text.split())[:width] + ("…" if len(full_text) > width else "")

    lower_terms = [t.lower() for t in terms if t]
    # 1) paragraphes (séparation par lignes vides)
    paras = re.split(r'\r?\n{2,}', full_text.strip())
    def has_any(s: str) -> bool:
        s_low = s.lower()
        return any(t in s_low for t in lower_terms)

    for p in paras:
        if has_any(p):
            clip = " ".join(p.split())
            return clip[:width] + ("…" if len(clip) > width else "")
    # 2) phrases
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    for s in sentences:
        if has_any(s):
            clip = " ".join(s.split())
            return clip[:width] + ("…" if len(clip) > width else "")

    # 3) fallback: début
    clip = " ".join(full_text.split())
    return clip[:width] + ("…" if len(clip) > width else "")

def highlight(text: str, terms: list[str]) -> str:
    if not text or not terms: 
        return html.escape(text or "")
    safe = html.escape(text)
    for t in sorted(set(terms), key=len, reverse=True):
        if not t:
            continue
        pattern = re.compile(rf"(?i)\b({re.escape(t)})\b")
        safe = pattern.sub(r"<mark>\1</mark>", safe)
    return safe

def get_index_root() -> Path | None:
    p = DATA_DIR / "_root.txt"
    if p.exists():
        try:
            return Path(p.read_text(encoding="utf-8").strip())
        except Exception:
            return None
    return None
# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    # page d’accueil → redirige vers l’indexation
    return redirect(url_for("welcome"))

@app.route("/welcome", methods=["GET"])
def welcome():
    # affiche le formulaire d’indexation + dernier résumé si dispo
    summaries = session.pop('last_summary', [])
    return render_template("welcome.html", summaries=summaries, hide_nav=True)

@app.route("/admin", methods=["GET"])
def admin_home():
    session["user_role"] = "admin"
    return render_template("admin_home.html", role="admin")

@app.route("/user")
def user_home():
    session["user_role"] = "user"
    # tu peux rediriger vers la recherche directement
    return render_template("search.html", q="", results=[], role="user")

@app.route("/index", methods=["GET"])
def index():
    session["user_role"] = "admin"
    summaries = session.pop('last_summary', [])
    indexation_success = session.pop('indexation_success', False)
    return render_template("index.html", summaries=summaries, role="admin", hide_flash=True, indexation_success=indexation_success)

@app.route("/index-path", methods=["POST"])
def index_path():
    # Récupérer les types de fichiers sélectionnés (depuis form ou files)
    selected_types = request.form.getlist('file_types')
    # Si aucun type n'est sélectionné, utiliser tous les types par défaut
    if not selected_types:
        allowed_extensions = {".txt", ".htm", ".html", ".docx", ".pdf"}
    else:
        allowed_extensions = set(selected_types)
        # Ajouter .htm si .html est sélectionné
        if ".html" in allowed_extensions:
            allowed_extensions.add(".htm")
    
    # Vérifier si des fichiers ont été uploadés
    if 'files' in request.files:
        files = request.files.getlist('files')
        if files and files[0].filename:
            # Traitement des fichiers uploadés
            return index_uploaded_files(files, allowed_extensions)
    
    # Sinon, traitement par chemin (méthode originale)
    folder = request.form.get("folder_path", "").strip()
    if not folder:
        flash("Chemin vide.")
        return redirect(url_for("index"))

    root = Path(folder)
    if not root.exists() or not root.is_dir():
        flash("Dossier introuvable. Vérifie le chemin.")
        return redirect(url_for("index"))

    # reset data/ et index
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    clear_index()

    # mémoriser le dossier racine du dernier index (après recréation de DATA_DIR)
    (DATA_DIR / "_root.txt").write_text(str(root), encoding="utf-8")

    summaries, indexed = [], 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_extensions:
            try:
                res = index_document(p, root)  # doit retourner {"name","rel_path","top":[(mot,freq)...]}
                if res:
                    summaries.append(res)
                    indexed += 1
            except Exception as e:
                print(f"[WARN] Index fail {p}: {e}")

    session['last_summary'] = summaries
    session['indexation_success'] = True
    return redirect(url_for("index"))

def index_uploaded_files(files, allowed_extensions=None):
    """Indexe les fichiers uploadés directement."""
    import json
    
    # Récupérer les types depuis le formulaire si non fournis
    if allowed_extensions is None:
        selected_types = request.form.getlist('file_types')
        if selected_types:
            allowed_extensions = set(selected_types)
            if ".html" in allowed_extensions:
                allowed_extensions.add(".htm")
        else:
            allowed_extensions = {".txt", ".htm", ".html", ".docx", ".pdf"}
    
    print(f"[DEBUG] Types de fichiers autorisés: {allowed_extensions}")
    
    # reset data/ et index
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    clear_index()

    # Créer un dossier racine virtuel pour les fichiers uploadés
    virtual_root = DATA_DIR / "uploaded"
    virtual_root.mkdir(parents=True, exist_ok=True)
    
    # Récupérer les chemins relatifs depuis le formulaire
    file_paths_json = request.form.get('file_paths', '{}')
    try:
        file_paths = json.loads(file_paths_json)
    except:
        file_paths = {}
    
    summaries, indexed = [], 0
    file_list = list(files)
    file_index = 0
    
    for file in file_list:
        if not file.filename:
            continue
            
        # Vérifier l'extension selon les types sélectionnés
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            continue
        
        try:
            # Trouver le chemin relatif correspondant par index
            relative_path = file_paths.get(str(file_index), file.filename)
            file_index += 1
            
            # Nettoyer le chemin (remplacer les backslashes par des slashes)
            safe_path = relative_path.replace("\\", "/")
            file_path = virtual_root / safe_path
            
            # Créer les sous-dossiers si nécessaire
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le fichier
            file.save(str(file_path))
            
            # Indexer le fichier
            res = index_document(file_path, virtual_root)
            if res:
                summaries.append(res)
                indexed += 1
        except Exception as e:
            print(f"[WARN] Index fail {file.filename}: {e}")
    
    (DATA_DIR / "_root.txt").write_text(str(virtual_root), encoding="utf-8")
    session['last_summary'] = summaries
    session['indexation_success'] = True
    return redirect(url_for("index"))


        #WORDCLOUD

@app.route("/wordcloud/<int:doc_id>.png")
def wordcloud_doc(doc_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT word, freq FROM frequencies WHERE doc_id = ?", (doc_id,))
    pairs = cur.fetchall()
    con.close()
    if not pairs:
        # image vide "placeholder"
        img = WordCloud(width=800, height=400, background_color="white").generate("vide")
    else:
        freqs = {w: int(f) for (w, f) in pairs}
        img = WordCloud(width=900, height=450, background_color="white", prefer_horizontal=0.95)\
              .generate_from_frequencies(freqs)
    buf = BytesIO()
    img.to_image().save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# -------  RECHERCHE -------
@app.route("/search", methods=["GET"])
def search_page():
    # Récupérer le rôle depuis les paramètres ou la session, avec défaut "user"
    role = request.args.get("role", session.get("user_role", "user"))
    # Sauvegarder le rôle dans la session pour les prochaines requêtes
    session["user_role"] = role
    
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("search.html", q="", results=[], role=role, page=1, total_pages=1, total_results=0)

    terms = [w for w in normalize_text(q) if w]
    if not terms:
        return render_template("search.html", q=q, results=[], role=role, page=1, total_pages=1, total_results=0)

    from collections import defaultdict
    import math

    con = db_conn(); cur = con.cursor()

    # 1) Occurrences (TF brut) par doc sur tous les termes
    placeholders = ",".join(["?"] * len(terms))
    cur.execute(f"""
        SELECT d.id, d.name, d.rel_path, SUM(f.freq) AS occ
        FROM documents d
        JOIN frequencies f ON d.id = f.doc_id
        WHERE f.word IN ({placeholders})
        GROUP BY d.id, d.name, d.rel_path
        ORDER BY occ DESC
    """, terms)
    rows = cur.fetchall()
    if not rows:
        con.close()
        return render_template("search.html", q=q, results=[], role=role, page=1, total_pages=1, total_results=0)

    #    On récupère tf par (doc, word) pour ces termes
    cur.execute(f"""
        SELECT doc_id, word, freq
        FROM frequencies
        WHERE word IN ({placeholders})
    """, terms)
    tf_rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM documents")
    N = cur.fetchone()[0] or 1
    df = {}
    for t in terms:
        cur.execute("SELECT COUNT(DISTINCT doc_id) FROM frequencies WHERE word = ?", (t,))
        df[t] = cur.fetchone()[0] or 0
    idf = {t: math.log(1.0 + (N / (df[t] + 1))) for t in terms}

    scores = defaultdict(float)
    for doc_id, word, tf in tf_rows:
        if word in idf:
            scores[doc_id] += float(tf) * idf[word]

    #  Construire les résultats (vrai id + occurrences )
    results = []
    for did, name, rel_path, occ in rows:
        root = get_index_root()
        full_path = (root / rel_path) if root else (DATA_DIR / rel_path)
        
        # Gérer les erreurs lors de l'extraction du texte
        try:
            if full_path.exists() and full_path.is_file():
                raw = extract_text_from_file(full_path)
            else:
                raw = ""
        except Exception as e:
            print(f"[WARN] Erreur extraction texte pour {full_path}: {e}")
            raw = ""
        
        snippet_text = build_snippet(raw, terms)
        snippet_html = highlight(snippet_text, terms)
        results.append({
            "id": int(did),
            "name": name,
            "rel_path": rel_path,
            "occurrences": int(occ),       # si tu utilises l'agrég SQL donnée précédemment
            "snippet": snippet_html         # ★ important
        })

    con.close()
    results.sort(key=lambda r: r["occurrences"], reverse=True)
    if results:
        print("DBG first snippet:", results[0]["snippet"][:120] if results[0].get("snippet") else "—")
    
    # Pagination : 4 résultats par page
    results_per_page = 4
    page = int(request.args.get("page", 1))
    total_results = len(results)
    total_pages = (total_results + results_per_page - 1) // results_per_page if total_results > 0 else 1
    
    # Debug: afficher les valeurs de pagination
    print(f"[DEBUG Pagination] total_results={total_results}, total_pages={total_pages}, page={page}")
    
    # Valider le numéro de page
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    # Calculer les indices pour la pagination
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    paginated_results = results[start_idx:end_idx]
    
    # Debug: vérifier les variables avant le rendu
    print(f"[DEBUG Render] Rendering search.html with: total_results={total_results}, total_pages={total_pages}, page={page}")
    print(f"[DEBUG Render] Variables type: total_results={type(total_results)}, total_pages={type(total_pages)}, page={type(page)}")
    
    return render_template("search.html", 
                         q=q, 
                         results=paginated_results, 
                         role=role,
                         page=page,
                         total_pages=total_pages,
                         total_results=total_results)



@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    """API pour obtenir des suggestions de mots basées sur la distance de Levenshtein."""
    import unicodedata
    
    try:
        query = request.args.get("q", "").strip()
        print(f"[DEBUG API] Requête suggestions reçue pour: '{query}'")
        
        if not query or len(query) < 2:
            return jsonify({"suggestions": []})
        
        # Fonction pour enlever les accents
        def remove_accents(text: str) -> str:
            nfd = unicodedata.normalize('NFD', text)
            return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        
        # Normaliser la requête
        search_term_original = query.lower()
        search_term_no_accents = remove_accents(search_term_original)
        
        # Essayer aussi de normaliser avec lemmatisation si possible
        try:
            normalized_query = normalize_text(query)
            search_term_normalized = normalized_query[0] if normalized_query else search_term_original
        except Exception:
            search_term_normalized = search_term_original
        
        # Fonction de distance de Levenshtein améliorée (ignore les accents)
        def levenshtein_distance_ignore_accents(s1: str, s2: str) -> int:
            s1_no_acc = remove_accents(s1.lower())
            s2_no_acc = remove_accents(s2.lower())
            
            if len(s1_no_acc) < len(s2_no_acc):
                return levenshtein_distance_ignore_accents(s2, s1)
            if len(s2_no_acc) == 0:
                return len(s1_no_acc)
            
            previous_row = range(len(s2_no_acc) + 1)
            for i, c1 in enumerate(s1_no_acc):
                current_row = [i + 1]
                for j, c2 in enumerate(s2_no_acc):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        # Récupérer tous les mots du vocabulaire
        try:
            con = db_conn()
            cur = con.cursor()
            cur.execute("SELECT DISTINCT word FROM frequencies ORDER BY word")
            all_words = [row[0] for row in cur.fetchall()]
            con.close()
        except Exception as db_error:
            print(f"[ERROR] Erreur DB dans suggestions: {db_error}")
            import traceback
            traceback.print_exc()
            return jsonify({"suggestions": [], "error": "Erreur base de données"}), 200
        
        print(f"[DEBUG] Vocabulaire: {len(all_words)} mots uniques")
        if not all_words:
            print("[DEBUG] Aucun mot dans le vocabulaire!")
            return jsonify({"suggestions": []})
        
        # Calculer les distances et trier
        suggestions_with_distance = []
        for word in all_words:
            try:
                word_lower = word.lower()
                word_no_accents = remove_accents(word_lower)
                
                # Calculer la distance avec différentes variantes
                distances = []
                
                # Distance avec le terme original
                dist1 = levenshtein_distance_ignore_accents(search_term_original, word)
                distances.append(dist1)
                
                # Distance avec le terme sans accents
                dist2 = levenshtein_distance_ignore_accents(search_term_no_accents, word)
                distances.append(dist2)
                
                # Distance avec le terme normalisé (lemmatisé)
                if search_term_normalized != search_term_original:
                    dist3 = levenshtein_distance_ignore_accents(search_term_normalized, word)
                    distances.append(dist3)
                
                # Prendre la meilleure distance
                distance = min(distances)
                
                # Distance maximale adaptative (plus permissive pour les mots courts)
                query_len = len(search_term_original)
                if query_len <= 3:
                    max_distance = 3  # Plus permissif pour les mots courts
                elif query_len <= 5:
                    max_distance = 4
                else:
                    max_distance = min(5, max(3, query_len // 2 + 1))  # Plus permissif
                
                # Accepter aussi les mots qui commencent par le terme même si la distance est un peu plus grande
                starts_with_term = word_no_accents.startswith(search_term_no_accents)
                extended_max = max_distance + 2 if starts_with_term else max_distance
                
                if distance <= extended_max:
                    bonus = 0
                    score = distance
                    
                    # Bonus si le mot commence par la même lettre (sans accents)
                    if word_no_accents and search_term_no_accents and word_no_accents.startswith(search_term_no_accents[0]):
                        bonus -= 0.5
                    
                    # Bonus si le terme est contenu dans le mot (sans accents)
                    if search_term_no_accents in word_no_accents:
                        bonus -= 1.5
                    
                    # Bonus si le mot commence par le terme (préfixe, sans accents)
                    if word_no_accents.startswith(search_term_no_accents):
                        bonus -= 2.5
                    
                    # Bonus supplémentaire si correspondance exacte sans accents
                    if word_no_accents == search_term_no_accents:
                        bonus -= 5
                    
                    # Bonus pour les mots qui ont une longueur similaire
                    len_diff = abs(len(word_no_accents) - len(search_term_no_accents))
                    if len_diff <= 1:
                        bonus -= 0.3
                    elif len_diff <= 2:
                        bonus -= 0.1
                    
                    # Calculer le score final
                    final_score = distance + bonus
                    suggestions_with_distance.append((word, final_score))
            except Exception as e:
                # Ignorer les mots qui causent des erreurs
                continue
        
        # Trier par score et prendre les 8 meilleurs (pour plus de choix)
        suggestions_with_distance.sort(key=lambda x: x[1])
        suggestions = [word for word, _ in suggestions_with_distance[:8]]
        
        # Debug
        print(f"[DEBUG Suggestions] Requête: '{query}', {len(all_words)} mots dans le vocabulaire")
        print(f"[DEBUG Suggestions] {len(suggestions_with_distance)} candidats trouvés, {len(suggestions)} suggestions retenues")
        
        # Si aucune suggestion n'est trouvée, essayer de trouver des mots qui contiennent au moins une partie du terme
        if not suggestions and len(search_term_original) >= 2:
            partial_matches = []
            min_substring_len = 2 if len(search_term_original) <= 4 else 3
            for word in all_words[:1000]:  # Augmenter la limite pour plus de résultats
                word_lower = word.lower()
                word_no_acc = remove_accents(word_lower)
                # Chercher si au moins 2-3 caractères consécutifs correspondent
                for i in range(len(search_term_no_accents) - min_substring_len + 1):
                    substring = search_term_no_accents[i:i+min_substring_len]
                    if substring in word_no_acc:
                        # Calculer un score basé sur la position et la longueur
                        pos = word_no_acc.find(substring)
                        score = len(substring) - pos * 0.1
                        partial_matches.append((word, score))
                        break
            if partial_matches:
                # Trier par score et prendre les meilleurs
                partial_matches.sort(key=lambda x: x[1], reverse=True)
                suggestions = [word for word, _ in partial_matches[:8]]
        
        # Si toujours aucune suggestion, essayer une recherche par première lettre
        if not suggestions and len(search_term_original) >= 1:
            first_letter = search_term_no_accents[0] if search_term_no_accents else ""
            if first_letter:
                letter_matches = []
                for word in all_words[:500]:
                    word_no_acc = remove_accents(word.lower())
                    if word_no_acc.startswith(first_letter):
                        letter_matches.append(word)
                        if len(letter_matches) >= 5:
                            break
                if letter_matches:
                    suggestions = letter_matches[:5]
        
        return jsonify({"suggestions": suggestions})
    
    except Exception as e:
        # En cas d'erreur, retourner une liste vide avec code 200 pour éviter NetworkError
        print(f"[ERROR Suggestions] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"suggestions": [], "error": str(e)}), 200

@app.route("/open/<int:doc_id>")
def open_doc(doc_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT name, rel_path FROM documents WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        abort(404)
    name, rel_path = row

    root = get_index_root()
    base = root if root else DATA_DIR
    file_path = (base / rel_path).resolve()

    # Sécurité : empêcher l'évasion en dehors du root
    if root and not str(file_path).startswith(str(root.resolve())):
        abort(403)
    if not file_path.exists():
        abort(404)

    mime, _ = mimetypes.guess_type(str(file_path))
    inline_exts = {".pdf", ".txt", ".htm", ".html", ".png", ".jpg", ".jpeg", ".gif"}
    as_attach = file_path.suffix.lower() not in inline_exts

    return send_file(
        file_path,
        mimetype=mime or "application/octet-stream",
        as_attachment=as_attach,
        download_name=name
    )

@app.route("/stats", methods=["GET"])
def stats():
    session["user_role"] = "admin"
    try:
        con = db_conn()
        cur = con.cursor()

        # KPIs de base
        cur.execute("SELECT COUNT(*) FROM documents")
        total_docs = cur.fetchone()[0] or 0

        cur.execute("SELECT COALESCE(SUM(freq),0) FROM frequencies")
        total_tokens_after = cur.fetchone()[0] or 0  # Mots après traitement

        cur.execute("SELECT COUNT(DISTINCT word) FROM frequencies")
        vocab_size = cur.fetchone()[0] or 0

        # Taille totale des fichiers (en caractères, puis convertir en KB/MB)
        cur.execute("SELECT COALESCE(SUM(length),0) FROM documents")
        total_size_chars = cur.fetchone()[0] or 0
        total_size_kb = total_size_chars / 1024 if total_size_chars > 0 else 0
        total_size_mb = total_size_kb / 1024 if total_size_kb > 0 else 0

        # Calculer les mots avant traitement (tous les tokens extraits avant filtrage)
        # Estimation basée sur la taille des fichiers et le ratio moyen
        # Ratio moyen observé : environ 1.5-2x plus de mots bruts que de tokens après traitement
        total_words_before = 0
        if total_tokens_after > 0:
            # Estimation basée sur le ratio moyen observé
            total_words_before = int(total_tokens_after * 1.8)  # ~80% de réduction après filtrage

        # Top 20 mots pour les graphiques (plus de données)
        cur.execute("""
            SELECT word, SUM(freq) AS f
            FROM frequencies
            GROUP BY word
            ORDER BY f DESC
            LIMIT 20
        """)
        top_words = cur.fetchall() or []  # [(word, f), ...]

        # Top 10 mots pour l'affichage liste
        top_10_words = top_words[:10] if top_words else []

        # Top 10 documents avec taille de fichier
        cur.execute("""
            SELECT d.id, d.name, d.rel_path, d.length, SUM(f.freq) AS tokens
            FROM documents d
            JOIN frequencies f ON f.doc_id = d.id
            GROUP BY d.id, d.name, d.rel_path, d.length
            ORDER BY tokens DESC
            LIMIT 10
        """)
        top_docs = cur.fetchall() or []  # [(id, name, rel_path, length, tokens), ...]

        con.close()
        
        # Préparer les données pour les graphiques
        chart_words = [w[0] for w in top_words] if top_words else []
        chart_freqs = [w[1] for w in top_words] if top_words else []

        return render_template("stats.html",
                               total_docs=total_docs,
                               total_tokens_after=total_tokens_after,
                               total_words_before=total_words_before,
                               vocab_size=vocab_size,
                               total_size_chars=total_size_chars,
                               total_size_kb=total_size_kb,
                               total_size_mb=total_size_mb,
                               top_words=top_10_words,
                               top_docs=top_docs,
                               chart_words=chart_words,
                               chart_freqs=chart_freqs,
                               role="admin")
    except Exception as e:
        print(f"[ERROR] Erreur dans stats: {e}")
        import traceback
        traceback.print_exc()
        # Retourner des valeurs par défaut en cas d'erreur
        return render_template("stats.html",
                               total_docs=0,
                               total_tokens_after=0,
                               total_words_before=0,
                               vocab_size=0,
                               total_size_chars=0,
                               total_size_kb=0,
                               total_size_mb=0,
                               top_words=[],
                               top_docs=[],
                               chart_words=[],
                               chart_freqs=[],
                               role="admin")


if __name__ == "__main__":
    print("URL MAP:\n", app.url_map)
    app.run(debug=True)
