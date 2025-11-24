import os, re, sqlite3, shutil
from pathlib import Path
import mimetypes
from flask import send_file, abort
from collections import Counter, defaultdict
from flask import Flask, render_template, request, redirect, url_for, flash, session
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

init_db()# ---------- Lemmatisation ----------
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

def build_snippet(full_text: str, terms: list[str], width: int = 400) -> str:
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
    return render_template("admin_home.html", role="admin")

@app.route("/user")
def user_home():
    # tu peux rediriger vers la recherche directement
    return render_template("search.html", q="", results=[], role="user")

@app.route("/index", methods=["GET"])
def index():
    summaries = session.pop('last_summary', [])
    return render_template("index.html", summaries=summaries, role="admin")

@app.route("/index-path", methods=["POST"])
def index_path():

    folder = request.form.get("folder_path", "").strip()
    if not folder:
        flash("Chemin vide.")
        return redirect(url_for("index"))   # <- page d'indexation

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
        if p.is_file() and p.suffix.lower() in {".txt", ".htm", ".html", ".docx", ".pdf"}:
            try:
                res = index_document(p, root)  # doit retourner {"name","rel_path","top":[(mot,freq)...]}
                if res:
                    summaries.append(res)
                    indexed += 1
            except Exception as e:
                print(f"[WARN] Index fail {p}: {e}")

    session['last_summary'] = summaries
    flash(f"Indexation terminée : {indexed} fichiers indexés.")
    return redirect(url_for("index"))  # <- assure-toi que cette route existe


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

# ------- Page 2 : RECHERCHE -------
@app.route("/search", methods=["GET"])
def search_page():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("search.html", q="", results=[])

    terms = [w for w in normalize_text(q) if w]
    if not terms:
        return render_template("search.html", q=q, results=[])

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
        return render_template("search.html", q=q, results=[])

    # 2) (Optionnel) calcul du score TF-IDF en plus du TF
    #    On récupère tf par (doc, word) pour ces termes
    cur.execute(f"""
        SELECT doc_id, word, freq
        FROM frequencies
        WHERE word IN ({placeholders})
    """, terms)
    tf_rows = cur.fetchall()

    # N et df(term) pour IDF
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

    # 3) Construire les résultats (vrai id + occurrences )
    results = []
    for did, name, rel_path, occ in rows:
        root = get_index_root()
        full_path = (root / rel_path) if root else (DATA_DIR / rel_path)
        raw = extract_text_from_file(full_path)
        snippet_text = build_snippet(raw, terms)
        snippet_html = highlight(snippet_text, terms)
        results.append({
            "id": int(did),
            "name": name,
            "rel_path": rel_path,
            "occurrences": int(occ),       # si tu utilises l’agrég SQL donnée précédemment
            "snippet": snippet_html         # ★ important
        })

    con.close()
    # Tri final : par TF-IDF si tu veux, sinon par occurrences
    results.sort(key=lambda r: r["occurrences"], reverse=True)
    print("DBG first snippet:", (results[0]["snippet"][:120] if results else "—"))
    role = request.args.get("role", "user")
    return render_template("search.html", q=q, results=results, role=role)



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
    con = db_conn(); cur = con.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM documents")
    total_docs = cur.fetchone()[0] or 0

    cur.execute("SELECT COALESCE(SUM(freq),0) FROM frequencies")
    total_tokens = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT word) FROM frequencies")
    vocab_size = cur.fetchone()[0] or 0

    # Top 10 mots (corpus entier)
    cur.execute("""
        SELECT word, SUM(freq) AS f
        FROM frequencies
        GROUP BY word
        ORDER BY f DESC
        LIMIT 10
    """)
    top_words = cur.fetchall()  # [(word, f), ...]

    # Top 10 documents par volume de tokens
    cur.execute("""
        SELECT d.id, d.name, d.rel_path, SUM(f.freq) AS tokens
        FROM documents d
        JOIN frequencies f ON f.doc_id = d.id
        GROUP BY d.id, d.name, d.rel_path
        ORDER BY tokens DESC
        LIMIT 10
    """)
    top_docs = cur.fetchall()  # [(id, name, rel_path, tokens), ...]

    con.close()
    return render_template("stats.html",
                           total_docs=total_docs,
                           total_tokens=total_tokens,
                           vocab_size=vocab_size,
                           top_words=top_words,
                           top_docs=top_docs,
                           role="admin")


if __name__ == "__main__":
    print("URL MAP:\n", app.url_map)
    app.run(debug=True)
