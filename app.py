import os, re, sqlite3, zipfile, shutil
from pathlib import Path
from collections import Counter, defaultdict
from flask import Flask, render_template, request, redirect, url_for, flash, session
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pdfminer.high_level import extract_text as pdf_extract_text

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

# ---------- Normalisation ----------
def normalize_text(text: str) -> list[str]:
    # lower + tokenize + remove stopwords + keep words len>=2
    tokens = [w.lower() for w in WORD_RE.findall(text)]
    return [w for w in tokens if w not in STOP_WORDS and len(w) >2]

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

def build_snippet(full_text: str, query_word: str, width: int = 120) -> str:
    
    # simple snippet autour de la 1re occurrence
    i = full_text.lower().find(query_word.lower())
    if i == -1:
        return full_text[:width] + ("..." if len(full_text) > width else "")
    start = max(0, i - width//2)
    end = min(len(full_text), i + width//2)
    snippet = full_text[start:end].replace("\n", " ")
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(full_text) else "")

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    summaries = session.pop('last_summary', [])
    return render_template("index.html", summaries=summaries)
#UPLOAD
@app.route("/upload", methods=["POST"])
def upload_zip():

    f = request.files.get("zipfile")
    if not f or f.filename == "":
        flash("Aucun fichier .zip sélectionné")
        return redirect(url_for("index"))
    if not f.filename.lower().endswith(".zip"):
        flash("Merci de fournir un dossier compressé .zip")
        return redirect(url_for("index"))

    # reset data/ et index
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    clear_index()

    # sauvegarder puis EXTRAIRE d'abord
    zip_path = DATA_DIR / "upload.zip"
    f.save(str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)

    # indexer et REMPLIR summaries UNE SEULE FOIS
    summaries = []
    indexed = 0
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".htm", ".html", ".docx", ".pdf"}:
            res = index_document(p, DATA_DIR)  # doit retourner {"name", "rel_path", "top":[(mot, freq),...]}
            if res:
                summaries.append(res)
                indexed += 1

    # stocker le résumé pour affichage unique 
    session['last_summary'] = summaries
    flash(f"Indexation terminée : {indexed} fichiers indexés.")
    return redirect(url_for("index"))
#RECHERCHE
@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("results.html", q="", results=[])

    # tokenizer simple (multi-termes)
    terms = [w for w in normalize_text(q) if w]
    if not terms:
        return render_template("results.html", q=q, results=[])

    # agrégation des scores (somme des fréquences)
    con = db_conn()
    cur = con.cursor()
    scores = defaultdict(int)
    for term in terms:
        cur.execute("SELECT doc_id, freq FROM frequencies WHERE word = ?", (term,))
        for doc_id, freq in cur.fetchall():
            scores[doc_id] += int(freq)

    if not scores:
        con.close()
        return render_template("results.html", q=q, results=[])

    # récupérer métadonnées + construire snippet
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50]
    results = []
    for doc_id, score in ranked:
        cur.execute("SELECT name, rel_path FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        if not row:
            continue
        name, rel_path = row
        full_path = DATA_DIR / rel_path
        raw = extract_text_from_file(full_path)
        snippet = build_snippet(raw, terms[0])
        results.append({
            "name": name,
            "rel_path": rel_path,
            "score": int(score),
            "snippet": snippet
        })
    con.close()
    return render_template("results.html", q=q, results=results)

if __name__ == "__main__":
    app.run(debug=True)
