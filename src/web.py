import logging
import os
import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import EmbeddingManager
from src.ingest import ingest_all
from src.llm import OllamaClient
from src.models import SourceCategory
from src.retrieval import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app.template_folder = _template_dir


def _get_retriever() -> HybridRetriever:
    emb = EmbeddingManager()
    retriever = HybridRetriever(emb)
    retriever.initialize()
    return retriever


def _compute_stats():
    emb = EmbeddingManager()
    chroma_count = emb.count()

    retriever = HybridRetriever(emb)
    retriever.initialize()

    unique_mnemonics = len(retriever.mnemonic_index.mnemonic_to_chunks)
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "data/chroma")

    db_size = 0
    chroma_path = Path(chroma_dir)
    if chroma_path.exists():
        for f in chroma_path.rglob("*"):
            if f.is_file():
                db_size += f.stat().st_size

    def _count_files(dirpath, *extensions):
        p = Path(dirpath)
        if not p.exists():
            return 0
        return sum(
            1 for f in p.iterdir() if f.is_file() and f.name != ".gitkeep"
        )

    return {
        "num_documents": (
            _count_files("docs/sdm_vol1")
            + _count_files("docs/sdm_vol2")
            + _count_files("docs/html_ref")
        ),
        "num_chunks": chroma_count,
        "num_unique_mnemonics": unique_mnemonics,
        "vector_db_size": _format_bytes(db_size),
        "sdm_vol1_count": _count_files("docs/sdm_vol1"),
        "sdm_vol2_count": _count_files("docs/sdm_vol2"),
        "html_ref_count": _count_files("docs/html_ref"),
    }


def _format_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template("query_result.html", error="Please enter a question.")

    retriever = _get_retriever()
    if retriever.embeddings.count() == 0:
        return render_template(
            "query_result.html",
            error="No indexed chunks found. Run ingestion first via the Ingest tab.",
        )

    result = retriever.retrieve(question)
    if not result.chunks:
        return render_template("query_result.html", error="No relevant sources found.")

    llm = OllamaClient()
    answer = llm.generate(question, result.chunks)

    return render_template(
        "query_result.html",
        answer=answer,
        chunks=result.chunks,
        question=question,
    )


@app.route("/lookup", methods=["POST"])
def lookup():
    mnemonic = request.form.get("mnemonic", "").strip().upper()
    if not mnemonic:
        return render_template("lookup_result.html", error="Please enter a mnemonic.")

    retriever = _get_retriever()
    chunks = retriever.lookup(mnemonic)

    return render_template("lookup_result.html", chunks=chunks, mnemonic=mnemonic)


@app.route("/sources", methods=["POST"])
def sources():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template("sources_result.html", error="Please enter a question.")

    retriever = _get_retriever()
    if retriever.embeddings.count() == 0:
        return render_template(
            "sources_result.html",
            error="No indexed chunks found. Run ingestion first.",
        )

    result = retriever.retrieve(question)
    return render_template("sources_result.html", chunks=result.chunks, question=question)


@app.route("/ingest", methods=["POST"])
def ingest():
    emb = EmbeddingManager()
    emb.delete_all()

    try:
        chunks = ingest_all()
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        return render_template("ingest_result.html", error=f"Ingestion failed: {e}", done=True)

    if not chunks:
        return render_template(
            "ingest_result.html",
            error="No chunks extracted. Make sure PDFs/HTML files are in docs/ directories.",
            done=True,
        )

    emb.add_chunks(chunks)

    retriever = HybridRetriever(emb)
    retriever.initialize(chunks)

    return render_template(
        "ingest_result.html",
        chunk_count=len(chunks),
        chroma_count=emb.count(),
        done=True,
    )


@app.route("/stats")
def stats():
    s = _compute_stats()
    return render_template("stats_partial.html", s=s)


@app.route("/health")
def health():
    return {"status": "ok"}


def main():
    port = int(os.environ.get("X86_RAG_PORT", 8086))
    debug = os.environ.get("X86_RAG_DEBUG", "0") == "1"
    logger.info("Starting x86-RAG web UI on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
