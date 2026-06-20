import logging
import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import EmbeddingManager
from src.ingest import ingest_all
from src.llm import OllamaClient
from src.models import SourceCategory, Stats
from src.retrieval import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="x86-rag",
    help="Intel x86-64 Instruction Set RAG Assistant",
)


def _get_retriever() -> HybridRetriever:
    emb = EmbeddingManager()
    retriever = HybridRetriever(emb)
    retriever.initialize()
    return retriever


@app.command()
def ingest(
    sdm_vol1: str = typer.Option(
        "docs/sdm_vol1", "--sdm-vol1", help="Path to SDM Volume 1 PDFs"
    ),
    sdm_vol2: str = typer.Option(
        "docs/sdm_vol2", "--sdm-vol2", help="Path to SDM Volume 2 PDFs"
    ),
    html_ref: str = typer.Option(
        "docs/html_ref", "--html-ref", help="Path to HTML reference files"
    ),
):
    """Parse all documents and build vector + mnemonic index."""
    emb = EmbeddingManager()
    emb.delete_all()

    chunks = ingest_all(
        sdm_vol1_dir=sdm_vol1,
        sdm_vol2_dir=sdm_vol2,
        html_ref_dir=html_ref,
    )

    if not chunks:
        typer.echo("No chunks extracted. Check doc directories and try again.")
        raise typer.Exit(1)

    emb.add_chunks(chunks)

    retriever = HybridRetriever(emb)
    retriever.initialize(chunks)

    typer.echo(f"\nIngestion complete.")
    typer.echo(f"  Chunks indexed: {len(chunks)}")
    typer.echo(f"  ChromaDB count: {emb.count()}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question about x86-64 instructions"),
):
    """Hybrid retrieval -> LLM answer with citations."""
    retriever = _get_retriever()

    chunk_count = retriever.embeddings.count()
    if chunk_count == 0:
        typer.echo("No indexed chunks found. Run 'ingest' first.")
        raise typer.Exit(1)

    result = retriever.retrieve(question)
    if not result.chunks:
        typer.echo("No relevant sources found.")
        raise typer.Exit(1)

    llm = OllamaClient()
    answer = llm.generate(question, result.chunks)

    typer.echo(f"\nAnswer: {answer}")


@app.command()
def lookup(
    mnemonic: str = typer.Argument(..., help="Instruction mnemonic (e.g. CMPXCHG16B)"),
):
    """Direct, LLM-free instruction lookup by mnemonic."""
    retriever = _get_retriever()
    chunks = retriever.lookup(mnemonic.upper())

    if not chunks:
        typer.echo(f"No instruction found for mnemonic: {mnemonic}")
        raise typer.Exit(1)

    current_section = None
    for chunk in chunks:
        if chunk.subsection != current_section:
            current_section = chunk.subsection
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"  {chunk.subsection.upper()}")
            typer.echo(f"{'=' * 60}")
        typer.echo(chunk.text)
        page_str = f" [Page {chunk.page}]" if chunk.page else ""
        typer.echo(f"  --- Source: {chunk.source}{page_str} ---\n")


@app.command()
def sources(
    question: str = typer.Argument(..., help="Question to search for"),
):
    """Show retrieved chunks only, no LLM call."""
    retriever = _get_retriever()

    chunk_count = retriever.embeddings.count()
    if chunk_count == 0:
        typer.echo("No indexed chunks found. Run 'ingest' first.")
        raise typer.Exit(1)

    result = retriever.retrieve(question)

    if not result.chunks:
        typer.echo("No relevant sources found.")
        return

    typer.echo(f"\nTop {len(result.chunks)} sources for: {question}\n")
    for i, chunk in enumerate(result.chunks, 1):
        page_str = f" | Page {chunk.page}" if chunk.page else ""
        typer.echo(f"[{i}] Source: {chunk.source}{page_str}")
        typer.echo(f"    Mnemonic: {chunk.mnemonic}")
        typer.echo(f"    Subsection: {chunk.subsection}")
        typer.echo(f"    Text: {chunk.text[:300]}...")
        typer.echo()


@app.command()
def stats():
    """Show index statistics."""
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

    num_docs = 0
    for d in ["docs/sdm_vol1", "docs/sdm_vol2", "docs/html_ref"]:
        p = Path(d)
        if p.exists():
            num_docs += len([f for f in p.iterdir() if f.is_file() and f.name != ".gitkeep"])

    s = Stats(
        num_documents=num_docs,
        num_chunks=chroma_count,
        num_unique_mnemonics=unique_mnemonics,
        vector_db_size_bytes=db_size,
        num_files_by_category={
            "sdm_vol1": len(list(Path("docs/sdm_vol1").glob("*.pdf"))),
            "sdm_vol2": len(list(Path("docs/sdm_vol2").glob("*.pdf"))),
            "html_ref": len(list(Path("docs/html_ref").glob("*.html")))
            + len(list(Path("docs/html_ref").glob("*.htm"))),
        },
    )

    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  x86-RAG Index Statistics")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"  Source documents:    {s.num_documents}")
    typer.echo(f"  Chunks indexed:      {s.num_chunks}")
    typer.echo(f"  Unique mnemonics:    {s.num_unique_mnemonics}")
    typer.echo(f"  Vector DB size:      {_format_bytes(s.vector_db_size_bytes)}")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"  Files by category:")
    for cat, count in s.num_files_by_category.items():
        typer.echo(f"    {cat}: {count} file(s)")
    typer.echo(f"{'=' * 50}")


def _format_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def main():
    app()


if __name__ == "__main__":
    main()
