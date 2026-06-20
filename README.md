# x86-RAG

A Retrieval-Augmented Generation (RAG) assistant scoped specifically to
**Intel's x86-64 instruction set** — instruction semantics, opcode encoding, operands,
flags affected, exceptions, and CPUID/feature requirements.

Runs completely locally (Ollama + ChromaDB + local embeddings). No external API calls.

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- Intel SDM PDFs (see "Data Sources" below)

### Setup

```bash
# Clone the repo
git clone https://github.com/Pranav-v-cs/x86-rag.git
cd x86-rag

# Install dependencies
pip install -r requirements.txt

# Pull the default LLM model
ollama pull qwen3:8b
```

### Data Sources

Place source documents in the following directories:

| Directory | Contents |
|-----------|----------|
| `docs/sdm_vol2/` | Intel® 64 and IA-32 Architectures SDM **Volume 2 (A, B, C, D)**: Instruction Set Reference, A–Z (PDF) |
| `docs/sdm_vol1/` | SDM **Volume 1**: Basic Architecture (PDF, optional) |
| `docs/html_ref/` | Offline HTML copies of community x86 instruction reference, one page per mnemonic (optional) |

**Where to get the SDMs:**
Download from [Intel's official website](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html).
The relevant files are:
- `325383-sdm-vol-2abcd.pdf` (Volume 2, combined)
- `253665-sdm-vol-1.pdf` (Volume 1)

Place the PDFs in the appropriate `docs/sdm_vol*` directories and run `ingest`
(via the web UI or CLI).

Missing directories are skipped gracefully — you can start with just Volume 2.

## Usage

### Web UI (primary)

```bash
./x86-rag
```

Opens a Flask web server at **http://localhost:8086** — the port number references
the Intel 8086 processor.

The UI provides:
- **Query** — ask questions about x86-64 instructions, get LLM-generated answers with source citations
- **Lookup** — direct mnemonic lookup (LLM-free), shows all subsections verbatim
- **Sources** — retrieve relevant chunks without an LLM call (debug retrieval quality)
- **Ingest** — parse PDFs/HTML files, build vector + mnemonic index
- **Stats** — view index statistics (documents, chunks, mnemonics, DB size)

All interactions use HTMX — no page reloads, no JavaScript framework.

### CLI (secondary)

CLI commands remain available via the `--cli` flag:

```bash
./x86-rag --cli ingest
./x86-rag --cli query "What does CMPXCHG16B do?"
./x86-rag --cli lookup CMPXCHG16B
./x86-rag --cli sources "which instructions write to the flags register"
./x86-rag --cli stats
```

Or directly via Python:

```bash
python -m src --cli ingest
python -m src --cli query "What does CMPXCHG16B do?"
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `qwen3:8b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformers model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `X86_RAG_PORT` | `8086` | Web UI port |
| `X86_RAG_DEBUG` | `0` | Enable Flask debug mode (`1`) |

## Architecture

```
                    ┌──────────────┐
                    │   PDF/HTML   │
                    │   Documents  │
                    └──────┬───────┘
                           │ ingest
                    ┌──────▼───────┐    ┌──────────────────┐
                    │  Chunking    │───▶│ Mnemonic Index   │
                    │ (subsection  │    │ (BM25 + dict)    │
                    │  boundaries) │    └──────────────────┘
                    └──────┬───────┘
                           │ embed
                    ┌──────▼───────┐
                    │  ChromaDB    │
                    │  (vectors)   │
                    └──────┬───────┘
                           │ query
                    ┌──────▼───────┐
                    │  Hybrid      │
                    │  Retrieval   │
                    │  (RRF fusion)│
                    └──────┬───────┘
                           │ context
                    ┌──────▼───────┐
                    │  Ollama      │
                    │  (LLM)       │
                    └──────┬───────┘
                           │ answer
                    ┌──────▼───────┐
                    │   Web UI     │
                    │ (Flask+HTMX) │
                    └──────────────┘
```

**Hybrid retrieval** combines:
1. **Exact/fuzzy mnemonic lookup** — direct dict match, checked first
2. **Vector similarity search** — ChromaDB cosine similarity
3. **BM25 keyword search** — sparse retrieval for token-level matches

Results are fused via **Reciprocal Rank Fusion (RRF)** with a score boost when
query tokens match a chunk's mnemonic tag.

## Validation

```bash
tests/benchmark_queries.json
```

Contains 6 benchmark queries with expected key facts. After ingestion, run queries
against the system and verify the expected facts appear in the retrieved chunks
or generated answers. This catches chunking/retrieval regressions.

To run validation:

```bash
./x86-rag --cli sources "What does CMPXCHG16B do and what's required to use it?"
```

## Project Structure

```
x86-rag/
├── docs/
│   ├── sdm_vol1/          # SDM Volume 1 PDFs
│   ├── sdm_vol2/          # SDM Volume 2 PDFs
│   └── html_ref/          # HTML reference files
├── data/
│   └── chroma/            # ChromaDB persistence
├── src/
│   ├── __init__.py
│   ├── __main__.py        # Entry: web server (default) or CLI (--cli)
│   ├── web.py             # Flask web UI
│   ├── cli.py             # Typer CLI (secondary)
│   ├── models.py          # Dataclasses
│   ├── chunking.py        # Subsection detection & chunking
│   ├── ingest.py          # PDF + HTML ingestion
│   ├── embeddings.py      # Sentence Transformers + ChromaDB
│   ├── retrieval.py       # BM25 + vector + RRF fusion
│   ├── llm.py             # Ollama client
│   └── templates/         # Jinja2 templates
│       ├── base.html
│       ├── index.html
│       ├── query_result.html
│       ├── lookup_result.html
│       ├── sources_result.html
│       ├── ingest_result.html
│       └── stats_partial.html
├── tests/
│   └── benchmark_queries.json
├── x86-rag                # Entry point shell script
├── requirements.txt
└── README.md
```
