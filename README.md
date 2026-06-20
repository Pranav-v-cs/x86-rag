# x86-RAG

A CLI-based Retrieval-Augmented Generation (RAG) assistant scoped specifically to
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

Place the PDFs in the appropriate `docs/sdm_vol*` directories and run `ingest`.

Missing directories are skipped gracefully — you can start with just Volume 2.

## Usage

### Ingest documents

```bash
./x86-rag ingest
```

This parses all PDFs/HTML, chunks instruction entries by subsection, generates
embeddings, and stores them in ChromaDB at `data/chroma/`.

### Query

```bash
./x86-rag query "What does CMPXCHG16B do and what's required to use it?"
```

Hybrid retrieval (BM25 + vector search + mnemonic boost) → LLM generates a
cited answer.

### Lookup (LLM-free)

```bash
./x86-rag lookup CMPXCHG16B
```

Direct mnemonic match against the index. Prints all subsections verbatim
(opcode, description, flags, exceptions) with source/page — no LLM involved.

### Sources only

```bash
./x86-rag sources "which instructions write to the flags register"
```

Shows the top retrieved chunks with metadata but no LLM call — useful for
debugging retrieval quality.

### Stats

```bash
./x86-rag stats
```

Shows number of documents indexed, chunks, unique mnemonics, and vector DB size.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `qwen3:8b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformers model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |

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
                    │    CLI       │
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
# Ingest first, then run each query manually:
./x86-rag query "What does CMPXCHG16B do and what's required to use it?"
./x86-rag sources "What does CMPXCHG16B do and what's required to use it?"
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
│   ├── __main__.py
│   ├── cli.py             # Typer CLI
│   ├── models.py          # Dataclasses
│   ├── chunking.py        # Subsection detection & chunking
│   ├── ingest.py          # PDF + HTML ingestion
│   ├── embeddings.py      # Sentence Transformers + ChromaDB
│   ├── retrieval.py       # BM25 + vector + RRF fusion
│   └── llm.py             # Ollama client
├── tests/
│   └── benchmark_queries.json
├── x86-rag                # Entry point shell script
├── requirements.txt
└── README.md
```
