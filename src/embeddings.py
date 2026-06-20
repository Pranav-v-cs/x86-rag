import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "data/chroma")
COLLECTION_NAME = "x86_instructions"


class EmbeddingManager:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "mnemonic": c.mnemonic,
                "subsection": c.subsection.value,
                "source": c.source,
                "page": str(c.page) if c.page else "",
                "category": c.category.value,
            }
            for c in chunks
        ]

        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = self.embed_texts(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to ChromaDB collection.", len(chunks))
        return len(chunks)

    def query(self, query_text: str, n_results: int = 10) -> list[Chunk]:
        query_embedding = self.embed_text(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        chunks: list[Chunk] = []
        if not results["ids"]:
            return chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            document = results["documents"][0][i] if results["documents"] else ""
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    mnemonic=metadata.get("mnemonic", ""),
                    subsection=metadata.get("subsection", ""),
                    text=document,
                    source=metadata.get("source", ""),
                    page=int(metadata["page"]) if metadata.get("page", "").isdigit() else None,
                    category=metadata.get("category", ""),
                )
            )

        return chunks

    def count(self) -> int:
        return self.collection.count()

    def delete_all(self):
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared existing ChromaDB collection.")
