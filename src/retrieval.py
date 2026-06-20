import logging
import os
import re
from collections import defaultdict
from typing import Optional

from rank_bm25 import BM25Okapi

from src.embeddings import EmbeddingManager
from src.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

MNEMONIC_TOKEN_RE = re.compile(r"[A-Z0-9]{2,}(?:[A-Za-z0-9]*)")
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return MNEMONIC_TOKEN_RE.findall(text.upper())


class MnemonicIndex:
    def __init__(self):
        self.mnemonic_to_chunks: dict[str, list[Chunk]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_chunks: list[Chunk] = []
        self.bm25_corpus: list[list[str]] = []

    def build(self, chunks: list[Chunk]):
        self.mnemonic_to_chunks = defaultdict(list)
        for chunk in chunks:
            for mnem in chunk.mnemonic.split("/"):
                self.mnemonic_to_chunks[mnem.strip()].append(chunk)
                for alt in _tokenize(chunk.text):
                    self.mnemonic_to_chunks[alt].append(chunk)

        self.bm25_chunks = chunks
        self.bm25_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        logger.info(
            "Mnemonic index built: %d unique keys, %d chunks.",
            len(self.mnemonic_to_chunks),
            len(chunks),
        )

    def exact_lookup(self, mnemonic: str) -> list[Chunk]:
        upper = mnemonic.upper().strip()
        result = self.mnemonic_to_chunks.get(upper, [])
        if result:
            result.sort(key=lambda c: c.subsection.value)
        return result

    def bm25_search(self, query: str, n: int = 10) -> list[tuple[Chunk, float]]:
        if self.bm25 is None:
            return []
        tokenized = _tokenize(query)
        if not tokenized:
            return []
        scores = self.bm25.get_scores(tokenized)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n]
        return [(self.bm25_chunks[i], scores[i]) for i in top_indices if scores[i] > 0]


class HybridRetriever:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embeddings = embedding_manager
        self.mnemonic_index = MnemonicIndex()
        self._initialized = False

    def initialize(self, chunks: Optional[list[Chunk]] = None):
        if chunks:
            self.mnemonic_index.build(chunks)
        self._initialized = True

    def retrieve(self, query: str, top_k: int = 8) -> RetrievalResult:
        result = RetrievalResult(chunks=[], query=query)

        tokenized = _tokenize(query)
        mnemonic_matches: dict[str, list[Chunk]] = {}
        for token in tokenized:
            lookup = self.mnemonic_index.exact_lookup(token)
            if lookup:
                mnemonic_matches[token] = lookup

        bm25_results = self.mnemonic_index.bm25_search(query, n=top_k * 2)
        vector_results = self.embeddings.query(query, n_results=top_k * 2)

        score_map: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        rank = 1
        for chunk, _ in bm25_results:
            chunk_map[chunk.chunk_id] = chunk
            score_map[chunk.chunk_id] += 1.0 / (rank + RRF_K)
            rank += 1

        rank = 1
        for chunk in vector_results:
            chunk_map[chunk.chunk_id] = chunk
            score_map[chunk.chunk_id] += 1.0 / (rank + RRF_K)
            rank += 1

        for token, chunks in mnemonic_matches.items():
            for chunk in chunks:
                chunk_map[chunk.chunk_id] = chunk
                score_map[chunk.chunk_id] += 3.0

        sorted_chunks = sorted(
            chunk_map.values(),
            key=lambda c: score_map.get(c.chunk_id, 0),
            reverse=True,
        )

        result.chunks = sorted_chunks[:top_k]
        return result

    def lookup(self, mnemonic: str) -> list[Chunk]:
        return self.mnemonic_index.exact_lookup(mnemonic)
