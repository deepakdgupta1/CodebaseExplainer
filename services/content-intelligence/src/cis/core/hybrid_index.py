"""
Hybrid Index combining FAISS (dense) and BM25 (sparse) search.

Uses Reciprocal Rank Fusion (RRF) to combine results from both
retrieval methods for improved accuracy.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    """Result from hybrid search."""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class HybridIndex:
    """
    Hybrid search index combining dense (FAISS) and sparse (BM25) retrieval.

    Uses CodeBERT for dense embeddings and BM25 for keyword matching,
    with Reciprocal Rank Fusion (RRF) to combine results.

    Attributes:
        model: SentenceTransformer for generating embeddings.
        faiss_index: FAISS HNSW index for dense search.
        bm25_index: BM25 index for sparse search.
        chunk_store: Mapping from chunk_id to chunk data.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        dimension: int = 768,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the hybrid index.

        Args:
            model_name: HuggingFace model for embeddings.
            dimension: Embedding dimension.
            device: Device for model inference.
        """
        logging.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = dimension

        # Dense index (FAISS HNSW)
        self.faiss_index: Optional[faiss.Index] = None
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

        # Sparse index (BM25)
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_ids: List[str] = []

        # Chunk storage
        self.chunk_store: Dict[str, Dict[str, Any]] = {}

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> None:
        """
        Build both dense and sparse indices from chunks.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'content', 'metadata'.
            batch_size: Batch size for embedding generation.
        """
        if not chunks:
            logging.warning("No chunks to index")
            return

        logging.info(f"Building hybrid index for {len(chunks)} chunks")

        # Store chunks
        for chunk in chunks:
            self.chunk_store[chunk["chunk_id"]] = chunk

        # Build dense index
        self._build_faiss_index(chunks, batch_size)

        # Build sparse index
        self._build_bm25_index(chunks)

        logging.info("Hybrid index built successfully")

    def _build_faiss_index(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int
    ) -> None:
        """Build FAISS HNSW index."""
        texts = [c["content"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Generate embeddings
        logging.info("Generating dense embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # Build HNSW index
        M = 16  # Number of connections per layer
        ef_construction = 200

        index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(embeddings.astype('float32'))

        self.faiss_index = index

        # Build ID mappings
        for idx, chunk_id in enumerate(chunk_ids):
            self.id_to_idx[chunk_id] = idx
            self.idx_to_id[idx] = chunk_id

    def _build_bm25_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build BM25 sparse index."""
        logging.info("Building BM25 index...")

        self.tokenized_corpus = []
        self.bm25_ids = []

        for chunk in chunks:
            tokens = self._tokenize(chunk["content"])
            self.tokenized_corpus.append(tokens)
            self.bm25_ids.append(chunk["chunk_id"])

        self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Basic tokenization - split on whitespace and punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def search(
        self,
        query: str,
        k: int = 20,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """
        Hybrid search using RRF fusion.

        Args:
            query: Search query.
            k: Number of results to return.
            alpha: Weight for dense search (1-alpha for sparse).

        Returns:
            List of SearchResult ordered by fused score.
        """
        if self.faiss_index is None or self.bm25_index is None:
            logging.warning("Index not built, returning empty results")
            return []

        # Dense search
        vector_results = self._dense_search(query, k=50)

        # Sparse search
        bm25_results = self._sparse_search(query, k=50)

        # RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results, alpha)

        # Return top-k results
        results = []
        for chunk_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]:
            chunk = self.chunk_store.get(chunk_id, {})
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=score,
                content=chunk.get("content", ""),
                metadata=chunk.get("metadata", {})
            ))

        return results

    def _dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Search using FAISS."""
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        self.faiss_index.hnsw.efSearch = 128
        scores, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                chunk_id = self.idx_to_id.get(idx)
                if chunk_id:
                    results.append((chunk_id, float(score)))

        return results

    def _sparse_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Search using BM25."""
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self.bm25_ids[idx]
                results.append((chunk_id, float(scores[idx])))

        return results

    def _rrf_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        alpha: float,
        rrf_k: int = 60
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion.

        Args:
            dense_results: Results from dense search.
            sparse_results: Results from sparse search.
            alpha: Weight for dense search.
            rrf_k: RRF constant (default 60).

        Returns:
            Dict mapping chunk_id to fused score.
        """
        fused_scores: Dict[str, float] = defaultdict(float)

        for rank, (chunk_id, _) in enumerate(dense_results):
            fused_scores[chunk_id] += alpha * (1.0 / (rrf_k + rank + 1))

        for rank, (chunk_id, _) in enumerate(sparse_results):
            fused_scores[chunk_id] += (1 - alpha) * (1.0 / (rrf_k + rank + 1))

        return dict(fused_scores)

    def update_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Update a single chunk in the index.

        Args:
            chunk: Chunk dict with 'chunk_id', 'content', 'metadata'.
        """
        chunk_id = chunk["chunk_id"]

        # Update store
        self.chunk_store[chunk_id] = chunk

        # Update dense index
        if chunk_id in self.id_to_idx:
            idx = self.id_to_idx[chunk_id]
            embedding = self.model.encode(
                [chunk["content"]],
                normalize_embeddings=True
            ).astype('float32')
            # Note: HNSW doesn't support direct updates, would need rebuild
            # For now, mark as needing reindex
            logging.warning(f"Chunk {chunk_id} updated in store, full reindex recommended")

        # Update BM25 index (also requires rebuild for efficiency)
        # In production, would use incremental BM25 or queue for batch rebuild
