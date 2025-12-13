"""
Index persistence for FAISS and BM25 indices.

Provides save/load functionality for hybrid search indices
to avoid rebuilding on service restart.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


class IndexPersistence:
    """
    Handles persistence for hybrid search indices.

    Saves and loads:
    - FAISS HNSW vector index
    - BM25 inverted index
    - ID mappings and chunk metadata
    """

    def __init__(self, base_path: Path) -> None:
        """
        Initialize index persistence.

        Args:
            base_path: Base directory for index files.
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.faiss_path = base_path / "faiss.index"
        self.bm25_path = base_path / "bm25.pkl"
        self.mappings_path = base_path / "mappings.pkl"
        self.chunks_path = base_path / "chunks.pkl"

    def save_faiss_index(
        self,
        index: faiss.Index,
        id_to_idx: Dict[str, int],
        idx_to_id: Dict[int, str]
    ) -> None:
        """
        Save FAISS index and ID mappings.

        Args:
            index: FAISS index to save.
            id_to_idx: Chunk ID to index mapping.
            idx_to_id: Index to chunk ID mapping.
        """
        # Save FAISS index
        faiss.write_index(index, str(self.faiss_path))

        # Save mappings
        mappings = {
            "id_to_idx": id_to_idx,
            "idx_to_id": idx_to_id
        }
        with open(self.mappings_path, 'wb') as f:
            pickle.dump(mappings, f)

        logging.info(f"FAISS index saved to {self.faiss_path}")

    def load_faiss_index(
        self
    ) -> Optional[Tuple[faiss.Index, Dict[str, int], Dict[int, str]]]:
        """
        Load FAISS index and ID mappings.

        Returns:
            Tuple of (index, id_to_idx, idx_to_id) or None if not found.
        """
        if not self.faiss_path.exists() or not self.mappings_path.exists():
            return None

        try:
            index = faiss.read_index(str(self.faiss_path))

            with open(self.mappings_path, 'rb') as f:
                mappings = pickle.load(f)

            logging.info(f"FAISS index loaded from {self.faiss_path}")
            return (
                index,
                mappings["id_to_idx"],
                mappings["idx_to_id"]
            )

        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}")
            return None

    def save_bm25_index(
        self,
        tokenized_corpus: List[List[str]],
        chunk_ids: List[str]
    ) -> None:
        """
        Save BM25 index data.

        Note: BM25Okapi doesn't serialize directly, so we save
        the corpus and rebuild on load.

        Args:
            tokenized_corpus: Tokenized documents.
            chunk_ids: Corresponding chunk IDs.
        """
        data = {
            "corpus": tokenized_corpus,
            "ids": chunk_ids
        }
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(data, f)

        logging.info(f"BM25 index saved to {self.bm25_path}")

    def load_bm25_index(
        self
    ) -> Optional[Tuple[BM25Okapi, List[List[str]], List[str]]]:
        """
        Load BM25 index data.

        Returns:
            Tuple of (bm25_index, tokenized_corpus, chunk_ids) or None.
        """
        if not self.bm25_path.exists():
            return None

        try:
            with open(self.bm25_path, 'rb') as f:
                data = pickle.load(f)

            corpus = data["corpus"]
            ids = data["ids"]

            # Rebuild BM25 index
            bm25 = BM25Okapi(corpus)

            logging.info(f"BM25 index loaded from {self.bm25_path}")
            return bm25, corpus, ids

        except Exception as e:
            logging.error(f"Failed to load BM25 index: {e}")
            return None

    def save_chunk_store(self, chunk_store: Dict[str, Dict[str, Any]]) -> None:
        """
        Save chunk metadata store.

        Args:
            chunk_store: Dict mapping chunk_id to chunk data.
        """
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunk_store, f)

        logging.info(f"Chunk store saved ({len(chunk_store)} chunks)")

    def load_chunk_store(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Load chunk metadata store.

        Returns:
            Chunk store dict or None if not found.
        """
        if not self.chunks_path.exists():
            return None

        try:
            with open(self.chunks_path, 'rb') as f:
                chunk_store = pickle.load(f)

            logging.info(f"Chunk store loaded ({len(chunk_store)} chunks)")
            return chunk_store

        except Exception as e:
            logging.error(f"Failed to load chunk store: {e}")
            return None

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about persisted indices."""
        stats = {
            "faiss_exists": self.faiss_path.exists(),
            "bm25_exists": self.bm25_path.exists(),
            "chunks_exists": self.chunks_path.exists(),
            "faiss_size_mb": 0,
            "bm25_size_mb": 0,
            "chunks_size_mb": 0,
        }

        if self.faiss_path.exists():
            stats["faiss_size_mb"] = self.faiss_path.stat().st_size / (1024 * 1024)

        if self.bm25_path.exists():
            stats["bm25_size_mb"] = self.bm25_path.stat().st_size / (1024 * 1024)

        if self.chunks_path.exists():
            stats["chunks_size_mb"] = self.chunks_path.stat().st_size / (1024 * 1024)

        stats["total_size_mb"] = (
            stats["faiss_size_mb"] +
            stats["bm25_size_mb"] +
            stats["chunks_size_mb"]
        )

        return stats

    def clear(self) -> None:
        """Delete all persisted index files."""
        for path in [self.faiss_path, self.bm25_path, 
                     self.mappings_path, self.chunks_path]:
            if path.exists():
                path.unlink()
        logging.info("Index files cleared")
