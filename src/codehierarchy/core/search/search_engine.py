from typing import Dict, List
from pathlib import Path
from collections import defaultdict
import numpy as np

from .embedder import HighQualityEmbedder
from .keyword_search import KeywordSearch
from .result import Result


class EnterpriseSearchEngine:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.embedder = HighQualityEmbedder()
        self.keyword_search = KeywordSearch(index_dir / "keyword.db")

        # Load vector index if exists
        try:
            self.vector_index, self.id_mapping = self.embedder.load_index(
                index_dir)
        except Exception:
            self.vector_index = None
            self.id_mapping = {}

    def search(
            self,
            query: str,
            mode: str = 'hybrid',
            top_k: int = 20) -> List[Result]:
        if mode == 'keyword':
            return self.keyword_search.search(query, top_k)
        elif mode == 'semantic':
            return self._semantic_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[Result]:
        if not self.vector_index:
            return []

        try:
            # Encode query
            embedding = self.embedder.encode_batch([query])[0]

            # Search FAISS
            # nprobe should ideally be configurable, defaulting to 64 if not
            # set
            self.vector_index.nprobe = 64
            scores, indices = self.vector_index.search(
                np.array([embedding]).astype('float32'), top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                nid = self.id_mapping.get(idx)
                if nid:
                    # We need to fetch node details.
                    # Ideally we have a metadata store.
                    # For now, return minimal result
                    results.append(Result(
                        node_id=nid,
                        name="?",  # Need lookup
                        file="?",  # Need lookup
                        line=0,
                        summary="?",  # Need lookup
                        score=float(score)
                    ))
            return results
        except Exception as e:
            # Log the error properly instead of just failing silently or
            # crashing
            import logging
            logging.error(f"Semantic search failed: {e}")
            return []

    def _hybrid_search(self, query: str, top_k: int) -> List[Result]:
        # Get candidates
        keyword_results = self.keyword_search.search(query, top_k * 2)
        semantic_results = self._semantic_search(query, top_k * 2)

        # RRF
        fused_scores: Dict[str, float] = defaultdict(float)

        # Map node_id to result object for reconstruction
        result_map = {}

        for rank, res in enumerate(keyword_results, 1):
            fused_scores[res.node_id] += 1.0 / (rank + 60)
            result_map[res.node_id] = res

        for rank, res in enumerate(semantic_results, 1):
            fused_scores[res.node_id] += 1.0 / (rank + 60)
            if res.node_id not in result_map:
                result_map[res.node_id] = res

        # Sort
        ranked_ids = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True)

        final_results = []
        for nid, score in ranked_ids[:top_k]:
            res = result_map[nid]
            res.score = score
            final_results.append(res)

        return final_results
