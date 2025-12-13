"""
Cross-Encoder Reranker for second-stage retrieval.

Processes query-document pairs jointly for higher accuracy
than bi-encoder approaches, at the cost of higher latency.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder


@dataclass
class RankedResult:
    """Result after cross-encoder reranking."""
    chunk_id: str
    score: float
    content: str
    original_score: float


class CrossEncoderReranker:
    """
    Cross-encoder for reranking retrieved candidates.

    Uses ms-marco-MiniLM model for joint query-document scoring,
    providing 20-40% accuracy improvement over bi-encoder results.

    Attributes:
        model: CrossEncoder model instance.
        top_k: Number of results to return after reranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k: int = 10,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model.
            top_k: Number of results to return.
            device: Device for inference.
        """
        logging.info(f"Loading cross-encoder: {model_name}")
        self.model = SentenceTransformersCrossEncoder(model_name, device=device)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, str]],
        top_k: int = None
    ) -> List[RankedResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Search query.
            candidates: List of (chunk_id, original_score, content) tuples.
            top_k: Override default top_k.

        Returns:
            List of RankedResult ordered by cross-encoder score.
        """
        if not candidates:
            return []

        k = top_k or self.top_k

        # Create query-document pairs
        pairs = [(query, content) for _, _, content in candidates]

        # Batch prediction
        logging.debug(f"Reranking {len(candidates)} candidates")
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Combine with original data
        results = []
        for (chunk_id, orig_score, content), score in zip(candidates, scores):
            results.append(RankedResult(
                chunk_id=chunk_id,
                score=float(score),
                content=content,
                original_score=orig_score
            ))

        # Sort by cross-encoder score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:k]
