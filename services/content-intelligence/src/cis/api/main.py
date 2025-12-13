"""
Content Intelligence Service - FastAPI Application

Provides multi-stage retrieval with hybrid search, cross-encoder
reranking, and dependency expansion for context-aware code retrieval.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cis.models.schemas import (
    ContextQueryRequest,
    ContextQueryResponse,
    ChunkResponse,
    RetrievalStages,
    ContextMetadata,
    UpdateRequest,
    UpdateResponse,
    HealthResponse,
    HealthStats,
    HealthPerformance,
)
from cis.core.hybrid_index import HybridIndex, SearchResult
from cis.core.reranker import CrossEncoderReranker
from cis.core.dependency_graph import DependencyGraph
from cis.core.completeness import CompletenessScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Content Intelligence Service",
    description="Multi-stage retrieval with hybrid search and cross-encoder reranking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (would be dependency injected in production)
hybrid_index: Optional[HybridIndex] = None
reranker: Optional[CrossEncoderReranker] = None
dependency_graph: Optional[DependencyGraph] = None
completeness_scorer: Optional[CompletenessScorer] = None

# Metrics
query_times: List[float] = []


@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    global hybrid_index, reranker, dependency_graph, completeness_scorer

    logger.info("Initializing Content Intelligence Service...")

    # Initialize components (lazy - full init requires data)
    hybrid_index = HybridIndex(
        model_name="microsoft/codebert-base",
        dimension=768
    )
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k=10
    )
    dependency_graph = DependencyGraph(max_depth=2)
    completeness_scorer = CompletenessScorer(min_threshold=0.9)

    logger.info("Content Intelligence Service ready")


@app.post("/v1/context/query", response_model=ContextQueryResponse)
async def context_query(request: ContextQueryRequest):
    """
    Multi-stage context retrieval.

    1. Query analysis and expansion
    2. Hybrid search (FAISS + BM25)
    3. Cross-encoder reranking
    4. Dependency expansion
    5. Completeness scoring
    """
    start_time = time.time()
    context_id = str(uuid.uuid4())[:8]

    try:
        # Stage 1: Hybrid search
        t1 = time.time()
        candidates = hybrid_index.search(
            request.query,
            k=50,
            alpha=0.5
        )
        hybrid_time = int((time.time() - t1) * 1000)

        # Stage 2: Cross-encoder reranking
        t2 = time.time()
        candidate_tuples = [
            (r.chunk_id, r.score, r.content)
            for r in candidates
        ]
        reranked = reranker.rerank(
            request.query,
            candidate_tuples,
            top_k=request.options.dependency_depth * 5
        )
        rerank_time = int((time.time() - t2) * 1000)

        # Stage 3: Dependency expansion
        t3 = time.time()
        expanded_ids = set()
        for result in reranked:
            deps = dependency_graph.get_dependencies(
                result.chunk_id,
                max_depth=request.options.dependency_depth
            )
            expanded_ids.update(deps)
        expand_time = int((time.time() - t3) * 1000)

        # Build chunks response
        chunks = []
        chunk_dicts = []
        for result in reranked:
            chunk_data = hybrid_index.chunk_store.get(result.chunk_id, {})
            chunks.append(ChunkResponse(
                file=chunk_data.get("file_path", ""),
                lines=f"{chunk_data.get('start_line', 0)}-{chunk_data.get('end_line', 0)}",
                relevance_score=result.score,
                chunk_type=chunk_data.get("chunk_type", ""),
                symbol=chunk_data.get("symbol_name", ""),
                content=result.content,
                dependencies=list(expanded_ids)[:5]
            ))
            chunk_dicts.append({
                "chunk_id": result.chunk_id,
                "content": result.content,
                "symbol_name": chunk_data.get("symbol_name", "")
            })

        # Calculate completeness
        completeness = completeness_scorer.score(chunk_dicts, list(expanded_ids))

        # Estimate tokens (rough: 4 chars per token)
        total_chars = sum(len(c.content) for c in chunks)
        token_count = total_chars // 4

        total_time = int((time.time() - start_time) * 1000)
        query_times.append(total_time)

        return ContextQueryResponse(
            context_id=context_id,
            token_count=token_count,
            completeness_score=completeness.overall_score,
            retrieval_time_ms=total_time,
            chunks=chunks,
            dependency_tree={},
            metadata=ContextMetadata(
                retrieval_stages=RetrievalStages(
                    hybrid_search_ms=hybrid_time,
                    reranking_ms=rerank_time,
                    dependency_expansion_ms=expand_time
                ),
                total_files=len(set(c.file for c in chunks)),
                patterns_detected=[]
            )
        )

    except Exception as e:
        logger.error(f"Context query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/context/update", response_model=UpdateResponse)
async def context_update(request: UpdateRequest):
    """
    Incremental update with red-green marking.
    """
    start_time = time.time()

    try:
        dirty_nodes = set()

        if request.action == "delete":
            # Remove from index
            # (implementation would remove from hybrid_index and graph)
            pass
        else:
            # Update content
            if request.content:
                dirty_nodes = dependency_graph.update_node(
                    request.file_path,
                    request.content
                )

        processing_time = int((time.time() - start_time) * 1000)

        return UpdateResponse(
            status="updated",
            affected_files=1,
            reindexed_chunks=1,
            dirty_nodes=len(dirty_nodes),
            processing_time_ms=processing_time,
            changes={
                "modified_symbols": [],
                "added_symbols": [],
                "deleted_symbols": []
            }
        )

    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Health check and statistics."""
    # Calculate percentiles
    sorted_times = sorted(query_times) if query_times else [0]
    p50_idx = len(sorted_times) // 2
    p95_idx = int(len(sorted_times) * 0.95)
    p99_idx = int(len(sorted_times) * 0.99)

    return HealthResponse(
        status="healthy",
        stats=HealthStats(
            total_files=0,
            total_chunks=len(hybrid_index.chunk_store) if hybrid_index else 0,
            index_size_mb=0.0,
            last_update=None,
            avg_query_time_ms=sum(query_times) / len(query_times) if query_times else 0,
            cache_hit_rate=0.0,
            completeness_avg=0.0
        ),
        performance=HealthPerformance(
            p50_latency_ms=int(sorted_times[p50_idx]),
            p95_latency_ms=int(sorted_times[min(p95_idx, len(sorted_times) - 1)]),
            p99_latency_ms=int(sorted_times[min(p99_idx, len(sorted_times) - 1)])
        )
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Content Intelligence Service",
        "version": "1.0.0",
        "status": "running"
    }
