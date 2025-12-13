"""
Pydantic models for CIS API requests and responses.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Request Models ---

class ContextOptions(BaseModel):
    """Options for context query."""
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    min_completeness: float = Field(default=0.9, ge=0.0, le=1.0)
    include_dependencies: bool = True
    dependency_depth: int = Field(default=2, ge=1, le=5)
    include_tests: bool = False


class ContextInfo(BaseModel):
    """Current cursor context."""
    current_file: Optional[str] = None
    cursor_line: Optional[int] = None
    selected_text: Optional[str] = None


class ContextQueryRequest(BaseModel):
    """Request for context query endpoint."""
    query: str = Field(..., min_length=1)
    context: Optional[ContextInfo] = None
    options: ContextOptions = Field(default_factory=ContextOptions)


class ContentIngestRequest(BaseModel):
    """Request for content ingestion."""
    source_type: str = Field(..., pattern="^(file|url|text|stream)$")
    content: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)


class UpdateRequest(BaseModel):
    """Request for incremental update."""
    action: str = Field(..., pattern="^(modify|create|delete)$")
    file_path: str
    content: Optional[str] = None
    force_reindex: bool = False


# --- Response Models ---

class ChunkResponse(BaseModel):
    """A chunk in context response."""
    file: str
    lines: str
    relevance_score: float
    chunk_type: str
    symbol: str
    content: str
    dependencies: List[str] = Field(default_factory=list)


class RetrievalStages(BaseModel):
    """Timing for retrieval stages."""
    hybrid_search_ms: int
    reranking_ms: int
    dependency_expansion_ms: int


class ContextMetadata(BaseModel):
    """Metadata about context retrieval."""
    retrieval_stages: RetrievalStages
    total_files: int = 0
    patterns_detected: List[str] = Field(default_factory=list)


class ContextQueryResponse(BaseModel):
    """Response for context query."""
    context_id: str
    token_count: int
    completeness_score: float
    retrieval_time_ms: int
    chunks: List[ChunkResponse]
    dependency_tree: Dict[str, Any] = Field(default_factory=dict)
    metadata: ContextMetadata


class UpdateResponse(BaseModel):
    """Response for incremental update."""
    status: str
    affected_files: int
    reindexed_chunks: int
    dirty_nodes: int
    processing_time_ms: int
    changes: Dict[str, List[str]] = Field(default_factory=dict)


class HealthStats(BaseModel):
    """Index statistics."""
    total_files: int
    total_chunks: int
    index_size_mb: float
    last_update: Optional[datetime] = None
    avg_query_time_ms: float
    cache_hit_rate: float
    completeness_avg: float


class HealthPerformance(BaseModel):
    """Performance metrics."""
    p50_latency_ms: int
    p95_latency_ms: int
    p99_latency_ms: int


class HealthResponse(BaseModel):
    """Response for health endpoint."""
    status: str
    stats: HealthStats
    performance: HealthPerformance
