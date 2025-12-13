"""
Pydantic models for LSS API requests and responses.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Request Models ---

class SummarizeRequest(BaseModel):
    """Request for summarization endpoint."""
    content: str = Field(..., min_length=1)
    summary_type: str = Field(
        default="abstractive",
        pattern="^(extractive|abstractive|hierarchical|custom)$"
    )
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Optional fields
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_template: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    context: Optional[Dict[str, Any]] = None


class BatchItem(BaseModel):
    """Single item in batch request."""
    content: str
    context: Optional[str] = None
    id: Optional[str] = None


class BatchSummarizeRequest(BaseModel):
    """Request for batch summarization."""
    items: List[BatchItem]
    summary_type: str = "abstractive"
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PromptCreateRequest(BaseModel):
    """Request to create custom prompt."""
    id: str = Field(..., min_length=1)
    name: str
    template: str
    summary_type: str = "custom"
    description: str = ""
    variables: List[str] = Field(default_factory=list)


# --- Response Models ---

class SummarizeResponse(BaseModel):
    """Response for summarization."""
    summary: str
    quality_score: float
    tokens_used: int
    provider: str
    model: str
    latency_ms: int
    cached: bool = False


class BatchItemResult(BaseModel):
    """Result for single batch item."""
    id: Optional[str]
    summary: str
    quality_score: float
    tokens_used: int


class BatchSummarizeResponse(BaseModel):
    """Response for batch summarization."""
    results: List[BatchItemResult]
    total_tokens: int
    total_latency_ms: int


class StreamChunk(BaseModel):
    """Chunk in streaming response."""
    chunk: str
    done: bool
    summary: Optional[str] = None
    quality_score: Optional[float] = None


class PromptResponse(BaseModel):
    """Response for prompt operations."""
    id: str
    name: str
    summary_type: str
    description: str
    variables: List[str]


class ProviderInfo(BaseModel):
    """Information about an LLM provider."""
    name: str
    model: str
    is_healthy: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    providers: List[ProviderInfo]
    prompt_count: int
    cache_size: int
