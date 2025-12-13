"""
LLM Summarization Service - FastAPI Application

Provides LLM-based summarization with configurable providers,
prompt templates, and summary types.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from lss.models.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    BatchSummarizeRequest,
    BatchSummarizeResponse,
    BatchItemResult,
    PromptCreateRequest,
    PromptResponse,
    HealthResponse,
    ProviderInfo,
)
from lss.backends import OpenAIBackend, AnthropicBackend, BaseLLMBackend
from lss.core import PromptRegistry, PromptTemplate, SummaryEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LLM Summarization Service",
    description="LLM-based summarization with configurable providers and prompts",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
backends: dict[str, BaseLLMBackend] = {}
prompt_registry: Optional[PromptRegistry] = None
summary_engine: Optional[SummaryEngine] = None


@app.on_event("startup")
async def startup():
    """Initialize service components."""
    global backends, prompt_registry, summary_engine
    
    logger.info("Initializing LLM Summarization Service...")
    
    # Initialize prompt registry
    prompt_registry = PromptRegistry()
    
    # Initialize backends based on available config
    # Default to OpenAI-compatible local server
    default_backend = OpenAIBackend(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
        model=os.getenv("LLM_MODEL", "local-model")
    )
    backends["default"] = default_backend
    backends["openai"] = default_backend
    
    # Add Anthropic if key available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        backends["anthropic"] = AnthropicBackend(
            api_key=anthropic_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        )
    
    # Initialize summary engine with default backend
    summary_engine = SummaryEngine(default_backend, prompt_registry)
    
    logger.info("LLM Summarization Service ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    for backend in backends.values():
        if hasattr(backend, 'close'):
            await backend.close()


@app.post("/v1/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Generate a summary for content.

    Supports multiple summary types:
    - extractive: Key points extraction
    - abstractive: Natural language summary
    - hierarchical: Multi-level summary
    - custom: Using custom prompt template
    """
    try:
        # Select backend
        backend = backends.get(request.provider or "default")
        if not backend:
            raise HTTPException(404, f"Provider not found: {request.provider}")
        
        # Create engine with selected backend
        engine = SummaryEngine(backend, prompt_registry)
        
        # Generate summary
        result = await engine.summarize(
            content=request.content,
            summary_type=request.summary_type,
            prompt_id=request.prompt_template,
            context=str(request.context) if request.context else None,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return SummarizeResponse(
            summary=result.summary,
            quality_score=result.quality_score,
            tokens_used=result.tokens_used,
            provider=result.provider,
            model=result.model,
            latency_ms=result.latency_ms,
            cached=result.cached
        )
    
    except KeyError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/v1/summarize/stream")
async def summarize_stream(request: SummarizeRequest):
    """
    Stream summary generation.

    Returns Server-Sent Events with incremental text.
    """
    backend = backends.get(request.provider or "default")
    if not backend:
        raise HTTPException(404, f"Provider not found: {request.provider}")
    
    engine = SummaryEngine(backend, prompt_registry)
    
    async def generate():
        full_text = ""
        async for chunk in engine.summarize_stream(
            content=request.content,
            summary_type=request.summary_type,
            prompt_id=request.prompt_template,
            context=str(request.context) if request.context else None,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        ):
            full_text += chunk
            yield f"data: {{'chunk': '{chunk}', 'done': false}}\n\n"
        
        yield f"data: {{'chunk': '', 'done': true, 'summary': '{full_text}'}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/v1/summarize/batch", response_model=BatchSummarizeResponse)
async def summarize_batch(request: BatchSummarizeRequest):
    """Summarize multiple items."""
    backend = backends.get("default")
    engine = SummaryEngine(backend, prompt_registry)
    
    results = []
    total_tokens = 0
    total_latency = 0
    
    for item in request.items:
        result = await engine.summarize(
            content=item.content,
            context=item.context,
            summary_type=request.summary_type
        )
        results.append(BatchItemResult(
            id=item.id,
            summary=result.summary,
            quality_score=result.quality_score,
            tokens_used=result.tokens_used
        ))
        total_tokens += result.tokens_used
        total_latency += result.latency_ms
    
    return BatchSummarizeResponse(
        results=results,
        total_tokens=total_tokens,
        total_latency_ms=total_latency
    )


@app.get("/v1/prompts")
async def list_prompts():
    """List available prompt templates."""
    prompts = prompt_registry.list_prompts()
    return [
        PromptResponse(
            id=p.id,
            name=p.name,
            summary_type=p.summary_type,
            description=p.description,
            variables=p.variables
        )
        for p in prompts
    ]


@app.post("/v1/prompts", response_model=PromptResponse)
async def create_prompt(request: PromptCreateRequest):
    """Register a custom prompt template."""
    template = PromptTemplate(
        id=request.id,
        name=request.name,
        template=request.template,
        summary_type=request.summary_type,
        description=request.description,
        variables=request.variables
    )
    prompt_registry.register(template)
    
    return PromptResponse(
        id=template.id,
        name=template.name,
        summary_type=template.summary_type,
        description=template.description,
        variables=template.variables
    )


@app.get("/v1/providers")
async def list_providers():
    """List available LLM providers."""
    providers = []
    for name, backend in backends.items():
        is_healthy = await backend.is_healthy()
        providers.append(ProviderInfo(
            name=name,
            model=backend.model_id,
            is_healthy=is_healthy
        ))
    return providers


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Health check."""
    providers = []
    for name, backend in backends.items():
        try:
            is_healthy = await backend.is_healthy()
        except Exception:
            is_healthy = False
        providers.append(ProviderInfo(
            name=name,
            model=backend.model_id,
            is_healthy=is_healthy
        ))
    
    return HealthResponse(
        status="healthy" if any(p.is_healthy for p in providers) else "degraded",
        providers=providers,
        prompt_count=len(prompt_registry.list_prompts()),
        cache_size=len(summary_engine._cache) if summary_engine else 0
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Summarization Service",
        "version": "1.0.0",
        "status": "running"
    }
