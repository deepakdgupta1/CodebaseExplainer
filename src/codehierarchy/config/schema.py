from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class SystemConfig(BaseModel):
    max_memory_gb: float = Field(default=12.0, description="Maximum memory usage in GB")
    temp_dir: Path = Field(default=Path(".codehierarchy/temp"), description="Directory for temporary files")
    output_dir: Path = Field(default=Path("output"), description="Default output directory")
    checkpointing_enabled: bool = Field(default=True, description="Enable checkpointing for long processes")
    checkpoint_interval: int = Field(default=100, description="Save checkpoint every N batches")

class ParsingConfig(BaseModel):
    languages: List[str] = Field(default=["python", "typescript"], description="Languages to parse")
    num_workers: int = Field(default=6, description="Number of parallel parsing workers")
    timeout_seconds: int = Field(default=60, description="Timeout per file in seconds")
    max_file_size_mb: float = Field(default=10.0, description="Max file size to process in MB")
    exclude_patterns: List[str] = Field(
        default=[
            "**/node_modules/**", 
            "**/venv/**", 
            "**/__pycache__/**", 
            "**/.git/**", 
            "**/dist/**", 
            "**/build/**"
        ], 
        description="Glob patterns to exclude"
    )

class GraphConfig(BaseModel):
    storage_mode: Literal["in_memory", "disk"] = Field(default="in_memory", description="Graph storage mode")
    cache_size_gb: float = Field(default=2.0, description="Max size for source code cache")
    compute_metrics: bool = Field(default=True, description="Compute centrality and other metrics")

class LLMConfig(BaseModel):
    model_name: str = Field(default="deepseek-coder-v2:16b-q4_K_M", description="Ollama model tag")
    context_window: int = Field(default=128000, description="Context window size in tokens")
    batch_size: int = Field(default=20, description="Number of nodes per LLM call")
    temperature: float = Field(default=0.2, description="Generation temperature")
    num_thread: int = Field(default=8, description="Number of CPU threads for Ollama")
    timeout_seconds: int = Field(default=300, description="Timeout per batch in seconds")
    max_retries: int = Field(default=2, description="Max retries for failed calls")

class EmbeddingsConfig(BaseModel):
    model_name: str = Field(default="all-mpnet-base-v2", description="SentenceTransformer model name")
    dimension: int = Field(default=768, description="Embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")

class SearchConfig(BaseModel):
    default_mode: Literal["keyword", "semantic", "hybrid"] = Field(default="hybrid", description="Default search mode")
    index_type: str = Field(default="IVF256,Flat", description="FAISS index type")
    nprobe: int = Field(default=64, description="FAISS nprobe parameter")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion k constant")

class Config(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
