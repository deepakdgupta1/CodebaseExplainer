"""
Configuration for Content Intelligence Service.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "microsoft/codebert-base"
    dimension: int = 768
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class RerankerConfig:
    """Cross-encoder reranker configuration."""
    model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    enabled: bool = True
    top_k: int = 10
    batch_size: int = 32


@dataclass
class SearchConfig:
    """Hybrid search configuration."""
    hybrid_enabled: bool = True
    alpha: float = 0.5  # Weight for vector search
    top_k_candidates: int = 50
    final_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class FAISSConfig:
    """FAISS index configuration."""
    index_type: str = "HNSW"
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 128


@dataclass 
class DependencyGraphConfig:
    """Dependency graph configuration."""
    max_depth: int = 2
    track_imports: bool = True
    track_calls: bool = True
    track_inheritance: bool = True


@dataclass
class ContextAssemblyConfig:
    """Context assembly configuration."""
    max_tokens: int = 4096
    min_completeness: float = 0.9
    prioritize_base_classes: bool = True
    recency_weight: float = 0.2


@dataclass
class ChunkingConfig:
    """AST chunking configuration."""
    strategy: str = "ast_aware"
    min_chunk_lines: int = 5
    include_docstrings: bool = True
    include_decorators: bool = True


@dataclass
class CacheConfig:
    """Cache paths configuration."""
    base_path: Path = field(default_factory=lambda: Path(".context_cache"))
    
    @property
    def ast_cache_path(self) -> Path:
        return self.base_path / "ast.db"
    
    @property
    def index_path(self) -> Path:
        return self.base_path / "index"
    
    @property
    def graph_path(self) -> Path:
        return self.base_path / "deps.gpickle"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8081
    workers: int = 1
    log_level: str = "info"


@dataclass
class CISConfig:
    """Main configuration for Content Intelligence Service."""
    server: ServerConfig = field(default_factory=ServerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    dependency_graph: DependencyGraphConfig = field(default_factory=DependencyGraphConfig)
    context_assembly: ContextAssemblyConfig = field(default_factory=ContextAssemblyConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "CISConfig":
        """Create config from dictionary."""
        return cls(
            server=ServerConfig(**data.get("server", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            reranker=RerankerConfig(**data.get("reranker", {})),
            search=SearchConfig(**data.get("search", {})),
            faiss=FAISSConfig(**data.get("faiss", {})),
            dependency_graph=DependencyGraphConfig(**data.get("dependency_graph", {})),
            context_assembly=ContextAssemblyConfig(**data.get("context_assembly", {})),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            cache=CacheConfig(base_path=Path(data.get("cache", {}).get("base_path", ".context_cache"))),
        )
