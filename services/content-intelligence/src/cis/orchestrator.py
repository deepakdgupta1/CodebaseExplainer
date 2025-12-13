"""
Content Intelligence Service Orchestrator.

Coordinates all components: chunking, indexing, searching,
and persistence for the complete retrieval pipeline.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .config import CISConfig
from .core.hybrid_index import HybridIndex
from .core.reranker import CrossEncoderReranker
from .core.chunker import ASTChunker, CodeChunk
from .core.dependency_graph import DependencyGraph
from .core.completeness import CompletenessScorer
from .storage.ast_cache import ASTCache, ASTCacheEntry
from .storage.index_persistence import IndexPersistence


logger = logging.getLogger(__name__)


class CISOrchestrator:
    """
    Orchestrates the Content Intelligence Service.

    Coordinates:
    - File scanning and AST parsing
    - Chunk extraction and indexing
    - Hybrid search and reranking
    - Dependency tracking
    - Persistence

    Usage:
        orchestrator = CISOrchestrator(config)
        orchestrator.index_codebase("/path/to/code")
        results = orchestrator.query("How does auth work?")
    """

    def __init__(self, config: CISConfig) -> None:
        """
        Initialize the orchestrator.

        Args:
            config: Service configuration.
        """
        self.config = config
        
        # Initialize components
        self.chunker = ASTChunker(
            include_docstrings=config.chunking.include_docstrings,
            include_decorators=config.chunking.include_decorators,
            min_chunk_lines=config.chunking.min_chunk_lines
        )
        
        self.hybrid_index = HybridIndex(
            model_name=config.embedding.model,
            dimension=config.embedding.dimension,
            device=config.embedding.device
        )
        
        self.reranker = CrossEncoderReranker(
            model_name=config.reranker.model,
            top_k=config.reranker.top_k
        )
        
        self.dependency_graph = DependencyGraph(
            max_depth=config.dependency_graph.max_depth
        )
        
        self.completeness_scorer = CompletenessScorer(
            min_threshold=config.context_assembly.min_completeness
        )
        
        # Initialize storage
        self.ast_cache = ASTCache(config.cache.ast_cache_path)
        self.index_persistence = IndexPersistence(config.cache.index_path)
        
        # Try to load persisted indices
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        """Load previously persisted indices if available."""
        # Load FAISS index
        faiss_result = self.index_persistence.load_faiss_index()
        if faiss_result:
            index, id_to_idx, idx_to_id = faiss_result
            self.hybrid_index.faiss_index = index
            self.hybrid_index.id_to_idx = id_to_idx
            self.hybrid_index.idx_to_id = idx_to_id
            logger.info("Loaded persisted FAISS index")
        
        # Load BM25 index
        bm25_result = self.index_persistence.load_bm25_index()
        if bm25_result:
            bm25, corpus, ids = bm25_result
            self.hybrid_index.bm25_index = bm25
            self.hybrid_index.tokenized_corpus = corpus
            self.hybrid_index.bm25_ids = ids
            logger.info("Loaded persisted BM25 index")
        
        # Load chunk store
        chunk_store = self.index_persistence.load_chunk_store()
        if chunk_store:
            self.hybrid_index.chunk_store = chunk_store
            logger.info("Loaded persisted chunk store")
        
        # Load dependency graph
        graph_path = self.config.cache.graph_path
        if graph_path.exists():
            self.dependency_graph.load(graph_path)
            logger.info("Loaded persisted dependency graph")

    def index_codebase(
        self,
        root_path: Path,
        extensions: List[str] = [".py"],
        exclude_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Index a codebase for context retrieval.

        Args:
            root_path: Root path of codebase.
            extensions: File extensions to include.
            exclude_patterns: Glob patterns to exclude.

        Returns:
            Dict with indexing statistics.
        """
        start_time = time.time()
        exclude = exclude_patterns or ["**/venv/**", "**/__pycache__/**"]
        
        # Scan files
        all_chunks = []
        files_processed = 0
        files_skipped = 0
        
        for ext in extensions:
            for file_path in root_path.rglob(f"*{ext}"):
                # Check exclusions
                skip = False
                for pattern in exclude:
                    if file_path.match(pattern):
                        skip = True
                        break
                
                if skip:
                    files_skipped += 1
                    continue
                
                # Read file
                try:
                    content = file_path.read_text()
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue
                
                # Check if file changed
                fingerprint = self._compute_fingerprint(content)
                cached = self.ast_cache.get(str(file_path))
                
                if cached and cached.fingerprint == fingerprint:
                    # Use cached chunks (reconstruct from symbols)
                    files_skipped += 1
                    continue
                
                # Parse and chunk
                chunks = self.chunker.chunk_file(str(file_path), content)
                
                for chunk in chunks:
                    # Convert to dict for indexing
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "chunk_type": chunk.chunk_type,
                        "symbol_name": chunk.symbol_name,
                        "signature": chunk.signature,
                        "content": chunk.content,
                        "metadata": {
                            "docstring": chunk.docstring,
                            "imports": chunk.imports,
                            "calls": chunk.calls,
                            "inherits": chunk.inherits,
                        }
                    }
                    all_chunks.append(chunk_dict)
                    
                    # Add to dependency graph
                    self.dependency_graph.add_node(
                        node_id=chunk.chunk_id,
                        node_type=chunk.chunk_type,
                        file_path=chunk.file_path,
                        symbol_name=chunk.symbol_name,
                        content=chunk.content
                    )
                    
                    # Add edges for calls
                    for call in chunk.calls:
                        # Note: would need symbol resolution for full edges
                        pass
                
                # Update AST cache
                self.ast_cache.put(ASTCacheEntry(
                    file_path=str(file_path),
                    ast_json="",  # Would store full AST if needed
                    symbols=[{"name": c.symbol_name, "type": c.chunk_type} for c in chunks],
                    imports=[],
                    classes=[c.__dict__ for c in chunks if c.chunk_type == "class"],
                    functions=[c.__dict__ for c in chunks if c.chunk_type == "function"],
                    fingerprint=fingerprint,
                    last_updated=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    file_size_bytes=len(content)
                ))
                
                files_processed += 1
        
        # Build indices
        if all_chunks:
            logger.info(f"Building index for {len(all_chunks)} chunks...")
            self.hybrid_index.build_index(
                all_chunks,
                batch_size=self.config.embedding.batch_size
            )
            
            # Persist indices
            self._persist_state()
        
        elapsed = time.time() - start_time
        
        return {
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "chunks_indexed": len(all_chunks),
            "elapsed_seconds": elapsed
        }

    def query(
        self,
        query: str,
        max_tokens: int = None,
        min_completeness: float = None
    ) -> Dict[str, Any]:
        """
        Query for relevant context.

        Args:
            query: Search query.
            max_tokens: Token budget (optional).
            min_completeness: Completeness threshold (optional).

        Returns:
            Dict with chunks, scores, and metadata.
        """
        start_time = time.time()
        
        # Stage 1: Hybrid search
        t1 = time.time()
        candidates = self.hybrid_index.search(
            query,
            k=self.config.search.top_k_candidates,
            alpha=self.config.search.alpha
        )
        hybrid_time = int((time.time() - t1) * 1000)
        
        # Stage 2: Rerank
        t2 = time.time()
        candidate_tuples = [(r.chunk_id, r.score, r.content) for r in candidates]
        reranked = self.reranker.rerank(
            query,
            candidate_tuples,
            top_k=self.config.search.final_top_k
        )
        rerank_time = int((time.time() - t2) * 1000)
        
        # Stage 3: Dependency expansion
        t3 = time.time()
        expanded_ids = set()
        for result in reranked:
            deps = self.dependency_graph.get_dependencies(result.chunk_id)
            expanded_ids.update(deps)
        expand_time = int((time.time() - t3) * 1000)
        
        # Calculate completeness
        chunk_dicts = [
            {"chunk_id": r.chunk_id, "content": r.content, "symbol_name": ""}
            for r in reranked
        ]
        completeness = self.completeness_scorer.score(
            chunk_dicts, list(expanded_ids)
        )
        
        total_time = int((time.time() - start_time) * 1000)
        
        return {
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content": r.content
                }
                for r in reranked
            ],
            "completeness_score": completeness.overall_score,
            "retrieval_time_ms": total_time,
            "stages": {
                "hybrid_search_ms": hybrid_time,
                "reranking_ms": rerank_time,
                "expansion_ms": expand_time
            }
        }

    def update_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Incrementally update index for a changed file.

        Args:
            file_path: Path to changed file.
            content: New file content.

        Returns:
            Dict with update statistics.
        """
        start_time = time.time()
        
        fingerprint = self._compute_fingerprint(content)
        cached = self.ast_cache.get(file_path)
        
        if cached and cached.fingerprint == fingerprint:
            return {"status": "unchanged", "reindexed": 0}
        
        # Re-chunk
        chunks = self.chunker.chunk_file(file_path, content)
        
        # Mark dirty nodes
        dirty_nodes = set()
        for chunk in chunks:
            dirty = self.dependency_graph.update_node(
                chunk.chunk_id,
                chunk.content
            )
            dirty_nodes.update(dirty)
        
        # Update cache
        self.ast_cache.put(ASTCacheEntry(
            file_path=file_path,
            ast_json="",
            symbols=[{"name": c.symbol_name} for c in chunks],
            imports=[],
            classes=[],
            functions=[],
            fingerprint=fingerprint,
            last_updated=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            file_size_bytes=len(content)
        ))
        
        elapsed = int((time.time() - start_time) * 1000)
        
        return {
            "status": "updated",
            "reindexed": len(chunks),
            "dirty_nodes": len(dirty_nodes),
            "processing_time_ms": elapsed
        }

    def _persist_state(self) -> None:
        """Persist all indices and graphs."""
        # Save FAISS
        if self.hybrid_index.faiss_index:
            self.index_persistence.save_faiss_index(
                self.hybrid_index.faiss_index,
                self.hybrid_index.id_to_idx,
                self.hybrid_index.idx_to_id
            )
        
        # Save BM25
        if self.hybrid_index.bm25_index:
            self.index_persistence.save_bm25_index(
                self.hybrid_index.tokenized_corpus,
                self.hybrid_index.bm25_ids
            )
        
        # Save chunk store
        if self.hybrid_index.chunk_store:
            self.index_persistence.save_chunk_store(self.hybrid_index.chunk_store)
        
        # Save dependency graph
        self.dependency_graph.save(self.config.cache.graph_path)

    def _compute_fingerprint(self, content: str) -> str:
        """Compute SHA-256 fingerprint."""
        return hashlib.sha256(content.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_stats = self.ast_cache.get_stats()
        index_stats = self.index_persistence.get_index_stats()
        
        return {
            "ast_cache": cache_stats,
            "index": index_stats,
            "chunks": len(self.hybrid_index.chunk_store),
            "graph_nodes": len(self.dependency_graph.node_cache)
        }
