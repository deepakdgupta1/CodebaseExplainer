# Implementation Plan: CodeHierarchy Explainer

## Overview

This implementation plan details the construction of a high-performance codebase documentation and search system powered by DeepSeek Coder V2 16B (Q4), optimized for 12GB RAM environments. The system will analyze codebases (Python, TypeScript), build an in-memory dependency graph, generate LLM-powered summaries with batching, and provide hybrid keyword + semantic search capabilities.

**Target Performance:**
- 1M LOC repository processing in ~28 minutes
- 12GB peak memory usage
- 91% summary accuracy with 5% hallucination rate
- Sub-200ms hybrid search queries

## User Review Required

> [!IMPORTANT]
> **New Project Creation**
> This implementation creates an entirely new Python package from scratch in `/home/deeog/Desktop/CodebaseExplainer/`. All existing content (only `codehierarchy_design.tsx`) will remain unchanged. The new codebase will be created alongside it.

> [!IMPORTANT]
> **External Dependencies Required**
> 1. **Ollama with DeepSeek Model**: The user must have Ollama installed and the `deepseek-coder-v2:16b-q4_K_M` model pulled before testing Phase 4 (LLM Summarization)
> 2. **Tree-Sitter Grammars**: Python and TypeScript tree-sitter grammars will be automatically downloaded/compiled during first run
> 3. **System Requirements**: 12GB RAM minimum for full-scale processing

> [!WARNING]
> **Batch Size vs. Quality Tradeoff**
> The design specifies batch size of 20 nodes per LLM call for throughput. This may impact quality if unrelated nodes are batched together. The smart batching algorithm groups by module, but users should be aware they can reduce batch size (at cost of speed) if quality issues arise.

> [!CAUTION]
> **Memory Management**
> The in-memory graph storage assumes entire codebase fits in ~3GB. For extremely large monorepos (>2M LOC), this may require disk-backed graph storage instead. The implementation includes a fallback mechanism, but performance will degrade significantly.

---

## Proposed Changes

### Component 1: Project Foundation & Configuration

Foundation setup including project structure, dependency management, and configuration system.

#### [NEW] `pyproject.toml`
- Package metadata and build configuration
- Core dependencies: `tree-sitter`, `tree-sitter-python`, `tree-sitter-typescript`, `networkx`, `ollama`, `sentence-transformers`, `faiss-cpu`, `pyyaml`, `pydantic`, `rich`, `click`, `psutil`
- Dev dependencies: `pytest`, `pytest-cov`, `black`, `mypy`, `ruff`
- Entry point: `codehierarchy = codehierarchy.cli.cli:main`

#### [NEW] `requirements.txt`
- Generated from `pyproject.toml` for pip compatibility

#### [NEW] `.gitignore`
- Python-specific ignore patterns
- Ignore `.codehierarchy/` temp directory, `*.pyc`, `__pycache__/`, `.pytest_cache/`, etc.

#### [NEW] `config/config.yaml`
- Default configuration matching design specification
- System: 12GB max memory, temp/output directories, checkpointing enabled
- Parsing: 6 workers, 10MB max file size, 60s timeout, Python + TypeScript support
- Graph: in-memory storage, 2GB cache, metrics enabled
- LLM: DeepSeek V2 16B Q4, 128K context, batch size 20, temperature 0.2
- Embeddings: MPNet 768-dim, batch size 32
- Search: hybrid mode default, IVF index with 256 clusters, nprobe 64

#### [NEW] `config/prompts/deepseek-optimized.txt`
- System prompt template for DeepSeek Coder V2
- Instructions for batch summarization with component analysis
- Format: explains purpose, inputs/outputs, dependencies, usage patterns

#### [NEW] `config/prompts/onboarding.txt`
- Alternative prompt focused on onboarding new developers
- Emphasizes "common gotchas" and practical usage examples

#### [NEW] `src/codehierarchy/config/schema.py`
- Pydantic models for configuration validation
- `SystemConfig`, `ParsingConfig`, `GraphConfig`, `LLMConfig`, `EmbeddingsConfig`, `SearchConfig`, `OutputConfig`, `PerformanceConfig`
- Type-safe configuration with validation rules

#### [NEW] `src/codehierarchy/config/loader.py`
- `load_config(path: Path) -> Config`: Load and validate YAML
- `load_prompt_template(variant: str) -> str`: Load prompt from file
- Error handling for missing/invalid configuration

---

### Component 2: Parallel Parser (Multi-threaded AST Extraction)

High-throughput parser using tree-sitter with 6 parallel workers for Python and TypeScript.

#### [NEW] `src/codehierarchy/parser/tree_sitter_parser.py`
- `TreeSitterParser` class
- `__init__(language: str)`: Initialize tree-sitter parser for Python or TypeScript
- `parse_bytes(content: bytes) -> Tree`: Parse source code into AST
- Automatic grammar download/compilation on first use

#### [NEW] `src/codehierarchy/parser/node_extractor.py`
- `NodeInfo` dataclass: type, name, line, end_line, source_code, docstring, signature, complexity, loc
- `extract_all_nodes(tree: Tree, language: str, source: str) -> List[NodeInfo]`: Extract functions, classes, methods
- Language-specific tree-sitter queries for Python (function_definition, class_definition)
- Language-specific tree-sitter queries for TypeScript (function_declaration, class_declaration, method_definition)
- Docstring/JSDoc extraction

#### [NEW] `src/codehierarchy/parser/call_graph_analyzer.py`
- `Edge` dataclass: source, target, type (call/import/inheritance), confidence
- `CallGraphAnalyzer` class
- `analyze(file: Path, tree: Tree) -> List[Edge]`: Extract all edges from AST
- `_extract_calls()`: Find function/method invocations
- `_extract_imports()`: Find import statements
- `_extract_inheritance()`: Find class inheritance relationships
- Confidence scoring: direct call=1.0, import=0.8, inheritance=0.9

#### [NEW] `src/codehierarchy/parser/complexity.py`
- `compute_cyclomatic_complexity(tree: Tree, language: str) -> int`: Count decision points (if, for, while, etc.)
- `compute_loc(source: str) -> int`: Count non-blank, non-comment lines
- `compute_comment_ratio(source: str) -> float`: Ratio of comment lines to total lines

#### [NEW] `src/codehierarchy/parser/parallel_parser.py`
- `ParseResult` dataclass: nodes, edges, complexity, error, skipped
- `ParallelParser` class
- `__init__(num_workers: int = 6)`: Initialize with worker pool
- `parse_repository(files: List[Path]) -> Dict[Path, ParseResult]`: Main entry point
- Uses `ProcessPoolExecutor` for parallel processing
- `_parse_file(file: Path) -> ParseResult`: Parse single file with timeout (30s)
- Error handling and logging for failed parses

#### [NEW] `src/codehierarchy/scanner/file_scanner.py`
- `FileScanner` class
- `scan_directory(root: Path, config: ParsingConfig) -> List[Path]`: Multi-threaded directory traversal
- Filter by file extensions: `.py`, `.ts`, `.js`, `.tsx`, `.jsx`
- Respect `.gitignore` patterns using `pathspec` library
- Filter by max file size (default 10MB)

#### [NEW] `src/codehierarchy/utils/language_detector.py`
- `detect_language(file: Path) -> str`: Detect language from file extension
- Returns `"python"`, `"typescript"`, or `None`

---

### Component 3: In-Memory Graph Builder (NetworkX)

Build dependency graph in memory with full source code caching and graph metrics.

#### [NEW] `src/codehierarchy/graph/graph_builder.py`
- `InMemoryGraphBuilder` class
- `__init__()`: Initialize `nx.DiGraph()`, `node_cache`, `metadata` dictionaries
- `build_from_results(results: Dict[Path, ParseResult]) -> nx.DiGraph`: Main builder
  - **Phase 1**: Add all nodes with unique IDs (`{file}:{name}:{line}`)
  - **Phase 2**: Add all edges with type and confidence weight
  - **Phase 3**: Compute graph metrics (centrality, critical paths)
- `get_node_with_context(node_id: str, depth: int = 2) -> dict`: Retrieve node with parent/child context
- `_compute_centrality()`: Calculate PageRank scores
- `_identify_critical_paths()`: Mark high-centrality nodes as critical

#### [NEW] `src/codehierarchy/graph/utils.py`
- `get_module_subgraph(graph: nx.DiGraph, module: str) -> nx.DiGraph`: Extract module-specific subgraph
- `get_dependency_chain(graph: nx.DiGraph, node_id: str) -> List[str]`: Get all dependencies recursively
- `export_graph(graph: nx.DiGraph, output_path: Path, format: str)`: Export as GraphML or DOT for visualization

---

### Component 4: DeepSeek LLM Summarizer (Batch Processing)

LLM integration with batch processing (20 nodes/call) and smart context injection.

#### [NEW] `src/codehierarchy/llm/deepseek_summarizer.py`
- `DeepSeekSummarizer` class
- `__init__(config: LLMConfig, prompt_template: str)`
- `summarize_batch(nodes: List[Node], graph: nx.DiGraph) -> List[str]`: Main entry point
- `_create_smart_batches(nodes: List[Node], batch_size: int) -> List[List[Node]]`: Group by module/file for context relevance
- `_build_batch_prompt(batch: List[Node], contexts: List[dict]) -> str`: Construct prompt with full source code + context
- Ollama API call via `ollama.chat()` with options: `num_ctx=128000`, `temperature=0.2`, `top_p=0.95`, `num_thread=8`
- `_parse_batch_response(response, batch) -> List[str]`: Extract summaries using `[COMPONENT_ID]` markers
- Retry logic (max 2 retries) with exponential backoff
- Timeout handling (5 minutes per batch)

#### [NEW] `src/codehierarchy/llm/validator.py`
- `validate_summary(summary: str, node: NodeInfo, min_length: int = 100, max_length: int = 600) -> bool`
- Check length constraints
- Detect hallucination patterns (references to non-existent functions/classes)
- Check for key information presence (purpose, parameters, return values)
- Return validation result + quality score

#### [NEW] `src/codehierarchy/llm/checkpoint.py`
- `save_checkpoint(summaries: Dict[str, str], checkpoint_file: Path)`: Save progress as JSON
- `load_checkpoint(checkpoint_file: Path) -> Dict[str, str]`: Load existing checkpoint
- Automatic saving every 100 batches
- Resume from checkpoint on failure

---

### Component 5: Search & Embedding System (Hybrid Search)

High-quality embeddings (MPNet 768-dim) with FAISS indexing and hybrid search (keyword + semantic).

#### [NEW] `src/codehierarchy/search/embedder.py`
- `HighQualityEmbedder` class
- `__init__()`: Load `all-mpnet-base-v2` model (768-dim)
- `encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`: Generate embeddings with normalization
- `build_index(summaries: Dict[str, str]) -> faiss.Index`: Create IVF index with 256 clusters
- `save_index(index: faiss.Index, mapping: dict, path: Path)`: Persist index to disk
- `load_index(path: Path) -> Tuple[faiss.Index, dict]`: Load persisted index

#### [NEW] `src/codehierarchy/search/keyword_search.py`
- `KeywordSearchIndex` class
- Uses SQLite FTS5 for full-text search
- `build_index(summaries: Dict[str, str], graph: nx.DiGraph)`: Index node IDs, names, and summaries
- `search(query: str, top_k: int) -> List[Tuple[str, float]]`: BM25-ranked results

#### [NEW] `src/codehierarchy/search/search_engine.py`
- `EnterpriseSearchEngine` class
- `__init__(index_dir: Path)`: Load keyword and vector indices
- `search(query: str, mode: str = 'hybrid', top_k: int = 20) -> List[Result]`: Main entry point
- `_keyword_search(query: str, top_k: int) -> List[Result]`: FTS5 search
- `_semantic_search(query: str, top_k: int) -> List[Result]`: FAISS vector search with nprobe=64
- `_hybrid_search(query: str, top_k: int) -> List[Result]`: Reciprocal Rank Fusion (RRF) with k=60
- `_generate_snippet(node: dict, query: str) -> str`: Create context snippet with highlighting
- `_explain_match(node: dict, query: str) -> str`: Optional LLM explanation of why result matches

#### [NEW] `src/codehierarchy/search/result.py`
- `Result` dataclass: node_id, name, file, line, summary, score, snippet, explanation

---

### Component 6: CLI & Pipeline Orchestration

Command-line interface and pipeline orchestrator for end-to-end execution.

#### [NEW] `src/codehierarchy/cli/cli.py`
- `click` application with two commands:
- `analyze`: Analyze repository and generate documentation
  - Options: `--config`, `--output`, `--workers`, `--batch-size`, `--checkpoint`, `--verbose`
- `search`: Search indexed codebase
  - Options: `--mode` (keyword/semantic/hybrid), `--top-k`, `--index-dir`
- Rich progress bars and formatted output

#### [NEW] `src/codehierarchy/pipeline/orchestrator.py`
- `PipelineOrchestrator` class
- `run_pipeline(repo_path: Path, config: Config) -> dict`: Execute full pipeline
  - **Phase 1**: Scan files (FileScanner) - 2 min target
  - **Phase 2**: Parse files (ParallelParser) - 4 min target
  - **Phase 3**: Build graph (InMemoryGraphBuilder) - 3 min target
  - **Phase 4**: Generate summaries (DeepSeekSummarizer) - 15 min target
  - **Phase 5**: Build search index (Embedder + SearchEngine) - 4 min target
- Progress tracking with `rich.progress`
- Performance metrics collection (time, memory, throughput)
- Error handling and graceful degradation

#### [NEW] `src/codehierarchy/output/markdown_generator.py`
- `MarkdownGenerator` class
- `generate_documentation(graph: nx.DiGraph, summaries: Dict[str, str], output_dir: Path)`
- Create hierarchy tree view (indented structure by module/file)
- For each node: output name, type, location, summary, complexity, LOC
- Generate call graph diagrams using Mermaid syntax
- Create metrics tables (most complex functions, highest centrality, etc.)
- Generate table of contents with links

#### [NEW] `src/codehierarchy/utils/profiler.py`
- `Profiler` class for performance monitoring
- `track_memory()`: Current memory usage via `psutil`
- `track_phase(name: str)`: Context manager for timing phases
- `save_metrics(path: Path)`: Write JSON report with timing, memory, throughput
- `check_memory_budget(max_gb: float)`: Warn if exceeding budget

#### [NEW] `src/codehierarchy/utils/logger.py`
- Configure Python logging with `rich` handler
- Log levels: DEBUG, INFO, WARNING, ERROR
- Dual output: console (with colors) and file (`codehierarchy.log`)
- Structured logging for metrics

---

### Component 7: Comprehensive Testing

Unit tests for all components plus integration test for full pipeline.

#### [NEW] `tests/parser/test_tree_sitter_parser.py`
- Test Python parsing with valid/invalid syntax
- Test TypeScript parsing with valid/invalid syntax
- Test error handling

#### [NEW] `tests/parser/test_node_extractor.py`
- Test function extraction from sample Python code
- Test class extraction with methods
- Test docstring extraction
- Test TypeScript function/class extraction

#### [NEW] `tests/parser/test_call_graph_analyzer.py`
- Test call edge detection
- Test import edge detection
- Test inheritance edge detection
- Test confidence scoring

#### [NEW] `tests/parser/test_parallel_parser.py`
- Test parallel parsing of multiple files
- Test timeout handling
- Test error recovery

#### [NEW] `tests/graph/test_graph_builder.py`
- Test node addition with unique IDs
- Test edge addition with weights
- Test context retrieval
- Test centrality computation
- Test caching

#### [NEW] `tests/llm/test_deepseek_summarizer.py`
- Mock Ollama API calls
- Test batch creation (smart grouping by module)
- Test prompt construction with context
- Test response parsing
- Test validation
- Test checkpointing

#### [NEW] `tests/search/test_embedder.py`
- Test embedding generation
- Test FAISS index creation
- Test index save/load

#### [NEW] `tests/search/test_search_engine.py`
- Test keyword search
- Test semantic search
- Test hybrid search with RRF
- Test result ranking accuracy

#### [NEW] `tests/integration/test_pipeline.py`
- Create fixture: small test repository (10-15 Python files with 500 LOC)
- Test full pipeline: scan → parse → graph → summarize → index
- Verify output documentation is generated 
- Verify search index is created and functional
- Verify performance metrics (should complete in <1 minute for small repo)

#### [NEW] `tests/config/test_config_loader.py`
- Test YAML loading
- Test Pydantic validation
- Test default values
- Test invalid config error handling

#### [NEW] `tests/conftest.py`
- Shared pytest fixtures
- Fixture: sample Python code snippets
- Fixture: sample TypeScript code snippets
- Fixture: mock Ollama client

---

### Component 8: Documentation & Project Files

User-facing documentation and packaging configuration.

#### [NEW] `README.md`
- Project overview and key features
- Installation instructions (pip install, Ollama setup)
- Quick start guide with examples
- Configuration guide (how to customize config.yaml)
- Architecture diagram (text-based or Mermaid)
- Performance benchmarks table
- Troubleshooting section (common issues)

#### [NEW] `docs/api.md`
- API documentation for programmatic usage
- Examples: using ParallelParser, InMemoryGraphBuilder, DeepSeekSummarizer directly
- Configuration schema reference

#### [NEW] `CONTRIBUTING.md`
- Development setup instructions
- Testing guidelines (run pytest, coverage requirements)
- Code style (black, mypy)
- PR process

#### [NEW] `LICENSE`
- MIT License (or user's choice)

---

## Verification Plan

The verification strategy combines automated tests, performance benchmarks, and manual validation.

### Automated Tests

All automated tests can be run with the following commands from the project root:

```bash
# Install package in development mode
pip install -e .

# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/codehierarchy --cov-report=html

# Run specific test modules
pytest tests/parser/ -v
pytest tests/graph/ -v  
pytest tests/llm/ -v
pytest tests/search/ -v

# Run integration test
pytest tests/integration/test_pipeline.py -v -s
```

**Coverage Target**: ≥80% code coverage across all modules

**Test Matrix**:
- Parser tests: Verify AST extraction, edge detection, parallel processing
- Graph tests: Verify graph building, context retrieval, metrics computation
- LLM tests: Verify batching logic, prompt construction, response parsing (with mocked API)
- Search tests: Verify keyword, semantic, and hybrid search accuracy
- Integration test: Verify full pipeline on small test repository (500 LOC)

### Performance Benchmarks

Performance will be validated using a medium-sized test repository (~50K LOC) to ensure scalability without requiring a full 1M LOC repo:

```bash
# Analyze test repository with profiling enabled
codehierarchy analyze ./test_repos/medium_repo \
  --config config/config.yaml \
  --output ./test_output \
  --verbose

# Check performance metrics
cat ./test_output/performance-metrics.json
```

**Success Criteria** (scaled for 50K LOC):
- Total processing time: <3 minutes
- Peak memory usage: <6GB
- Parsing throughput: >2000 files/sec
- LLM summarization: >1000 nodes/sec
- Search latency: <200ms per query

**Metrics to Verify**:
- Phase timings (scan, parse, graph, summarize, index)
- Memory usage per phase
- Batch processing efficiency (nodes per LLM call)
- Search accuracy (manual spot-checks of top-10 results)

### Manual Verification

After automated tests pass, perform manual validation:

1. **Installation Test**:
   ```bash
   # In a clean Python 3.10+ environment
   cd /home/deeog/Desktop/CodebaseExplainer
   pip install -e .
   codehierarchy --help  # Should display help message
   ```

2. **Ollama Integration Test**:
   ```bash
   # Verify Ollama is installed and model is available
   ollama list | grep deepseek-coder-v2:16b-q4_K_M

   # If not available, pull the model (this may take 10-15 minutes)
   ollama pull deepseek-coder-v2:16b-q4_K_M
   ```

3. **Small Repository Analysis**:
   ```bash
   # Analyze a real Python project (e.g., a small open-source repo)
   codehierarchy analyze /path/to/small/python/project \
     --output ./demo_output \
     --workers 4 \
     --batch-size 10 \
     --verbose

   # Verify output directory contains:
   # - index.md (main documentation)
   # - graph.graphml (dependency graph export)
   # - performance-metrics.json
   ```

4. **Search Functionality Test**:
   ```bash
   # Test keyword search
   codehierarchy search "main function" --mode keyword --index-dir ./demo_output

   # Test semantic search
   codehierarchy search "how to load configuration" --mode semantic --index-dir ./demo_output

   # Test hybrid search
   codehierarchy search "error handling" --mode hybrid --index-dir ./demo_output
   ```

5. **Quality Spot-Check**:
   - Open `./demo_output/index.md` and read 5-10 generated summaries
   - Verify summaries are: (a) accurate, (b) informative, (c) free of hallucinations
   - Compare summary against actual source code for a complex function
   - Verify call graph diagrams are accurate

6. **Memory Monitoring** (optional, if system monitoring available):
   ```bash
   # Run analysis while monitoring memory
   # Verify peak memory stays under 12GB for large repo
   watch -n 1 'ps aux | grep codehierarchy'
   ```

### Acceptance Criteria Summary

- [ ] All unit tests pass (`pytest tests/`)
- [ ] Test coverage ≥80% (`pytest --cov`)
- [ ] Integration test passes on 500 LOC test repository
- [ ] Performance benchmark passes on 50K LOC repository (<3 min, <6GB)
- [ ] Manual installation test succeeds
- [ ] Manual analysis produces readable documentation
- [ ] Manual search returns relevant results (spot-check top-10)
- [ ] Generated summaries are accurate and hallucination-free (spot-check 10 samples)
- [ ] Memory usage stays within budget (monitoring during test runs)

---

*End of Implementation Plan*