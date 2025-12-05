# CodeHierarchy Explainer Implementation Tasks

## Phase 1: Project Setup & Configuration

### 1.1 Project Structure Setup
- [x] Create root directory structure (`src/`, `tests/`, `config/`, `docs/`)
- [x] Create `src/codehierarchy/` package with `__init__.py`
- [x] Create subdirectories: `parser/`, `graph/`, `llm/`, `search/`, `utils/`
- [x] Create `tests/` subdirectories mirroring `src/` structure
- [x] Create `config/` directory for configuration files and prompts

### 1.2 Dependencies & Environment
- [x] Create `pyproject.toml` with project metadata
- [x] Add core dependencies: `tree-sitter`, `networkx`, `ollama`, `sentence-transformers`, `faiss-cpu`
- [x] Add utility dependencies: `pyyaml`, `pydantic`, `rich`, `click`
- [x] Add development dependencies: `pytest`, `pytest-cov`, `black`, `mypy`
- [x] Create `requirements.txt` from pyproject.toml
- [x] Create `.gitignore` for Python projects

### 1.3 Configuration System
- [x] Create `config/schema.py` with Pydantic models for configuration validation
- [x] Define `SystemConfig` model (memory limits, directories, checkpointing)
- [x] Define `ParsingConfig` model (languages, workers, timeouts, file size limits)
- [x] Define `GraphConfig` model (storage mode, cache settings, metrics)
- [x] Define `LLMConfig` model (model name, context window, batch size, temperature)
- [x] Define `EmbeddingsConfig` model (model name, dimension, batch size)
- [x] Define `SearchConfig` model (modes, index type, clusters, nprobe)
- [x] Create `config/config.yaml` with default values per design spec
- [x] Create `config/loader.py` to load and validate YAML configuration

### 1.4 Prompt Templates
- [x] Create `config/prompts/` directory
- [x] Create `deepseek-optimized.txt` prompt template for batch summarization
- [x] Create `onboarding.txt` prompt template for onboarding-focused summaries
- [x] Add prompt template variables: `{components}`, `{context}`, `{instructions}`

---

## Phase 2: Parallel Parser Implementation

### 2.1 Tree-Sitter Setup
- [x] Create `src/codehierarchy/parser/tree_sitter_parser.py`
- [x] Implement `TreeSitterParser` class with language initialization
- [x] Add method `parse_bytes(content: bytes) -> Tree` to parse source code
- [x] Download and compile tree-sitter grammar for Python
- [x] Download and compile tree-sitter grammar for TypeScript/JavaScript
- [x] Create language detection utility in `utils/language_detector.py`

### 2.2 AST Node Extraction
- [x] Create `src/codehierarchy/parser/node_extractor.py`
- [x] Define `NodeInfo` dataclass (type, name, line, end_line, source_code, docstring, signature)
- [x] Implement `extract_functions()` for Python using tree-sitter queries
- [x] Implement `extract_classes()` for Python using tree-sitter queries
- [x] Implement `extract_methods()` for Python using tree-sitter queries
- [x] Implement `extract_functions()` for TypeScript using tree-sitter queries
- [x] Implement `extract_classes()` for TypeScript using tree-sitter queries
- [x] Add docstring extraction for Python nodes
- [x] Add JSDoc extraction for TypeScript nodes

### 2.3 Call Graph Analysis
- [x] Create `src/codehierarchy/parser/call_graph_analyzer.py`
- [x] Define `Edge` dataclass (source, target, type, confidence)
- [x] Implement `analyze_calls()` to extract function calls from AST
- [x] Implement `analyze_imports()` to extract import dependencies
- [x] Implement `analyze_inheritance()` to extract class inheritance
- [x] Calculate confidence scores based on edge type (direct call=1.0, import=0.8)

### 2.4 Complexity Metrics
- [x] Create `src/codehierarchy/parser/complexity.py`
- [x] Implement cyclomatic complexity calculation for Python
- [x] Implement cyclomatic complexity calculation for TypeScript
- [x] Implement lines of code (LOC) counting
- [x] Implement comment ratio calculation

### 2.5 Parallel Processing
- [x] Create `src/codehierarchy/parser/parallel_parser.py`
- [x] Define `ParseResult` dataclass (nodes, edges, complexity, error, skipped)
- [x] Implement `ParallelParser.__init__(num_workers: int = 6)`
- [x] Implement `parse_repository(files: List[Path]) -> Dict[Path, ParseResult]`
- [x] Use `ProcessPoolExecutor` for parallel file processing
- [x] Implement `_parse_file(file: Path) -> ParseResult` for single file parsing
- [x] Add timeout handling (30 seconds per file)
- [x] Add error handling and logging for failed parses

### 2.6 File Scanner
- [x] Create `src/codehierarchy/scanner/file_scanner.py`
- [x] Implement `scan_directory(root: Path, languages: List[str]) -> List[Path]`
- [x] Add file type filtering by extension (`.py`, `.ts`, `.js`, `.tsx`, `.jsx`)
- [x] Implement `.gitignore` pattern matching
- [x] Add max file size filtering (configurable, default 10MB)
- [x] Implement multi-threaded directory traversal

---

## Phase 3: In-Memory Graph Builder

### 3.1 Graph Construction
- [x] Create `src/codehierarchy/graph/graph_builder.py`
- [x] Implement `InMemoryGraphBuilder.__init__()`
- [x] Initialize `self.graph` as `nx.DiGraph()`
- [x] Initialize `self.node_cache: Dict[str, dict]` for source code storage
- [x] Initialize `self.metadata: Dict[str, dict]` for metrics storage
- [x] Implement `build_from_results(results: Dict[Path, ParseResult]) -> nx.DiGraph`

### 3.2 Node Addition
- [x] Implement Phase 1 of graph building: add all nodes
- [x] Create unique node IDs: `{file}:{name}:{line}`
- [x] Add node attributes: type, name, file, line, end_line
- [x] Cache full source code in `node_cache`
- [x] Cache docstrings and signatures in `node_cache`
- [x] Store complexity, LOC, and dependency count in `metadata`

### 3.3 Edge Addition
- [x] Implement Phase 2 of graph building: add all edges
- [x] Add edges with attributes: type, weight (confidence score)
- [x] Handle cross-file dependencies
- [x] Deduplicate duplicate edges

### 3.4 Graph Metrics
- [x] Implement Phase 3 of graph building: compute metrics
- [x] Implement `_compute_centrality()` using PageRank
- [x] Implement `_identify_critical_paths()` using graph traversal
- [x] Store centrality scores in node metadata
- [x] Identify and mark critical nodes (high centrality)

### 3.5 Context Retrieval
- [x] Implement `get_node_with_context(node_id: str, depth: int = 2) -> dict`
- [x] Retrieve node data from graph
- [x] Get parent nodes (predecessors)
- [x] Get child nodes (successors)
- [x] Retrieve source code from cache
- [x] Create context dictionary with all related information
- [x] Add shortest path from root to node

### 3.6 Graph Utilities
- [x] Create `src/codehierarchy/graph/utils.py`
- [x] Implement `get_module_subgraph(module: str) -> nx.DiGraph`
- [x] Implement `get_dependency_chain(node_id: str) -> List[str]`
- [x] Implement `export_graph(format: str)` for visualization (GraphML, DOT)

---

## Phase 4: DeepSeek LLM Summarizer

### 4.1 LLM Integration
- [x] Create `src/codehierarchy/llm/deepseek_summarizer.py`
- [x] Implement `DeepSeekSummarizer.__init__(config: LLMConfig)`
- [x] Set model to `deepseek-coder-v2:16b-q4_K_M`
- [x] Set context window to `128000`
- [x] Set batch size to `20`
- [x] Load prompt variants from config

### 4.2 Batch Processing
- [x] Implement `summarize_batch(nodes: List[Node], graph: nx.DiGraph) -> List[str]`
- [x] Implement `_create_smart_batches(nodes, batch_size) -> List[List[Node]]`
- [x] Group nodes by module/file to maximize context relevance
- [x] Ensure batches don't exceed context window token limits

### 4.3 Prompt Construction
- [x] Implement `_build_batch_prompt(batch: List[Node], contexts: List[dict]) -> str`
- [x] Load system prompt template from file
- [x] Add instructions for: purpose, inputs/outputs, dependencies, usage patterns
- [x] Add formatting instruction: `[COMPONENT_ID] <explanation>`
- [x] For each node, include: type, location, full source code
- [x] Add rich context: docstring, calls, called by, complexity

### 4.4 LLM API Call
- [x] Implement Ollama API call using `ollama.chat()`
- [x] Set system message with loaded prompt
- [x] Set user message with batch prompt
- [x] Configure options: `num_ctx`, `temperature`, `top_p`, `num_thread`, `num_gpu`
- [x] Add timeout handling (5 minutes per batch)
- [x] Add retry logic (max 2 retries)

### 4.5 Response Parsing
- [x] Implement `_parse_batch_response(response, batch) -> List[str]`
- [x] Extract summaries using `[COMPONENT_ID]` markers
- [x] Validate summary length (min 100, max 600 characters)
- [x] Handle missing or malformed summaries
- [x] Log quality metrics (length, completeness)

### 4.6 Quality Validation
- [x] Create `src/codehierarchy/llm/validator.py`
- [x] Implement `validate_summary(summary: str, node: Node) -> bool`
- [x] Check minimum length requirement
- [x] Check for hallucination patterns (made-up function names)
- [x] Check for key information presence (purpose, inputs/outputs)
- [x] Flag low-quality summaries for regeneration

### 4.7 Checkpointing
- [x] Create `src/codehierarchy/llm/checkpoint.py`
- [x] Implement `save_checkpoint(summaries: Dict[str, str], checkpoint_file: Path)`
- [x] Implement `load_checkpoint(checkpoint_file: Path) -> Dict[str, str]`
- [x] Save checkpoints every 100 batches
- [x] Allow resuming from checkpoint on failure

---

## Phase 5: Search & Embedding System

### 5.1 High-Quality Embeddings
- [x] Create `src/codehierarchy/search/embedder.py`
- [x] Implement `HighQualityEmbedder.__init__()`
- [x] Load `all-mpnet-base-v2` model using SentenceTransformer
- [x] Set dimension to 768
- [x] Implement `encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`
- [x] Add progress bar for encoding
- [x] Normalize embeddings for cosine similarity

### 5.2 FAISS Index Construction
- [x] Implement `build_index(summaries: Dict[str, str]) -> faiss.Index`
- [x] Create IVF index with 256 clusters
- [x] Use `IndexFlatIP` quantizer for inner product
- [x] Train index on embeddings
- [x] Add embeddings to index
- [x] Store node ID to index mapping
- [x] Save index to disk for persistence

### 5.3 Keyword Search
- [x] Create `src/codehierarchy/search/keyword_search.py`
- [x] Create SQLite FTS5 full-text search table
- [x] Index node IDs, names, and summaries
- [x] Implement `search(query: str, top_k: int) -> List[Result]`
- [x] Use BM25 ranking algorithm
- [x] Return results with scores

### 5.4 Semantic Search
- [x] Implement `_semantic_search(query: str, top_k: int) -> List[Result]`
- [x] Encode query using same model
- [x] Set `nprobe=64` for high accuracy
- [x] Search FAISS index
- [x] Fetch node details for results
- [x] Generate snippets highlighting match context
- [x] Add match explanation

### 5.5 Hybrid Search
- [x] Create `src/codehierarchy/search/search_engine.py`
- [x] Implement `EnterpriseSearchEngine.__init__(index_dir: Path)`
- [x] Load keyword index and vector index
- [x] Implement `_hybrid_search(query: str, top_k: int) -> List[Result]`
- [x] Get candidates from keyword search (top 2k)
- [x] Get candidates from semantic search (top 2k)
- [x] Apply Reciprocal Rank Fusion (RRF) with k=60
- [x] Merge and rank results by fused score
- [x] Return top-k results

### 5.6 Search Result Formatting
- [x] Define `Result` dataclass (node, score, snippet, explanation)
- [x] Implement `_generate_snippet(node, query) -> str`
- [x] Highlight query terms in snippet
- [x] Implement `_explain_match(node, query) -> str` using LLM
- [x] Format results for CLI and API output

---

## Phase 6: CLI & Output Generation

### 6.1 CLI Framework
- [x] Create `src/codehierarchy/cli/cli.py`
- [x] Use `click` for CLI framework
- [x] Implement `analyze` command with options: `--config`, `--output`, `--workers`, `--batch-size`
- [x] Implement `search` command with options: `--mode`, `--top-k`
- [x] Add `--verbose` flag for detailed logging
- [x] Add `--checkpoint` flag to enable checkpointing

### 6.2 Pipeline Orchestration
- [x] Create `src/codehierarchy/pipeline/orchestrator.py`
- [x] Implement `run_pipeline(repo_path: Path, config: Config) -> dict`
- [x] Phase 1: Scan files (call FileScanner)
- [x] Phase 2: Parse files (call ParallelParser)
- [x] Phase 3: Build graph (call InMemoryGraphBuilder)
- [x] Phase 4: Generate summaries (call DeepSeekSummarizer)
- [x] Phase 5: Build index (call HighQualityEmbedder + SearchEngine)
- [x] Add progress bars for each phase using `rich`
- [x] Track and report timing and memory metrics

### 6.3 Markdown Output
- [x] Create `src/codehierarchy/output/markdown_generator.py`
- [x] Implement `generate_documentation(graph: nx.DiGraph, summaries: Dict) -> str`
- [x] Create hierarchy tree view (indented structure)
- [x] For each node, output: name, type, location, summary
- [x] Add call graph diagrams using Mermaid
- [x] Add metrics tables (complexity, LOC, centrality)
- [x] Generate index/table of contents
- [x] Write to output directory

### 6.4 Metrics & Profiling
- [x] Create `src/codehierarchy/utils/profiler.py`
- [x] Track memory usage per phase using `psutil`
- [x] Track execution time per phase
- [x] Calculate throughput (files/sec, nodes/sec)
- [x] Save metrics to JSON file (`performance-metrics.json`)
- [x] Add warnings if memory exceeds budget

### 6.5 Logging
- [x] Create `src/codehierarchy/utils/logger.py`
- [x] Configure Python logging with `rich` handler
- [x] Add log levels: DEBUG, INFO, WARNING, ERROR
- [x] Log to console and file (`codehierarchy.log`)
- [x] Add structured logging for metrics

---

## Phase 7: Testing

### 7.1 Unit Tests - Parser
- [x] Create `tests/parser/test_tree_sitter_parser.py`
- [x] Test Python parsing with sample code
- [x] Test TypeScript parsing with sample code
- [x] Test error handling for invalid syntax
- [x] Create `tests/parser/test_node_extractor.py`
- [x] Test function extraction
- [x] Test class extraction
- [x] Test docstring extraction
- [x] Create `tests/parser/test_call_graph_analyzer.py`
- [x] Test call edge detection
- [x] Test import edge detection
- [x] Test inheritance edge detection

### 7.2 Unit Tests - Graph
- [x] Create `tests/graph/test_graph_builder.py`
- [x] Test node addition with attributes
- [x] Test edge addition with weights
- [x] Test context retrieval
- [x] Test graph metrics computation
- [x] Test caching mechanisms

### 7.3 Unit Tests - LLM
- [x] Create `tests/llm/test_deepseek_summarizer.py`
- [x] Mock Ollama API calls
- [x] Test batch creation logic
- [x] Test prompt construction
- [x] Test response parsing
- [x] Test validation logic
- [x] Test checkpointing

### 7.4 Unit Tests - Search
- [x] Create `tests/search/test_embedder.py`
- [x] Test embedding generation
- [x] Test FAISS index building
- [x] Create `tests/search/test_search_engine.py`
- [x] Test keyword search
- [x] Test semantic search
- [x] Test hybrid search with RRF
- [x] Test result ranking

### 7.5 Integration Tests
- [x] Create `tests/integration/test_pipeline.py`
- [x] Create small test repository (10 Python files)
- [x] Test full pipeline end-to-end
- [x] Verify output documentation is generated
- [x] Verify search index is created
- [x] Test search functionality on test repo
- [x] Verify performance metrics are within expected range

### 7.6 Configuration Tests
- [x] Create `tests/config/test_config_loader.py`
- [x] Test YAML loading
- [x] Test Pydantic validation
- [x] Test default value handling
- [x] Test invalid configuration error handling

---

## Phase 8: Documentation & Deployment

### 8.1 README Documentation
- [x] Create `README.md` with project overview
- [x] Add installation instructions
- [x] Add quick start guide
- [x] Add usage examples for analyze and search commands
- [x] Add configuration guide
- [x] Add architecture diagram
- [x] Add performance benchmarks table
- [x] Add troubleshooting section

### 8.2 API Documentation
- [x] Create `docs/api.md`
- [x] Document all public classes and methods
- [x] Add code examples for programmatic usage
- [x] Document configuration schema

### 8.3 Contributing Guide
- [x] Create `CONTRIBUTING.md`
- [x] Add development setup instructions
- [x] Add testing guidelines
- [x] Add code style guidelines (black, mypy)
- [x] Add PR template

### 8.4 Package Setup
- [x] Configure `pyproject.toml` for `setuptools`
- [x] Add entry point for CLI: `codehierarchy = codehierarchy.cli.cli:main`
- [x] Add package metadata (author, license, description)
- [x] Test local installation with `pip install -e .`

### 8.5 CI/CD (Optional)
- [ ] Create `.github/workflows/test.yml` for automated testing
- [ ] Run pytest with coverage on push
- [ ] Add linting checks (black, mypy)
- [ ] Add badge to README

---

## Verification Checklist

- [ ] All unit tests pass (`pytest tests/`)
- [ ] Test coverage ≥ 80% (`pytest --cov`)
- [ ] Integration test passes on sample repository
- [ ] Memory usage stays within 12GB budget
- [ ] Processing time for 1M LOC repo ≤ 30 minutes (benchmark)
- [ ] Search returns relevant results (manual testing)
- [ ] Documentation builds without errors
- [ ] Package installs successfully
- [ ] CLI commands work as expected