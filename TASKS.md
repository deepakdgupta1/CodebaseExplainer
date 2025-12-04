# CodeHierarchy Explainer Implementation Tasks

## Phase 1: Project Setup & Configuration

### 1.1 Project Structure Setup
- [ ] Create root directory structure (`src/`, `tests/`, `config/`, `docs/`)
- [ ] Create `src/codehierarchy/` package with `__init__.py`
- [ ] Create subdirectories: `parser/`, `graph/`, `llm/`, `search/`, `utils/`
- [ ] Create `tests/` subdirectories mirroring `src/` structure
- [ ] Create `config/` directory for configuration files and prompts

### 1.2 Dependencies & Environment
- [ ] Create `pyproject.toml` with project metadata
- [ ] Add core dependencies: `tree-sitter`, `networkx`, `ollama`, `sentence-transformers`, `faiss-cpu`
- [ ] Add utility dependencies: `pyyaml`, `pydantic`, `rich`, `click`
- [ ] Add development dependencies: `pytest`, `pytest-cov`, `black`, `mypy`
- [ ] Create `requirements.txt` from pyproject.toml
- [ ] Create `.gitignore` for Python projects

### 1.3 Configuration System
- [ ] Create `config/schema.py` with Pydantic models for configuration validation
- [ ] Define `SystemConfig` model (memory limits, directories, checkpointing)
- [ ] Define `ParsingConfig` model (languages, workers, timeouts, file size limits)
- [ ] Define `GraphConfig` model (storage mode, cache settings, metrics)
- [ ] Define `LLMConfig` model (model name, context window, batch size, temperature)
- [ ] Define `EmbeddingsConfig` model (model name, dimension, batch size)
- [ ] Define `SearchConfig` model (modes, index type, clusters, nprobe)
- [ ] Create `config/config.yaml` with default values per design spec
- [ ] Create `config/loader.py` to load and validate YAML configuration

### 1.4 Prompt Templates
- [ ] Create `config/prompts/` directory
- [ ] Create `deepseek-optimized.txt` prompt template for batch summarization
- [ ] Create `onboarding.txt` prompt template for onboarding-focused summaries
- [ ] Add prompt template variables: `{components}`, `{context}`, `{instructions}`

---

## Phase 2: Parallel Parser Implementation

### 2.1 Tree-Sitter Setup
- [ ] Create `src/codehierarchy/parser/tree_sitter_parser.py`
- [ ] Implement `TreeSitterParser` class with language initialization
- [ ] Add method `parse_bytes(content: bytes) -> Tree` to parse source code
- [ ] Download and compile tree-sitter grammar for Python
- [ ] Download and compile tree-sitter grammar for TypeScript/JavaScript
- [ ] Create language detection utility in `utils/language_detector.py`

### 2.2 AST Node Extraction
- [ ] Create `src/codehierarchy/parser/node_extractor.py`
- [ ] Define `NodeInfo` dataclass (type, name, line, end_line, source_code, docstring, signature)
- [ ] Implement `extract_functions()` for Python using tree-sitter queries
- [ ] Implement `extract_classes()` for Python using tree-sitter queries
- [ ] Implement `extract_methods()` for Python using tree-sitter queries
- [ ] Implement `extract_functions()` for TypeScript using tree-sitter queries
- [ ] Implement `extract_classes()` for TypeScript using tree-sitter queries
- [ ] Add docstring extraction for Python nodes
- [ ] Add JSDoc extraction for TypeScript nodes

### 2.3 Call Graph Analysis
- [ ] Create `src/codehierarchy/parser/call_graph_analyzer.py`
- [ ] Define `Edge` dataclass (source, target, type, confidence)
- [ ] Implement `analyze_calls()` to extract function calls from AST
- [ ] Implement `analyze_imports()` to extract import dependencies
- [ ] Implement `analyze_inheritance()` to extract class inheritance
- [ ] Calculate confidence scores based on edge type (direct call=1.0, import=0.8)

### 2.4 Complexity Metrics
- [ ] Create `src/codehierarchy/parser/complexity.py`
- [ ] Implement cyclomatic complexity calculation for Python
- [ ] Implement cyclomatic complexity calculation for TypeScript
- [ ] Implement lines of code (LOC) counting
- [ ] Implement comment ratio calculation

### 2.5 Parallel Processing
- [ ] Create `src/codehierarchy/parser/parallel_parser.py`
- [ ] Define `ParseResult` dataclass (nodes, edges, complexity, error, skipped)
- [ ] Implement `ParallelParser.__init__(num_workers: int = 6)`
- [ ] Implement `parse_repository(files: List[Path]) -> Dict[Path, ParseResult]`
- [ ] Use `ProcessPoolExecutor` for parallel file processing
- [ ] Implement `_parse_file(file: Path) -> ParseResult` for single file parsing
- [ ] Add timeout handling (30 seconds per file)
- [ ] Add error handling and logging for failed parses

### 2.6 File Scanner
- [ ] Create `src/codehierarchy/scanner/file_scanner.py`
- [ ] Implement `scan_directory(root: Path, languages: List[str]) -> List[Path]`
- [ ] Add file type filtering by extension (`.py`, `.ts`, `.js`, `.tsx`, `.jsx`)
- [ ] Implement `.gitignore` pattern matching
- [ ] Add max file size filtering (configurable, default 10MB)
- [ ] Implement multi-threaded directory traversal

---

## Phase 3: In-Memory Graph Builder

### 3.1 Graph Construction
- [ ] Create `src/codehierarchy/graph/graph_builder.py`
- [ ] Implement `InMemoryGraphBuilder.__init__()`
- [ ] Initialize `self.graph` as `nx.DiGraph()`
- [ ] Initialize `self.node_cache: Dict[str, dict]` for source code storage
- [ ] Initialize `self.metadata: Dict[str, dict]` for metrics storage
- [ ] Implement `build_from_results(results: Dict[Path, ParseResult]) -> nx.DiGraph`

### 3.2 Node Addition
- [ ] Implement Phase 1 of graph building: add all nodes
- [ ] Create unique node IDs: `{file}:{name}:{line}`
- [ ] Add node attributes: type, name, file, line, end_line
- [ ] Cache full source code in `node_cache`
- [ ] Cache docstrings and signatures in `node_cache`
- [ ] Store complexity, LOC, and dependency count in `metadata`

### 3.3 Edge Addition
- [ ] Implement Phase 2 of graph building: add all edges
- [ ] Add edges with attributes: type, weight (confidence score)
- [ ] Handle cross-file dependencies
- [ ] Deduplicate duplicate edges

### 3.4 Graph Metrics
- [ ] Implement Phase 3 of graph building: compute metrics
- [ ] Implement `_compute_centrality()` using PageRank
- [ ] Implement `_identify_critical_paths()` using graph traversal
- [ ] Store centrality scores in node metadata
- [ ] Identify and mark critical nodes (high centrality)

### 3.5 Context Retrieval
- [ ] Implement `get_node_with_context(node_id: str, depth: int = 2) -> dict`
- [ ] Retrieve node data from graph
- [ ] Get parent nodes (predecessors)
- [ ] Get child nodes (successors)
- [ ] Retrieve source code from cache
- [ ] Create context dictionary with all related information
- [ ] Add shortest path from root to node

### 3.6 Graph Utilities
- [ ] Create `src/codehierarchy/graph/utils.py`
- [ ] Implement `get_module_subgraph(module: str) -> nx.DiGraph`
- [ ] Implement `get_dependency_chain(node_id: str) -> List[str]`
- [ ] Implement `export_graph(format: str)` for visualization (GraphML, DOT)

---

## Phase 4: DeepSeek LLM Summarizer

### 4.1 LLM Integration
- [ ] Create `src/codehierarchy/llm/deepseek_summarizer.py`
- [ ] Implement `DeepSeekSummarizer.__init__(config: LLMConfig)`
- [ ] Set model to `deepseek-coder-v2:16b-q4_K_M`
- [ ] Set context window to `128000`
- [ ] Set batch size to `20`
- [ ] Load prompt variants from config

### 4.2 Batch Processing
- [ ] Implement `summarize_batch(nodes: List[Node], graph: nx.DiGraph) -> List[str]`
- [ ] Implement `_create_smart_batches(nodes, batch_size) -> List[List[Node]]`
- [ ] Group nodes by module/file to maximize context relevance
- [ ] Ensure batches don't exceed context window token limits

### 4.3 Prompt Construction
- [ ] Implement `_build_batch_prompt(batch: List[Node], contexts: List[dict]) -> str`
- [ ] Load system prompt template from file
- [ ] Add instructions for: purpose, inputs/outputs, dependencies, usage patterns
- [ ] Add formatting instruction: `[COMPONENT_ID] <explanation>`
- [ ] For each node, include: type, location, full source code
- [ ] Add rich context: docstring, calls, called by, complexity

### 4.4 LLM API Call
- [ ] Implement Ollama API call using `ollama.chat()`
- [ ] Set system message with loaded prompt
- [ ] Set user message with batch prompt
- [ ] Configure options: `num_ctx`, `temperature`, `top_p`, `num_thread`, `num_gpu`
- [ ] Add timeout handling (5 minutes per batch)
- [ ] Add retry logic (max 2 retries)

### 4.5 Response Parsing
- [ ] Implement `_parse_batch_response(response, batch) -> List[str]`
- [ ] Extract summaries using `[COMPONENT_ID]` markers
- [ ] Validate summary length (min 100, max 600 characters)
- [ ] Handle missing or malformed summaries
- [ ] Log quality metrics (length, completeness)

### 4.6 Quality Validation
- [ ] Create `src/codehierarchy/llm/validator.py`
- [ ] Implement `validate_summary(summary: str, node: Node) -> bool`
- [ ] Check minimum length requirement
- [ ] Check for hallucination patterns (made-up function names)
- [ ] Check for key information presence (purpose, inputs/outputs)
- [ ] Flag low-quality summaries for regeneration

### 4.7 Checkpointing
- [ ] Create `src/codehierarchy/llm/checkpoint.py`
- [ ] Implement `save_checkpoint(summaries: Dict[str, str], checkpoint_file: Path)`
- [ ] Implement `load_checkpoint(checkpoint_file: Path) -> Dict[str, str]`
- [ ] Save checkpoints every 100 batches
- [ ] Allow resuming from checkpoint on failure

---

## Phase 5: Search & Embedding System

### 5.1 High-Quality Embeddings
- [ ] Create `src/codehierarchy/search/embedder.py`
- [ ] Implement `HighQualityEmbedder.__init__()`
- [ ] Load `all-mpnet-base-v2` model using SentenceTransformer
- [ ] Set dimension to 768
- [ ] Implement `encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`
- [ ] Add progress bar for encoding
- [ ] Normalize embeddings for cosine similarity

### 5.2 FAISS Index Construction
- [ ] Implement `build_index(summaries: Dict[str, str]) -> faiss.Index`
- [ ] Create IVF index with 256 clusters
- [ ] Use `IndexFlatIP` quantizer for inner product
- [ ] Train index on embeddings
- [ ] Add embeddings to index
- [ ] Store node ID to index mapping
- [ ] Save index to disk for persistence

### 5.3 Keyword Search
- [ ] Create `src/codehierarchy/search/keyword_search.py`
- [ ] Create SQLite FTS5 full-text search table
- [ ] Index node IDs, names, and summaries
- [ ] Implement `search(query: str, top_k: int) -> List[Result]`
- [ ] Use BM25 ranking algorithm
- [ ] Return results with scores

### 5.4 Semantic Search
- [ ] Implement `_semantic_search(query: str, top_k: int) -> List[Result]`
- [ ] Encode query using same model
- [ ] Set `nprobe=64` for high accuracy
- [ ] Search FAISS index
- [ ] Fetch node details for results
- [ ] Generate snippets highlighting match context
- [ ] Add match explanation

### 5.5 Hybrid Search
- [ ] Create `src/codehierarchy/search/search_engine.py`
- [ ] Implement `EnterpriseSearchEngine.__init__(index_dir: Path)`
- [ ] Load keyword index and vector index
- [ ] Implement `_hybrid_search(query: str, top_k: int) -> List[Result]`
- [ ] Get candidates from keyword search (top 2k)
- [ ] Get candidates from semantic search (top 2k)
- [ ] Apply Reciprocal Rank Fusion (RRF) with k=60
- [ ] Merge and rank results by fused score
- [ ] Return top-k results

### 5.6 Search Result Formatting
- [ ] Define `Result` dataclass (node, score, snippet, explanation)
- [ ] Implement `_generate_snippet(node, query) -> str`
- [ ] Highlight query terms in snippet
- [ ] Implement `_explain_match(node, query) -> str` using LLM
- [ ] Format results for CLI and API output

---

## Phase 6: CLI & Output Generation

### 6.1 CLI Framework
- [ ] Create `src/codehierarchy/cli/cli.py`
- [ ] Use `click` for CLI framework
- [ ] Implement `analyze` command with options: `--config`, `--output`, `--workers`, `--batch-size`
- [ ] Implement `search` command with options: `--mode`, `--top-k`
- [ ] Add `--verbose` flag for detailed logging
- [ ] Add `--checkpoint` flag to enable checkpointing

### 6.2 Pipeline Orchestration
- [ ] Create `src/codehierarchy/pipeline/orchestrator.py`
- [ ] Implement `run_pipeline(repo_path: Path, config: Config) -> dict`
- [ ] Phase 1: Scan files (call FileScanner)
- [ ] Phase 2: Parse files (call ParallelParser)
- [ ] Phase 3: Build graph (call InMemoryGraphBuilder)
- [ ] Phase 4: Generate summaries (call DeepSeekSummarizer)
- [ ] Phase 5: Build index (call HighQualityEmbedder + SearchEngine)
- [ ] Add progress bars for each phase using `rich`
- [ ] Track and report timing and memory metrics

### 6.3 Markdown Output
- [ ] Create `src/codehierarchy/output/markdown_generator.py`
- [ ] Implement `generate_documentation(graph: nx.DiGraph, summaries: Dict) -> str`
- [ ] Create hierarchy tree view (indented structure)
- [ ] For each node, output: name, type, location, summary
- [ ] Add call graph diagrams using Mermaid
- [ ] Add metrics tables (complexity, LOC, centrality)
- [ ] Generate index/table of contents
- [ ] Write to output directory

### 6.4 Metrics & Profiling
- [ ] Create `src/codehierarchy/utils/profiler.py`
- [ ] Track memory usage per phase using `psutil`
- [ ] Track execution time per phase
- [ ] Calculate throughput (files/sec, nodes/sec)
- [ ] Save metrics to JSON file (`performance-metrics.json`)
- [ ] Add warnings if memory exceeds budget

### 6.5 Logging
- [ ] Create `src/codehierarchy/utils/logger.py`
- [ ] Configure Python logging with `rich` handler
- [ ] Add log levels: DEBUG, INFO, WARNING, ERROR
- [ ] Log to console and file (`codehierarchy.log`)
- [ ] Add structured logging for metrics

---

## Phase 7: Testing

### 7.1 Unit Tests - Parser
- [ ] Create `tests/parser/test_tree_sitter_parser.py`
- [ ] Test Python parsing with sample code
- [ ] Test TypeScript parsing with sample code
- [ ] Test error handling for invalid syntax
- [ ] Create `tests/parser/test_node_extractor.py`
- [ ] Test function extraction
- [ ] Test class extraction
- [ ] Test docstring extraction
- [ ] Create `tests/parser/test_call_graph_analyzer.py`
- [ ] Test call edge detection
- [ ] Test import edge detection
- [ ] Test inheritance edge detection

### 7.2 Unit Tests - Graph
- [ ] Create `tests/graph/test_graph_builder.py`
- [ ] Test node addition with attributes
- [ ] Test edge addition with weights
- [ ] Test context retrieval
- [ ] Test graph metrics computation
- [ ] Test caching mechanisms

### 7.3 Unit Tests - LLM
- [ ] Create `tests/llm/test_deepseek_summarizer.py`
- [ ] Mock Ollama API calls
- [ ] Test batch creation logic
- [ ] Test prompt construction
- [ ] Test response parsing
- [ ] Test validation logic
- [ ] Test checkpointing

### 7.4 Unit Tests - Search
- [ ] Create `tests/search/test_embedder.py`
- [ ] Test embedding generation
- [ ] Test FAISS index building
- [ ] Create `tests/search/test_search_engine.py`
- [ ] Test keyword search
- [ ] Test semantic search
- [ ] Test hybrid search with RRF
- [ ] Test result ranking

### 7.5 Integration Tests
- [ ] Create `tests/integration/test_pipeline.py`
- [ ] Create small test repository (10 Python files)
- [ ] Test full pipeline end-to-end
- [ ] Verify output documentation is generated
- [ ] Verify search index is created
- [ ] Test search functionality on test repo
- [ ] Verify performance metrics are within expected range

### 7.6 Configuration Tests
- [ ] Create `tests/config/test_config_loader.py`
- [ ] Test YAML loading
- [ ] Test Pydantic validation
- [ ] Test default value handling
- [ ] Test invalid configuration error handling

---

## Phase 8: Documentation & Deployment

### 8.1 README Documentation
- [ ] Create `README.md` with project overview
- [ ] Add installation instructions
- [ ] Add quick start guide
- [ ] Add usage examples for analyze and search commands
- [ ] Add configuration guide
- [ ] Add architecture diagram
- [ ] Add performance benchmarks table
- [ ] Add troubleshooting section

### 8.2 API Documentation
- [ ] Create `docs/api.md`
- [ ] Document all public classes and methods
- [ ] Add code examples for programmatic usage
- [ ] Document configuration schema

### 8.3 Contributing Guide
- [ ] Create `CONTRIBUTING.md`
- [ ] Add development setup instructions
- [ ] Add testing guidelines
- [ ] Add code style guidelines (black, mypy)
- [ ] Add PR template

### 8.4 Package Setup
- [ ] Configure `pyproject.toml` for `setuptools`
- [ ] Add entry point for CLI: `codehierarchy = codehierarchy.cli.cli:main`
- [ ] Add package metadata (author, license, description)
- [ ] Test local installation with `pip install -e .`

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