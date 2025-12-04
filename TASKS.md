# CodeHierarchy Implementation Tasks

## Phase 1: Project Setup & Configuration

- [ ] Create root directory structure (`src/`, `tests/`, `config/`, `docs/`)
- [ ] Create `pyproject.toml` with metadata and dependencies
- [ ] Create `requirements.txt`
- [ ] Create `.gitignore`
- [ ] Create `config/config.yaml` with default settings
- [ ] Add prompt templates `deepseek-optimized.txt` and `onboarding.txt`
- [ ] Implement configuration schema (`src/codehierarchy/config/schema.py`)
- [ ] Implement config loader (`src/codehierarchy/config/loader.py`)

## Phase 2: Parallel Parser

- [ ] Implement `TreeSitterParser` (`src/codehierarchy/parser/tree_sitter_parser.py`)
- [ ] Implement `NodeInfo` and extraction logic (`src/codehierarchy/parser/node_extractor.py`)
- [ ] Implement `CallGraphAnalyzer` (`src/codehierarchy/parser/call_graph_analyzer.py`)
- [ ] Implement complexity utilities (`src/codehierarchy/parser/complexity.py`)
- [ ] Implement `ParallelParser` (`src/codehierarchy/parser/parallel_parser.py`)
- [ ] Implement `FileScanner` (`src/codehierarchy/scanner/file_scanner.py`)
- [ ] Implement language detector (`src/codehierarchy/utils/language_detector.py`)

## Phase 3: In-Memory Graph Builder

- [ ] Implement `InMemoryGraphBuilder` (`src/codehierarchy/graph/graph_builder.py`)
- [ ] Implement graph utilities (`src/codehierarchy/graph/utils.py`)

## Phase 4: DeepSeek LLM Summarizer

- [ ] Implement `DeepSeekSummarizer` (`src/codehierarchy/llm/deepseek_summarizer.py`)
- [ ] Implement summary validator (`src/codehierarchy/llm/validator.py`)
- [ ] Implement checkpointing (`src/codehierarchy/llm/checkpoint.py`)

## Phase 5: Search & Embedding System

- [ ] Implement `HighQualityEmbedder` (`src/codehierarchy/search/embedder.py`)
- [ ] Implement keyword search index (`src/codehierarchy/search/keyword_search.py`)
- [ ] Implement hybrid search engine (`src/codehierarchy/search/search_engine.py`)
- [ ] Define `Result` dataclass (`src/codehierarchy/search/result.py`)

## Phase 6: CLI & Orchestration

- [ ] Implement CLI (`src/codehierarchy/cli/cli.py`)
- [ ] Implement pipeline orchestrator (`src/codehierarchy/pipeline/orchestrator.py`)
- [ ] Implement markdown documentation generator (`src/codehierarchy/output/markdown_generator.py`)
- [ ] Implement profiler (`src/codehierarchy/utils/profiler.py`)
- [ ] Implement logger (`src/codehierarchy/utils/logger.py`)

## Phase 7: Testing

- [ ] Write unit tests for parser components
- [ ] Write unit tests for graph builder
- [ ] Write unit tests for LLM summarizer (mock Ollama)
- [ ] Write unit tests for search engine
- [ ] Write integration test for full pipeline
- [ ] Write config loader tests
- [ ] Add shared fixtures (`tests/conftest.py`)

## Phase 8: Documentation & Packaging

- [ ] Write `README.md`
- [ ] Write API docs (`docs/api.md`)
- [ ] Write CONTRIBUTING guide
- [ ] Add LICENSE file

## Verification

- [ ] All unit tests pass (`pytest tests/`)
- [ ] Coverage ≥80%
- [ ] Integration test passes on small repo
- [ ] Performance benchmarks meet targets (≤3 min, ≤6 GB for 50K LOC)
- [ ] Manual installation and analysis succeed
- [ ] Search returns relevant results
- [ ] Summaries are accurate and hallucination‑free
- [ ] Memory stays within budget
