# CodeHierarchy Explainer

A high-performance codebase documentation and search system.

## Architecture Evolution

This project is evolving from a monolithic CLI tool into a set of loosely coupled platform services.

### Platform Services (New)

Independent microservices for scalable deployment:

- **[Content Intelligence Service (CIS)](services/content-intelligence)**:
  - Hybrid search (FAISS + BM25)
  - AST-aware chunking & dependency graph
  - Incremental updates

- **[LLM Summarization Service (LSS)](services/llm-summarization)**:
  - Multi-provider LLM support (OpenAI, Anthropic)
  - Prompt registry & management
  - Streaming summarization

See [services/README.md](services/README.md) for deployment instructions.

### Legacy CLI Tool

The original monolithic CLI is still available for local, single-machine analysis.

#### Features
- **Fast Analysis**: Parallel parsing.
- **AI Summaries**: Context-aware via LM Studio.
- **Hybrid Search**: Combine keyword and semantic search.
- **Metrics**: Complexity, LOC, centrality.

#### Quick Start

```bash
# Install
pip install -e .

# Analyze a repo
codehierarchy analyze /path/to/repo --output ./output

# Search
codehierarchy search "how does the parser work?" --index-dir ./output/index
```

## Configuration

See [src/codehierarchy/config/schema.py](src/codehierarchy/config/schema.py) for all configuration options.
