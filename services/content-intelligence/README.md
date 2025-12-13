# Content Intelligence Service

Multi-stage retrieval with hybrid search and cross-encoder reranking.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run server
uvicorn cis.api.main:app --host 0.0.0.0 --port 8081
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/context/query` | POST | Multi-stage context retrieval |
| `/v1/context/update` | POST | Incremental update |
| `/v1/health` | GET | Health and metrics |

## Architecture

```
Query → Hybrid Search (FAISS+BM25) → Cross-Encoder Rerank → Dependency Expansion
                    ↓                        ↓                      ↓
                 top-50                    top-10              Completeness Score
```

## Core Components

- **HybridIndex**: FAISS HNSW + BM25 with RRF fusion
- **CrossEncoderReranker**: ms-marco-MiniLM for accuracy
- **ASTChunker**: tree-sitter based code chunking
- **DependencyGraph**: NetworkX with red-green marking
- **CompletenessScorer**: Symbol resolution + dependency coverage
