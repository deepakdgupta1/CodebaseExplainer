# Platform Services

Two independently deployable microservices extracted from CodebaseExplainer.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **CIS** | 8081 | Content Intelligence - hybrid search, AST chunking |
| **LSS** | 8082 | LLM Summarization - multi-provider, prompt registry |

## Quick Start

```bash
# Run with Docker
cd services && docker-compose up -d

# Or run individually
cd content-intelligence && pip install -e . && uvicorn cis.api.main:app --port 8081
cd llm-summarization && pip install -e . && uvicorn lss.api.main:app --port 8082
```

## Run Tests

```bash
# CIS tests
cd content-intelligence && pip install -e ".[dev]" && pytest

# LSS tests
cd llm-summarization && pip install -e ".[dev]" && pytest
```

## Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | LSS | OpenAI-compatible API URL |
| `LLM_MODEL` | LSS | Model identifier |
| `OPENAI_API_KEY` | LSS | OpenAI API key |
| `ANTHROPIC_API_KEY` | LSS | Anthropic API key |
