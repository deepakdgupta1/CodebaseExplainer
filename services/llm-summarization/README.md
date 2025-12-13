# LLM Summarization Service

LLM-based summarization with configurable providers and prompts.

## Quick Start

```bash
# Install
pip install -e .

# Run server
uvicorn lss.api.main:app --host 0.0.0.0 --port 8082
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API URL | `http://localhost:8080/v1` |
| `LLM_MODEL` | Model identifier | `local-model` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/summarize` | POST | Generate summary |
| `/v1/summarize/stream` | POST | Stream summary (SSE) |
| `/v1/summarize/batch` | POST | Batch summarization |
| `/v1/prompts` | GET/POST | Manage prompts |
| `/v1/providers` | GET | List LLM providers |
| `/v1/health` | GET | Health check |

## Summary Types

- **extractive**: Key points extraction
- **abstractive**: Natural language summary
- **hierarchical**: Multi-level summary
- **custom**: Using custom prompt template
