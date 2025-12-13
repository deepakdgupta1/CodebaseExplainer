# Platform Services Architecture & Technical Design

## Executive Summary

This document describes the architecture for extracting two independent, reusable platform services from the existing CodebaseExplainer application:

1. **Content Intelligence Service (CIS)** - Content ingestion, parsing, embedding, and graph construction
2. **LLM Summarization Service (LSS)** - LLM-based summarization with configurable backends

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT APPLICATIONS                                │
│              (CodebaseExplainer, DocAnalyzer, ChatBot, etc.)                │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ REST/gRPC
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
┌───────────────────────────────┐   ┌─────────────────────────────────┐
│   CONTENT INTELLIGENCE SVC    │   │   LLM SUMMARIZATION SERVICE     │
│                               │   │                                 │
│  ┌─────────┐  ┌────────────┐  │   │  ┌──────────┐  ┌─────────────┐  │
│  │ Ingest  │─▶│ Parse/Chunk│  │   │  │ Prompt   │─▶│ LLM Backend │  │
│  │ Gateway │  │  Pipeline  │  │   │  │ Manager  │  │   Router    │  │
│  └─────────┘  └─────┬──────┘  │   │  └──────────┘  └─────────────┘  │
│                     │         │   │                                 │
│  ┌─────────┐  ┌────┴──────┐   │   │  ┌──────────────────────────┐   │
│  │ Graph   │◀─│ Embedding │   │   │  │ Summary Type Handlers    │   │
│  │ Builder │  │  Engine   │   │   │  │ (extractive/abstractive) │   │
│  └─────────┘  └───────────┘   │   │  └──────────────────────────┘   │
└───────────────┬───────────────┘   └─────────────────┬───────────────┘
                │                                     │
        ┌───────┴───────┐                     ┌───────┴───────┐
        ▼               ▼                     ▼               ▼
┌─────────────┐  ┌────────────┐        ┌───────────┐   ┌───────────┐
│  Vector DB  │  │  Graph DB  │        │ LLM Infra │   │  Cache    │
│   (FAISS/   │  │  (Neo4j/   │        │ (llama.cpp│   │  (Redis)  │
│   Qdrant)   │  │  NetworkX) │        │  /OpenAI) │   │           │
└─────────────┘  └────────────┘        └───────────┘   └───────────┘
```

---

## 2. Service Responsibilities

### 2.1 Content Intelligence Service (CIS)

| Responsibility | Reused Component | New Development |
|----------------|------------------|-----------------|
| Content Ingestion | `FileScanner` | Multi-source adapters |
| Parsing & Chunking | `ParallelParser`, `TreeSitterParser` | Document parsers |
| Embedding Generation | `HighQualityEmbedder` | Batched async API |
| Graph Construction | `InMemoryGraphBuilder` | Persistence layer |
| Storage & Indexing | - | Vector/Graph DB integration |

### 2.2 LLM Summarization Service (LSS)

| Responsibility | Reused Component | New Development |
|----------------|------------------|-----------------|
| LLM Backend Management | `backends/` (llamacpp, lmstudio) | OpenAI, Anthropic adapters |
| Prompt Management | Prompt templates | Dynamic prompt registry |
| Summary Generation | `LMStudioSummarizer` | Summary type handlers |
| Validation | `validator.py` | Quality scoring API |

---

## 3. Content Intelligence Service - Detailed Design

### 3.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT INTELLIGENCE SERVICE                  │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Stage Retrieval Pipeline                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │Query Analysis│─▶│Hybrid Search │─▶│Cross-Encoder │           │
│  │& Expansion   │  │(FAISS+BM25)  │  │  Reranking   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                │ RRF              │ top-10            │
│         ▼                ▼ fusion           ▼ scores            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │Symbol Extract│  │  top-50      │  │ Dependency   │           │
│  │Intent Detect │  │  candidates  │  │  Expansion   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                            │ depth=2            │
│                                            ▼                    │
│                                      ┌──────────────┐           │
│                                      │Completeness  │           │
│                                      │  Scoring     │           │
│                                      └──────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│  Processing Pipeline                                             │
│  ┌────────────┐  ┌─────────────┐  ┌────────────┐                │
│  │ AST Parser │─▶│ AST-Aware   │─▶│ Dual       │                │
│  │(tree-sitter)│  │  Chunker    │  │ Embedder   │                │
│  └────────────┘  └─────────────┘  └────────────┘                │
│        │                               │                        │
│        ▼                               ▼                        │
│  ┌─────────────┐              ┌────────────────┐                │
│  │ Dependency  │              │  Hybrid Index  │                │
│  │   Graph     │              │ (FAISS + BM25) │                │
│  │(red-green)  │              └────────────────┘                │
│  └─────────────┘                                                │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                   │
│  ├── FAISS HNSW     (dense vectors, CodeBERT 768d)              │
│  ├── BM25 Index     (sparse, inverted index for keywords)       │
│  ├── SQLite         (AST cache, metadata, fingerprints)         │
│  ├── NetworkX       (dependency graph with gpickle)             │
│  └── Object Storage (raw content, large files)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Algorithms

#### Hybrid Search with RRF Fusion
```python
def hybrid_search(query: str, k: int = 20, alpha: float = 0.5):
    # Dense vector search (CodeBERT)
    vector_results = faiss_index.search(embed(query), k=50)
    # Sparse keyword search (BM25)
    bm25_results = bm25_index.search(tokenize(query), k=50)
    # Reciprocal Rank Fusion
    fused = {}
    for rank, (id, _) in enumerate(vector_results):
        fused[id] = alpha * (1 / (60 + rank + 1))
    for rank, (id, _) in enumerate(bm25_results):
        fused[id] = fused.get(id, 0) + (1-alpha) * (1 / (60 + rank + 1))
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
```

#### Cross-Encoder Reranking
- Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Process query + document jointly (20-40% accuracy gain)
- Latency: 200-400ms for 50 chunks on CPU

#### Completeness Scoring
```
Completeness = 0.7 * symbol_resolution + 0.3 * dependency_coverage
Target: >0.90
```

### 3.3 Enhanced API Contracts

#### Context Query (Primary API)
```yaml
POST /v1/context/query
Request:
  query: string
  context:
    current_file: string (optional)
    cursor_line: int (optional)
  options:
    max_tokens: int (default: 4096)
    min_completeness: float (default: 0.9)
    dependency_depth: int (default: 2)

Response:
  context_id: string
  token_count: int
  completeness_score: float
  retrieval_time_ms: int
  chunks:
    - file: string
      lines: string
      relevance_score: float
      symbol: string
      content: string
      dependencies: string[]
  dependency_tree: object
  metadata:
    retrieval_stages:
      hybrid_search_ms: int
      reranking_ms: int
      dependency_expansion_ms: int
```

#### Incremental Update (Red-Green Marking)
```yaml
POST /v1/context/update
Request:
  action: "modify" | "create" | "delete"
  file_path: string
  content: string

Response:
  status: "updated"
  reindexed_chunks: int
  dirty_nodes: int
  processing_time_ms: int
```

### 3.3 Data Models

```python
@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function|class|module
    symbol_name: str
    signature: str
    content: str
    
    # Dual embeddings
    dense_embedding: np.ndarray  # CodeBERT 768d
    tokens: List[str]            # For BM25
    
    # Dependencies
    imports: List[str]
    calls: List[str]
    inherits: List[str]
    
    # Metadata
    complexity: int
    fingerprint: str  # SHA-256 for incremental updates
    last_modified: datetime
```

### 3.5 Performance Targets

| Metric | Target |
|--------|--------|
| Hybrid search | 80-150ms |
| Cross-encoder rerank | 200-400ms |
| **Total query latency** | **350-700ms** |
| Incremental update | <1s/file |
| Token savings | 90-95% |
| Completeness | >0.90 |
| Precision@10 | >0.85 |

### 3.6 Technology Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `microsoft/codebert-base` (768d) |
| Reranker | `ms-marco-MiniLM-L-12-v2` |
| Vector DB | FAISS HNSW (M=16, ef=200) |
| Sparse Index | BM25 (rank-bm25) |
| AST Parser | tree-sitter |
| Graph | NetworkX + gpickle |
| Cache | SQLite |

---

## 4. LLM Summarization Service - Detailed Design

### 4.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM SUMMARIZATION SERVICE                      │
├─────────────────────────────────────────────────────────────────┤
│  API Layer                                                       │
│  ├── POST /v1/summarize           (single/batch summarization)  │
│  ├── POST /v1/summarize/stream    (streaming response)          │
│  ├── GET  /v1/prompts             (list prompt templates)       │
│  ├── POST /v1/prompts             (register custom prompt)      │
│  └── GET  /v1/providers           (available LLM backends)      │
├─────────────────────────────────────────────────────────────────┤
│  Core Components                                                 │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Prompt Manager │─▶│ Summary Engine  │─▶│ Backend Router  │   │
│  └────────────────┘  └─────────────────┘  └────────┬────────┘   │
│                             │                      │             │
│                      ┌──────┴──────┐       ┌───────┴───────┐     │
│                      │  Validator  │       │   Backends    │     │
│                      │   & Score   │       │  ┌─────────┐  │     │
│                      └─────────────┘       │  │llamacpp │  │     │
│                                            │  │lmstudio │  │     │
│                                            │  │openai   │  │     │
│                                            │  │anthropic│  │     │
│                                            │  └─────────┘  │     │
│                                            └───────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  Storage & Cache                                                 │
│  ├── Redis          (response cache, rate limiting)             │
│  └── PostgreSQL     (prompt registry, audit logs)               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 API Contracts

#### Summarize Content
```yaml
POST /v1/summarize
Request:
  content: string | ContentReference
  summary_type: "extractive" | "abstractive" | "hierarchical" | "custom"
  options:
    provider: string (default: from config)
    model: string (optional)
    prompt_template: string (template ID or inline)
    temperature: float (default: 0.2)
    max_tokens: int
    context:
      graph_data: object (optional, from CIS)
      embeddings: float[] (optional)
      metadata: object

Response:
  summary: string
  quality_score: float
  tokens_used: int
  provider: string
  model: string
  cached: bool
```

#### Streaming Summarization
```yaml
POST /v1/summarize/stream
Request: (same as above)
Response: Server-Sent Events
  data: {"chunk": "partial text...", "done": false}
  data: {"chunk": "", "done": true, "summary": "full text", "score": 0.85}
```

### 4.3 Summary Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Extractive** | Selects key sentences verbatim | Quick overviews |
| **Abstractive** | Generates new text | Documentation |
| **Hierarchical** | Multi-level summaries | Large codebases |
| **Task-Specific** | Custom prompts | Domain-specific |

### 4.4 Backend Abstraction (Reused)

```python
# Existing backends/ package fully reusable
class BaseLLMBackend(ABC):
    @abstractmethod
    def setup(self) -> None: ...
    @abstractmethod
    def load_model(self) -> str | None: ...
    @abstractmethod
    def is_healthy(self) -> bool: ...
    @abstractmethod
    def shutdown(self) -> None: ...
    @property
    def base_url(self) -> str: ...
    def get_extra_body(self) -> dict: ...

# Add new backends:
class OpenAIBackend(BaseLLMBackend): ...
class AnthropicBackend(BaseLLMBackend): ...
```

---

## 5. Inter-Service Communication

```
┌────────────────┐         ┌───────────────────┐
│ Client App     │         │                   │
│                │         │  LSS              │
│ 1. Submit      │    3. Request context       │
│    content ────┼────┐    │      ◄────────────┤
│                │    │    │                   │
│ 2. Get         │    ▼    │  4. Summarize     │
│    summaries ──┼───CIS───┼──────────────────▶│
│          ▲     │         │                   │
│          │     │         │  5. Return        │
│          └─────┼─────────┼───────────────────┤
└────────────────┘         └───────────────────┘

Flow:
1. Client ingests content → CIS
2. CIS returns content_id, embeddings, graph
3. Client requests summary from LSS, passing content_id
4. LSS optionally fetches context from CIS
5. LSS returns summary to client
```

**Communication Pattern:** Services are loosely coupled. LSS can operate independently with inline content, or fetch rich context from CIS when `content_id` is provided.

---

## 6. Scalability & Performance

### 6.1 Horizontal Scaling

| Component | Scaling Strategy |
|-----------|------------------|
| CIS API | Stateless replicas behind load balancer |
| Embedding Workers | Queue-based (Celery/RQ) autoscaling |
| Graph Builder | Partitioned by content namespace |
| LSS API | Stateless replicas |
| LLM Backends | GPU node pools with affinity |

### 6.2 Performance Optimizations

- **Batching:** Embed chunks in batches of 32-128
- **Caching:** Redis cache for repeated queries (TTL: 1h)
- **Async Processing:** Job queue for large content
- **Connection Pooling:** Persistent LLM backend connections
- **Incremental Updates:** Content hash-based dedup

### 6.3 Resource Estimates

| Workload | CIS | LSS |
|----------|-----|-----|
| 1K docs/day | 2 CPU, 4GB | 1 GPU, 8GB |
| 10K docs/day | 4 CPU, 16GB | 2 GPU, 16GB |
| 100K docs/day | 8 CPU, 32GB, distributed | 4 GPU, autoscale |

---

## 7. Fault Tolerance

| Failure Mode | Mitigation |
|--------------|------------|
| LLM Backend Down | Circuit breaker, fallback provider |
| Vector DB Unavailable | Read replicas, graceful degradation |
| Parsing Timeout | Dead letter queue, max retries (3) |
| OOM on Large Content | Streaming ingestion, chunk limits |

**Retry Strategy:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=60),
    retry=retry_if_exception_type(TransientError)
)
```

---

## 8. Deployment Model

### 8.1 Container Architecture

```yaml
# docker-compose.yml (dev/staging)
services:
  cis-api:
    image: platform/content-intelligence:latest
    replicas: 2
    ports: ["8081:8080"]
    
  cis-worker:
    image: platform/content-intelligence:latest
    command: celery worker
    replicas: 4
    
  lss-api:
    image: platform/llm-summarization:latest
    replicas: 2
    ports: ["8082:8080"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
            
  postgres:
    image: postgres:15
    
  redis:
    image: redis:7
    
  qdrant:
    image: qdrant/qdrant:latest
```

### 8.2 Kubernetes Production

```yaml
# CIS Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: content-intelligence-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cis
  template:
    spec:
      containers:
      - name: api
        image: platform/content-intelligence:v1
        resources:
          requests: {cpu: "500m", memory: "1Gi"}
          limits: {cpu: "2", memory: "4Gi"}
        livenessProbe:
          httpGet: {path: /health, port: 8080}
        readinessProbe:
          httpGet: {path: /ready, port: 8080}
```

---

## 9. Observability

### 9.1 Logging

```python
# Structured logging (JSON)
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "service": "content-intelligence",
    "trace_id": "abc123",
    "span_id": "def456",
    "message": "Content ingested",
    "content_id": "uuid",
    "chunks": 42,
    "duration_ms": 1234
}
```

### 9.2 Metrics (Prometheus)

| Metric | Type | Labels |
|--------|------|--------|
| `cis_ingestion_total` | Counter | source_type, status |
| `cis_embedding_duration_seconds` | Histogram | model |
| `lss_summarization_total` | Counter | provider, summary_type |
| `lss_tokens_used_total` | Counter | provider, model |
| `backend_health` | Gauge | backend, status |

### 9.3 Tracing

- OpenTelemetry for distributed tracing
- Trace context propagation between services
- Span for each processing stage

---

## 10. Extension Points

| Extension | Mechanism |
|-----------|-----------|
| New Content Sources | Implement `ContentAdapter` interface |
| New Parsers | Register in `ParserRegistry` |
| New LLM Providers | Extend `BaseLLMBackend` |
| Custom Summary Types | Add to `SummaryTypeRegistry` |
| Graph Algorithms | Plugin to `GraphBuilder` |
| Vector Models | Configure `EmbedderConfig` |

---

## 11. Migration Path from CodebaseExplainer

### Phase 1: Extract CIS (Week 1-2)
1. Create new repo: `platform-content-intelligence`
2. Copy: `analysis/`, `core/search/`, `utils/`
3. Add FastAPI wrapper with REST endpoints
4. Add PostgreSQL + Qdrant integration

### Phase 2: Extract LSS (Week 2-3)
1. Create new repo: `platform-llm-summarization`
2. Copy: `core/llm/backends/`, `validator.py`
3. Add prompt registry and summary type handlers
4. Add streaming support

### Phase 3: Refactor CodebaseExplainer (Week 3-4)
1. Replace internal modules with service clients
2. Add retry/circuit-breaker patterns
3. Update tests to use service mocks

---

## 12. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Content Injection | Input validation, sanitization |
| LLM Prompt Injection | Template parameterization, blocklists |
| API Authentication | JWT/OAuth2, API keys |
| Data Isolation | Tenant-scoped storage |
| Secrets Management | Vault/K8s secrets |
