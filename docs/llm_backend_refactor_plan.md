# LLM Backend Abstraction Implementation Plan

## Overview

Refactor the LLM infrastructure to support configurable backends with a common interface. This enables switching between LM Studio (with xvfb automation) and llama.cpp via configuration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LMStudioSummarizer                     │
│                (unchanged public interface)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ uses
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   BaseLLMBackend (ABC)                      │
│                                                             │
│  @abstractmethod setup() -> None                            │
│  @abstractmethod load_model() -> str | None                 │
│  @abstractmethod is_healthy() -> bool                       │
│  @abstractmethod shutdown() -> None                         │
│  @property base_url: str                                    │
│  @property model_id: str                                    │
└────────────────┬─────────────────────┬──────────────────────┘
                 │                     │
        ┌────────┴────────┐   ┌────────┴────────┐
        ▼                 ▼   ▼                 ▼
┌───────────────────┐ ┌───────────────────────────┐
│  LMStudioBackend  │ │    LlamaCppBackend        │
│  (xvfb-automated) │ │  (llama-server direct)    │
└───────────────────┘ └───────────────────────────┘
```

---

## Proposed Changes

### 1. Configuration Schema Updates

#### [MODIFY] `config.yaml`
Add `backend` field to switch between implementations:

```yaml
llm:
  backend: "llamacpp"  # or "lmstudio"
  model_name: "Qwen2.5-Coder-32B-Instruct.IQ4_XS.gguf"
  
  # Common settings
  base_url: "http://localhost:8080/v1"  # Auto-set per backend
  context_window: 32768
  
  # LM Studio specific
  lmstudio:
    appimage_path: "/path/to/LMStudio.AppImage"
    use_xvfb: true
    port: 1234
    
  # llama.cpp specific  
  llamacpp:
    server_path: "/usr/local/bin/llama-server"
    model_path: "/models/Qwen2.5-Coder-32B-Instruct.IQ4_XS.gguf"
    port: 8080
    n_gpu_layers: -1  # -1 = all layers on GPU
    n_ctx: 32768
```

#### [MODIFY] `schema.py`
Add new config classes:

```python
class LMStudioConfig(BaseModel):
    appimage_path: Optional[Path] = None
    use_xvfb: bool = True
    port: int = 1234

class LlamaCppConfig(BaseModel):
    server_path: Path = Path("/usr/local/bin/llama-server")
    model_path: Path
    port: int = 8080
    n_gpu_layers: int = -1
    n_ctx: int = 32768

class LLMConfig(BaseModel):
    backend: Literal["lmstudio", "llamacpp"] = "llamacpp"
    # ... existing fields ...
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    llamacpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
```

---

### 2. Backend Abstraction Layer

#### [CREATE] `core/llm/backends/__init__.py`
Package initialization and factory function.

#### [CREATE] `core/llm/backends/base.py`
Abstract base class defining the interface:

```python
from abc import ABC, abstractmethod

class BaseLLMBackend(ABC):
    """Abstract base for LLM server backends."""
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize and start the backend server."""
        
    @abstractmethod  
    def load_model(self) -> str | None:
        """Load model and return identifier or None on failure."""
        
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if backend is responding."""
        
    @abstractmethod
    def shutdown(self) -> None:
        """Clean shutdown of the backend."""
        
    @property
    @abstractmethod
    def base_url(self) -> str:
        """OpenAI-compatible API base URL."""
        
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Currently loaded model identifier."""
```

#### [CREATE] `core/llm/backends/lmstudio.py`
LM Studio backend with xvfb automation:

- Locates LM Studio AppImage
- Starts with `xvfb-run` if `use_xvfb` is True
- Waits for server health
- Uses existing `lms` CLI for model loading

#### [CREATE] `core/llm/backends/llamacpp.py`
llama.cpp backend:

- Starts `llama-server` process directly
- Manages process lifecycle
- Health checking via API

#### [CREATE] `core/llm/backends/factory.py`
Factory function:

```python
def create_backend(config: LLMConfig) -> BaseLLMBackend:
    if config.backend == "lmstudio":
        return LMStudioBackend(config)
    elif config.backend == "llamacpp":
        return LlamaCppBackend(config)
    raise ValueError(f"Unknown backend: {config.backend}")
```

---

### 3. Refactor Existing Code

#### [MODIFY] `model_manager.py`
Rename to `lmstudio_manager.py` and move into `backends/lmstudio.py`.

#### [MODIFY] `lmstudio_summarizer.py`
Rename to `summarizer.py` and update to use backend factory:

```python
# Before
self.model_manager = ModelManager(config)

# After
from .backends.factory import create_backend
self.backend = create_backend(config)
self.backend.setup()
```

---

### 4. File Structure After Changes

```
core/llm/
├── __init__.py           # Updated exports
├── backends/
│   ├── __init__.py       # NEW: Package init + factory export
│   ├── base.py           # NEW: BaseLLMBackend ABC
│   ├── factory.py        # NEW: create_backend()
│   ├── lmstudio.py       # NEW: LMStudioBackend (from model_manager.py)
│   └── llamacpp.py       # NEW: LlamaCppBackend
├── summarizer.py         # RENAMED from lmstudio_summarizer.py
├── checkpoint.py         # Unchanged
├── progress.py           # Unchanged
└── validator.py          # Unchanged
```

---

## Implementation Steps

### Phase 1: Create Backend Abstraction (4 files)
1. [ ] Create `backends/__init__.py`
2. [ ] Create `backends/base.py` with `BaseLLMBackend` ABC
3. [ ] Create `backends/factory.py` with factory function
4. [ ] Update `core/llm/__init__.py` exports

### Phase 2: Implement llama.cpp Backend (2 files)
5. [ ] Create `backends/llamacpp.py`
6. [ ] Add `LlamaCppConfig` to `schema.py`

### Phase 3: Implement xvfb LM Studio Backend (2 files)
7. [ ] Create `backends/lmstudio.py` (refactor from `model_manager.py`)
8. [ ] Add `LMStudioConfig` to `schema.py`

### Phase 4: Update Configuration (2 files)
9. [ ] Update `config.yaml` with backend options
10. [ ] Update `LLMConfig` in `schema.py` with backend field

### Phase 5: Refactor Summarizer (3 files)
11. [ ] Rename `lmstudio_summarizer.py` → `summarizer.py`
12. [ ] Update summarizer to use backend factory
13. [ ] Update imports in `orchestrator.py`

### Phase 6: Cleanup (2 files)
14. [ ] Remove old `model_manager.py` (code moved to lmstudio.py)
15. [ ] Update tests

---

## Verification

```bash
# Test with llama.cpp backend
python -c "
from codehierarchy.core.llm.backends.factory import create_backend
from codehierarchy.config.loader import load_config
config = load_config()
config.llm.backend = 'llamacpp'
backend = create_backend(config.llm)
backend.setup()
print(f'Healthy: {backend.is_healthy()}')
"

# Test with LM Studio backend  
# (requires xvfb and LM Studio AppImage)
```
