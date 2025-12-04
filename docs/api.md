# API Documentation

## Core Components

### Orchestrator

The main entry point for running the analysis pipeline.

```python
from codehierarchy.pipeline.orchestrator import Orchestrator
from codehierarchy.config.schema import Config

config = Config()
orchestrator = Orchestrator(config)
results = orchestrator.run_pipeline(Path("/path/to/repo"))
```

### Graph Builder

Manages the dependency graph.

```python
from codehierarchy.graph.graph_builder import InMemoryGraphBuilder

builder = InMemoryGraphBuilder()
# ... build from parse results ...
context = builder.get_node_with_context("file.py:func:10")
```

### Search Engine

Performs hybrid search.

```python
from codehierarchy.search.search_engine import EnterpriseSearchEngine

engine = EnterpriseSearchEngine(Path("./index"))
results = engine.search("query", mode="hybrid")
```

## Configuration Schema

See `src/codehierarchy/config/schema.py` for the full Pydantic models defining the configuration options.
