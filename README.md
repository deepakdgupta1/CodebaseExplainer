# CodeHierarchy Explainer

A high-performance codebase documentation and search system powered by DeepSeek Coder V2.

## Overview

CodeHierarchy Explainer analyzes your codebase to generate comprehensive documentation and provide a powerful semantic search interface. It uses:
- **Tree-Sitter** for robust parsing of Python and TypeScript.
- **DeepSeek Coder V2** for generating high-quality summaries of code components.
- **NetworkX** for building a dependency graph of your code.
- **FAISS & MPNet** for state-of-the-art semantic search.
- **Rich & Click** for a beautiful CLI experience.

## Features

- 🚀 **Fast Analysis**: Parallel parsing and optimized graph building.
- 🧠 **AI Summaries**: Context-aware summaries using Large Language Models.
- 🔍 **Hybrid Search**: Combine keyword (BM25) and semantic (Vector) search for best results.
- 📊 **Metrics**: Cyclomatic complexity, LOC, and centrality metrics.
- 📄 **Markdown Output**: Generates a single, navigable markdown file of your entire codebase.

## Installation

### Quick Setup (Recommended)

Use the provided setup script to create a virtual environment and install all dependencies:

```bash
git clone https://github.com/yourusername/codehierarchy-explainer.git
cd codehierarchy-explainer
./setup_venv.sh
```

### Manual Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/codehierarchy-explainer.git
   cd codehierarchy-explainer
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

4. Install Ollama and pull the model:
   ```bash
   ollama pull deepseek-coder-v2:16b-q4_K_M
   ```

**Note**: All configuration files are now bundled within the package. No external config directory is needed.

## Quick Start

### Analyze a Repository

To generate documentation for a repository:

```bash
codehierarchy analyze /path/to/repo --output ./output
```

This will:
1. Scan and parse files.
2. Build a dependency graph.
3. Generate summaries using the LLM.
4. Create a search index.
5. Output `CODEBASE_EXPLAINER.md` in the output directory.

### Search the Codebase

Once analyzed, you can search the knowledge base:

```bash
codehierarchy search "how does the parser work?" --index-dir ./output/index
```

## Configuration

You can customize the behavior by creating a `config.yaml` file. See `config/config.yaml` for defaults.

```yaml
system:
  max_memory_gb: 16.0

llm:
  model_name: "deepseek-coder-v2:16b-q4_K_M"
  batch_size: 10
```

Pass it to the CLI:
```bash
codehierarchy analyze /path/to/repo --config my_config.yaml
```

## Architecture

The system is organized into functional groups for better maintainability:

### Package Structure

```
codehierarchy/
├── analysis/           # Code analysis and parsing
│   ├── parser/        # AST parsing with Tree-sitter
│   ├── scanner/       # File system scanning
│   └── graph/         # Dependency graph construction
├── core/              # Core pipeline functionality
│   ├── pipeline/      # Orchestration logic
│   ├── llm/           # LLM integration and summarization
│   └── search/        # Semantic and keyword search
├── interface/         # User-facing interfaces
│   ├── cli/           # Command-line interface
│   └── output/        # Markdown and report generation
├── config/            # Configuration management
└── utils/             # Shared utilities
```

### Pipeline Flow

1. **Scanner**: Discovers relevant files using `analysis.scanner`
2. **Parser**: Extracts AST nodes and call graphs using `analysis.parser`
3. **Graph Builder**: Constructs dependency graph using `analysis.graph`
4. **Summarizer**: Generates AI summaries using `core.llm`
5. **Indexer**: Builds search indices using `core.search`
6. **Generator**: Produces documentation using `interface.output`

### Import Examples

```python
# New structure imports
from codehierarchy.analysis.parser import TreeSitterParser
from codehierarchy.core.llm import DeepSeekSummarizer
from codehierarchy.interface.cli import main

# Backward-compatible imports (convenience)
from codehierarchy import TreeSitterParser, DeepSeekSummarizer
```

## Installation

### Quick Setup (Recommended)

Use the provided setup script to create a virtual environment and install all dependencies:

```bash
git clone https://github.com/yourusername/codehierarchy-explainer.git
cd codehierarchy-explainer
./scripts/setup_venv.sh
```

| Metric | Target |
|--------|--------|
| RAM Usage | < 12GB |
| Processing Time (1M LOC) | ~30 mins |
| Search Latency | < 100ms |

## Troubleshooting

- **Ollama Connection Error**: Ensure Ollama is running (`ollama serve`).
- **Memory Issues**: Reduce `llm.batch_size` or `parsing.num_workers` in config.
