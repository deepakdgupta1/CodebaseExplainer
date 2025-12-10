# Codebase Refactoring Documentation

**Date**: December 5, 2025  
**Version**: 0.1.0 → 0.1.1 (Structure Update)

## Overview

This document describes the major refactoring of the CodebaseExplainer project organization from a flat module structure to a functional grouping architecture, following industry best practices.

## Motivation

The previous flat structure with 12 top-level submodules was becoming unwieldy and didn't clearly communicate the architectural layers of the system. As the project grows, a well-organized structure improves:

- **Discoverability**: New contributors can quickly understand where to find code
- **Maintainability**: Related modules are grouped together
- **Scalability**: Clear boundaries for future expansion
- **Separation of Concerns**: Distinct layers for analysis, core logic, and interfaces

## Changes Summary

### Directory Structure

#### Before
```
src/codehierarchy/
├── cli/
├── config/
├── graph/
├── llm/
├── output/
├── parser/
├── pipeline/
├── scanner/
├── search/
└── utils/
```

#### After
```
src/codehierarchy/
├── analysis/         # NEW: Code analysis group
│   ├── parser/
│   ├── scanner/
│   └── graph/
├── core/             # NEW: Core functionality group  
│   ├── pipeline/
│   ├── llm/
│   └── search/
├── interface/        # NEW: User interface group
│   ├── cli/
│   └── output/
├── config/           # Kept at top level
└── utils/            # Kept at top level
```

### Functional Groups

#### 1. Analysis Package (`codehierarchy.analysis`)
- **Purpose**: Code analysis and parsing
- **Modules**:
  - `parser/`: Tree-sitter AST parsing, node extraction, call graph analysis
  - `scanner/`: File system scanning with gitignore support
  - `graph/`: Dependency graph construction and metrics

#### 2. Core Package (`codehierarchy.core`)
- **Purpose**: Core pipeline and processing
- **Modules**:
  - `pipeline/`: Main orchestration logic
  - `llm/`: DeepSeek LLM integration and summarization
  - `search/`: Semantic (vector) and keyword search engines

#### 3. Interface Package (`codehierarchy.interface`)
- **Purpose**: User-facing interfaces
- **Modules**:
  - `cli/`: Command-line interface with Click
  - `output/`: Markdown generation and report formatting

#### 4. Top-Level Packages
- `config/`: Configuration management (used by all groups)
- `utils/`: Shared utilities (logger, profiler, language detection)

## Breaking Changes

### Import Paths

All import paths have changed to reflect the new structure:

| Old Import | New Import |
|------------|------------|
| `from codehierarchy.parser import TreeSitterParser` | `from codehierarchy.analysis.parser import TreeSitterParser` |
| `from codehierarchy.llm import DeepSeekSummarizer` | `from codehierarchy.core.llm import DeepSeekSummarizer` |
| `from codehierarchy.cli import main` | `from codehierarchy.interface.cli import main` |
| `from codehierarchy.search import SearchEngine` | `from codehierarchy.core.search import SearchEngine` |

### Backward Compatibility

For convenience, the main `codehierarchy/__init__.py` re-exports commonly used classes:

```python
# These still work!
from codehierarchy import TreeSitterParser, DeepSeekSummarizer, main
```

**Note**: While backward-compatible imports work for top-level classes, it's recommended to update to the new paths for clarity.

## Files Affected

### Source Code (34 files updated)
- All Python files in moved modules had their imports updated
- Key files:
  - `core/pipeline/orchestrator.py`
  - `core/llm/lmstudio_summarizer.py`
  - `interface/cli/cli.py`
  - `analysis/parser/parallel_parser.py`
  - `analysis/graph/graph_builder.py`

### Tests (10 files updated)
- Test directory structure now mirrors source structure
- New placeholder tests added for previously untested modules:
  - `tests/analysis/scanner/test_file_scanner.py`
  - `tests/core/pipeline/test_orchestrator.py`
  - `tests/interface/cli/test_cli.py`
  - `tests/interface/output/test_markdown_generator.py`

### Configuration
- `pyproject.toml`: Updated CLI entry point
- `.gitignore`: Fixed to only exclude `/output/` at root, not module directories

## Additional Improvements

### 1. Build Artifacts Cleanup
- Removed `src/codehierarchy.egg-info/` from repository
- Added proper gitignore patterns

### 2. Documentation Organization
- Created `docs/archived/` for historical planning documents
- Created `docs/planning/` for design artifacts
- Moved 5 MD files from root to appropriate locations
- Added `docs/VENV_CLEANUP.md` with instructions

### 3. Scripts Directory
- Created `scripts/` directory
- Moved `setup_venv.sh` to `scripts/`

### 4. Test Coverage
- Added placeholder test files for 4 previously untested modules
- Test structure now matches source structure exactly

## Migration Guide

### For External Users

If you're using CodebaseExplainer as a library:

1. **Update imports** in your code:
   ```python
   # Old
   from codehierarchy.parser import TreeSitterParser
   
   # New (recommended)
   from codehierarchy.analysis.parser import TreeSitterParser
   
   # Or use backward-compatible import
   from codehierarchy import TreeSitterParser
   ```

2. **Pull latest changes**:
   ```bash
   git pull origin main
   pip install -e . --force-reinstall
   ```

3. **Verify CLI still works**:
   ```bash
   codehierarchy --help
   ```

### For Contributors

1. **Clean your environment**:
   ```bash
   # Remove old venv if present
   rm -rf venv/
   
   # Remove build artifacts
   rm -rf src/codehierarchy.egg-info/
   
   # Reinstall
   ./scripts/setup_venv.sh
   ```

2. **Update your code**:
   - Use the new import paths when creating new files
   - Follow the functional grouping when adding new modules
   - Place tests in the matching `tests/` subdirectory

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## Implementation Details

### Package Init Files

Each functional group has an `__init__.py` that re-exports key classes:

**`analysis/__init__.py`**:
```python
from codehierarchy.analysis.parser import *
from codehierarchy.analysis.scanner import *
from codehierarchy.analysis.graph import *
```

**`core/__init__.py`**:
```python
from codehierarchy.core.pipeline import *
from codehierarchy.core.llm import *
from codehierarchy.core.search import *
```

**`interface/__init__.py`**:
```python
from codehierarchy.interface.cli import *
from codehierarchy.interface.output import *
```

### Absolute vs Relative Imports

The codebase now uses **absolute imports** for better clarity:

```python
# Preferred
from codehierarchy.analysis.parser import TreeSitterParser

# Not used
from ..parser import TreeSitterParser
```

## Verification

All changes were verified through:

1. ✅ Import verification for all new module paths
2. ✅ Package installation test via `pip install -e .`
3. ✅ CLI functionality test via `codehierarchy --help`
4. ✅ Test discovery verification
5. ✅ Documentation accuracy review

## Future Considerations

### Potential Additions

As the project grows, consider:

1. **Further subgrouping** within large packages (e.g., `analysis/parser/extractors/`)
2. **API layer** in `interface/` for programmatic access
3. **Plugin system** for extensibility
4. **Performance package** for profiling and optimization utilities

### Guidelines

When adding new code:

- **Analysis code**: Put in `analysis/` if it extracts information from source code
- **Processing code**: Put in `core/` if it transforms or enriches data
- **User-facing code**: Put in `interface/` if it interacts with users
- **Configuration**: Keep in `config/` if it's about settings/schemas
- **Utilities**: Keep in `utils/` if it's a general-purpose helper

## References

- **Implementation Plan**: See artifacts directory
- **Task Tracking**: See artifacts directory

## Contact

For questions about this refactoring:
- Review the implementation plan in the artifacts directory
- Check the test files for usage examples
- Consult this document for migration guidance
