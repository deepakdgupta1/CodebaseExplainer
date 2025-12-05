# Virtual Environment Refactoring Tasks

## Objective
Refactor the CodebaseExplainer project to be fully self-contained within a virtual environment structure, with all configuration and data files properly organized.

## Current Issues
- Configuration files (`config/`) are outside the package structure
- Documentation files are scattered in the root
- No clear separation between development files and package files

## Refactoring Plan

### Phase 1: Restructure Configuration
- [x] Move `config/` directory into `src/codehierarchy/` as a package resource
- [x] Update `config/loader.py` to use package resources for loading config files
- [x] Update all references to config files in the codebase
- [x] Test configuration loading after changes

### Phase 2: Organize Documentation
- [x] Keep `README.md`, `CONTRIBUTING.md` in root (standard practice)
- [x] Ensure `docs/` remains in root for documentation
- [x] Verify all documentation links are correct

### Phase 3: Update Import Paths
- [x] Audit all import statements for correct package paths
- [x] Update any hardcoded file paths to use package resources
- [x] Fix cross-module dependencies

### Phase 4: Virtual Environment Setup
- [x] Create setup script for virtual environment creation
- [x] Document virtual environment activation
- [x] Test package installation in clean venv

### Phase 5: Verification
- [x] Test all imports work correctly
- [x] Verify CLI commands function properly
- [x] Run syntax checks on all Python files
- [x] Update installation instructions in README

## Progress Tracking
- **Started**: 2025-12-04 23:48
- **Completed**: 2025-12-04 23:50
- **Status**: ✅ Complete
- **Completed Tasks**: 20/20

## Summary of Changes

### Configuration Files
- ✅ Moved `config/` directory into `src/codehierarchy/config/`
- ✅ Updated `config/loader.py` to use `importlib.resources` for package-relative paths
- ✅ Added package data configuration in `pyproject.toml`
- ✅ All config files (`config.yaml`, prompt templates) now bundled with package

### Virtual Environment
- ✅ Created `setup_venv.sh` script for automated environment setup
- ✅ Updated README with comprehensive virtual environment instructions
- ✅ Documented both quick setup and manual setup procedures

### Package Structure
```
CodebaseExplainer/
├── src/codehierarchy/          # Main package (fully self-contained)
│   ├── config/                 # Configuration (now part of package)
│   │   ├── config.yaml
│   │   ├── prompts/
│   │   ├── schema.py
│   │   └── loader.py
│   ├── parser/
│   ├── graph/
│   ├── llm/
│   ├── search/
│   ├── utils/
│   ├── cli/
│   ├── pipeline/
│   └── output/
├── tests/                      # Test suite
├── docs/                       # Documentation
├── README.md                   # Updated with venv instructions
├── CONTRIBUTING.md
├── pyproject.toml              # Updated with package data
├── setup_venv.sh               # New setup script
└── requirements.txt

```

### Benefits
1. **Self-Contained**: All configuration files are now part of the installed package
2. **Portable**: Package can be installed anywhere without external dependencies on config files
3. **Virtual Environment Ready**: Easy setup with provided script
4. **Standard Compliant**: Follows Python packaging best practices
5. **Backward Compatible**: Still supports external config files via `--config` flag
