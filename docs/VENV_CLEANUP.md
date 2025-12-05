# Virtual Environment Cleanup Guide

## Why Remove venv/?

The `venv/` directory should **never** be committed to version control because:
- It contains environment-specific binaries and paths
- It massively inflates repository size (thousands of files)
- Dependencies should be managed via `pyproject.toml`
- Each developer/environment should create their own venv

## How to Remove venv/

### If venv/ exists in your clone:

```bash
# Remove the directory
rm -rf venv/

# Verify it's gone
ls -la | grep venv

# Check git status (should show venv/ as deleted if it was tracked)
git status
```

### If venv/ was previously committed:

```bash
# Remove from git history (if needed)
git rm -r --cached venv/

# Commit the removal
git commit -m "Remove venv directory from version control"
```

## Creating a Clean Virtual Environment

Use the provided setup script:

```bash
# Make it executable (if needed)
chmod +x scripts/setup_venv.sh

# Run the setup
./scripts/setup_venv.sh
```

Or manually:

```bash
# Create new venv
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Verification

After cleanup, verify `.gitignore` is working:

```bash
# Create a test venv
python3 -m venv venv

# Check git status - venv should NOT appear
git status

# If venv appears, check your .gitignore
cat .gitignore | grep venv
```

## Notes

- The `.gitignore` file has been updated to exclude `venv/`
- All future venv directories will be automatically ignored
- This is a one-time cleanup task
