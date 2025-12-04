#!/bin/bash
# Virtual Environment Setup Script for CodebaseExplainer

echo "🚀 Setting up CodebaseExplainer virtual environment..."

# Check if Python 3.10+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode with dependencies
echo "📥 Installing CodebaseExplainer and dependencies..."
pip install -e .

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install -e ".[dev]"

echo ""
echo "✨ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  codehierarchy --help"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
