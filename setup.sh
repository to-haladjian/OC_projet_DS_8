#!/bin/bash
echo "Setting up development environment..."

# Configure git hooks path
git config core.hooksPath git-hooks
echo "✓ Git hooks configured"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt && pip install -r requirements-tests.txt && pip install -r requirements-monitoring.txt
echo "✓ Dependencies installed"

echo "Setup complete!"