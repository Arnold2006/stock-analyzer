#!/usr/bin/env bash
# install.sh – Create virtual environment and install Python dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "⬆️  Upgrading pip..."
venv/bin/pip install --upgrade pip

echo "📥 Installing dependencies..."
venv/bin/pip install -r requirements.txt

echo "✅ Installation complete."
