#!/usr/bin/env bash
# start.sh – Activate virtual environment and launch the Gradio application.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
  echo "❌ Virtual environment not found. Run install.sh first."
  exit 1
fi

echo "🚀 Starting Stock Analyzer..."
source venv/bin/activate
python app.py
