#!/usr/bin/env bash
# update.sh – Pull latest changes from git and refresh dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Pulling latest changes..."
git pull

if [ ! -d "venv" ]; then
  echo "❌ Virtual environment not found. Run install.sh first."
  exit 1
fi

echo "📥 Updating dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "✅ Update complete. Run start.sh to launch."
