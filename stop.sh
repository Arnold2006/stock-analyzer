#!/usr/bin/env bash
# stop.sh – Stop the running Stock Analyzer Python process.

set -euo pipefail

echo "⏹️  Stopping Stock Analyzer..."

PIDS=$(pgrep -f "python app.py" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
  echo "ℹ️  No running Stock Analyzer process found."
else
  echo "$PIDS" | xargs kill -TERM
  echo "✅ Stock Analyzer stopped (PIDs: $PIDS)."
fi
