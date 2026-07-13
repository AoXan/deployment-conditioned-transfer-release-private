#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: experiments/apsim_comparison/run_apsim.sh

Run every generated APSIM configuration with the pinned locally built runtime.
Use bootstrap_apsim.py and build_experiment.py before this command.
EOF
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APSIM="$ROOT/experiments/apsim_comparison/runtime/ApsimX/bin/Release/net8.0/Models.dll"
CONFIG_DIR="$ROOT/experiments/apsim_comparison/configs/generated"
LOG_DIR="$ROOT/outputs/apsim_comparison/logs"

if ! command -v dotnet >/dev/null 2>&1; then
  echo "dotnet is required; install .NET 8 and ensure dotnet is on PATH" >&2
  exit 2
fi
if [[ ! -f "$APSIM" ]]; then
  echo "APSIM runtime missing: $APSIM" >&2
  echo "Run: python3 experiments/apsim_comparison/bootstrap_apsim.py" >&2
  exit 2
fi
if ! compgen -G "$CONFIG_DIR/*.apsimx" >/dev/null; then
  echo "No generated APSIM configurations in $CONFIG_DIR" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
for config in "$CONFIG_DIR"/*.apsimx; do
  name="$(basename "$config" .apsimx)"
  rm -f "$CONFIG_DIR/$name.db"
  echo "[$(date -u +%FT%TZ)] running $name"
  dotnet "$APSIM" "$config" 2>&1 | tee "$LOG_DIR/$name.log"
done
