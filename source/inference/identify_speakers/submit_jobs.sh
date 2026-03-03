#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") --daily|--historical"
  exit 1
}

if [[ $# -lt 1 ]]; then
  echo "Error: missing required flag."
  usage
fi

MODE=""
case "${1:-}" in
  --daily) MODE="daily" ;;
  --historical) MODE="historical" ;;
  *)
    echo "Error: unknown flag '$1'."
    usage
    ;;
esac

num_chunks=100   # number of original chunk_* files

for i in $(seq 0 $((num_chunks - 1))); do
  sbatch source/inference/identify_speakers/job_template.sh "$MODE" "$i"
done

