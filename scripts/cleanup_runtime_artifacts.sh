#!/usr/bin/env bash
set -euo pipefail

apply=false

case "${1:-}" in
  --apply)
    apply=true
    shift
    ;;
  -h|--help)
    echo "Usage: $0 [--apply]"
    echo "Moves root-level runtime artifacts into ARAGORA_DATA_DIR (default: .nomic)."
    exit 0
    ;;
esac

data_dir="${ARAGORA_DATA_DIR:-.nomic}"
target_dir="${data_dir}/root-artifacts"

files=()
while IFS= read -r f; do
  files+=("$f")
done < <(find . -maxdepth 1 -type f \( \
  -name "*.db" -o -name "*.db-shm" -o -name "*.db-wal" -o -name "*.db-journal" -o \
  -name "*.sqlite" -o -name "*.sqlite3" -o -name "elo_snapshot.json" -o -name "system_health.log" \
\))

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No root-level runtime artifacts found."
  exit 0
fi

echo "ARAGORA_DATA_DIR: ${data_dir}"
echo "Target directory: ${target_dir}"

if [[ "$apply" == false ]]; then
  echo "Dry run (use --apply to move):"
  for f in "${files[@]}"; do
    echo "  would move ${f} -> ${target_dir}/$(basename "$f")"
  done
  exit 0
fi

mkdir -p "$target_dir"
for f in "${files[@]}"; do
  base="$(basename "$f")"
  dest="${target_dir}/${base}"
  if [[ -e "$dest" ]]; then
    echo "Skip (exists): ${dest}"
    continue
  fi
  mv "$f" "$dest"
  echo "Moved ${f} -> ${dest}"
done
