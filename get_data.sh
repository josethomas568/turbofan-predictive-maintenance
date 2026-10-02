#!/usr/bin/env bash
# Download the NASA C-MAPSS turbofan files (FD001-FD004) into data/.
# Original source: NASA Prognostics Center of Excellence data repository.
# This script uses a GitHub mirror of the unmodified files.
set -euo pipefail
mkdir -p data
BASE=https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData
for fd in FD001 FD002 FD003 FD004; do
  for kind in train test RUL; do
    curl -sSf -o "data/${kind}_${fd}.txt" "$BASE/${kind}_${fd}.txt"
  done
done
echo "Downloaded $(ls data | wc -l) files to data/"
