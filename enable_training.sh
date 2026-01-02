#!/usr/bin/env bash
# Usage: ./strip_types.sh [SRC_DIR] [OUT_DIR]
# Default: SRC_DIR="."  OUT_DIR="no_types"

set -e

SRC_DIR="${1:-.}"
OUT_DIR="${2:-no_types}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
rsync -a --include='*.py' --exclude='*' "$SRC_DIR"/ "$OUT_DIR"/

find "$OUT_DIR" -name "*.py" -print0 | xargs -0 -n1 strip-hints --inplace
find "$OUT_DIR" -name "*.py" -print0 | xargs -0 autoflake --in-place --remove-all-unused-imports --remove-unused-variables
find "$OUT_DIR" -name "*.py" -print0 | xargs -0 sed -i -E '/^\s*_\w+_t = .+/d'
find "$OUT_DIR" -name "*.py" -print0 | xargs -0 black --line-length 999 

source .secrets

scp -r -P $CONT_PORT "$OUT_DIR"/* $CONT_USER@$CONT_HOST:$CONT_PATH
