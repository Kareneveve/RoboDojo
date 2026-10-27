#!/usr/bin/env bash
# Compatibility entry point; all benchmark commands live in ../evaluate.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../evaluate.sh" --bench MiVerse "$@"
