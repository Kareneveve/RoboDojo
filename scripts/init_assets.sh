#!/usr/bin/env bash
# Download assets for RoboDojo, MiVerse, or both from the shared Git/LFS source.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/internal/assets.py" "$@"
