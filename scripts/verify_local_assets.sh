#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -f models/checksums.sha256 ]]; then
  echo "缺少models/checksums.sha256" >&2
  exit 2
fi
sha256sum --check models/checksums.sha256
