#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if git grep -n '/home/hezhou' -- \
  README.md atec_pipeline configs docs/zh-CN scripts tools ':!third_party' ':!xcx'; then
  echo '公开文件仍包含维护者本机绝对路径。' >&2
  exit 1
fi
echo 'PUBLIC_PORTABILITY_ASSERTIONS_PASSED'
