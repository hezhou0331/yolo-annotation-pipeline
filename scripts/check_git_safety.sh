#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "尚未初始化Git仓库。" >&2
  exit 2
fi

status=0
mapfile -d '' tracked < <(git ls-files -z)
for path in "${tracked[@]}"; do
  case "$path" in
    *.pt|*.pth|*.ckpt|*.onnx|*.engine|docs/private/*.pdf|projects/*/data/*|projects/*/datasets/*|projects/*/assets/*|projects/*/reports/*|third_party/*/*)
      echo "[禁止提交] $path" >&2
      status=1
      ;;
  esac
  if [[ -f "$path" ]]; then
    size="$(stat -c %s "$path")"
    if (( size > 95000000 )); then
      echo "[超大文件] $path ($size bytes)" >&2
      status=1
    fi
  fi
done

TOKEN_PREFIX="gh""o_"
if git grep -I -n -F -e "$HOME/" -e "$TOKEN_PREFIX" -- . ':(exclude)scripts/check_git_safety.sh'; then
  echo "[敏感检查失败] 发现当前用户绝对路径或疑似GitHub token。" >&2
  status=1
fi

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
echo "Git安全检查通过：未跟踪真实数据、模型、私有PDF、第三方源码或超大文件。"
