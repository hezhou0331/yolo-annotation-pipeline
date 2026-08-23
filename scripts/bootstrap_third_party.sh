#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
MODE="${1:---check}"

if [[ "$MODE" != "--check" && "$MODE" != "--install" ]]; then
  echo "用法: $0 [--check|--install]" >&2
  exit 2
fi

mkdir -p "$THIRD_PARTY"

check_or_install() {
  local name="$1" url="$2" commit="$3" dest="$THIRD_PARTY/$1"
  if [[ -d "$dest/.git" ]]; then
    local current
    current="$(git -C "$dest" rev-parse --short HEAD)"
    if [[ "$current" == "$commit" || "$(git -C "$dest" rev-parse HEAD)" == "$commit"* ]]; then
      echo "[OK] $name @ $current"
      return 0
    fi
    echo "[版本不符] $name: 当前$current，要求$commit" >&2
    if [[ "$MODE" == "--check" ]]; then return 1; fi
    if [[ -n "$(git -C "$dest" status --porcelain)" ]]; then
      echo "[停止] $name存在本地改动，不自动切换版本：$dest" >&2
      return 1
    fi
    git -C "$dest" fetch origin "$commit"
    git -C "$dest" checkout --detach "$commit"
    return 0
  fi
  if [[ -e "$dest" ]]; then
    echo "[停止] 路径已存在但不是Git仓库：$dest" >&2
    return 1
  fi
  if [[ "$MODE" == "--check" ]]; then
    echo "[缺失] $name：运行 $0 --install" >&2
    return 1
  fi
  git clone "$url" "$dest"
  git -C "$dest" checkout --detach "$commit"
  echo "[已安装] $name @ $commit"
}

status=0
check_or_install FoundationPose https://github.com/NVlabs/FoundationPose.git a1b694b || status=1
check_or_install OrbbecSDK_v2 https://github.com/orbbec/OrbbecSDK_v2.git b71adc7 || status=1
check_or_install pyorbbecsdk https://github.com/orbbec/pyorbbecsdk.git 0f089c9 || status=1
check_or_install BundleSDF https://github.com/NVlabs/BundleSDF.git ffa67d4 || status=1
check_or_install nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4f || status=1
check_or_install pytorch3d https://github.com/facebookresearch/pytorch3d.git fdaf9bd || status=1

if [[ "$status" -ne 0 ]]; then
  echo "第三方依赖检查未通过。" >&2
  exit "$status"
fi
echo "第三方依赖检查通过。模型和编译状态请另行检查。"
