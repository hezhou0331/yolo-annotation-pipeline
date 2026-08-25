#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="hezhou0331/yolo-annotation-pipeline"
TAG="${ATEC_DATA_TAG:-data-20260825}"
ASSET="${ATEC_DATA_ASSET:-atec-real-data-20260825.tar.zst}"
read -r -a PART_SUFFIXES <<< "${ATEC_DATA_PART_SUFFIXES:-part-aa part-ab}"
DOWNLOAD_DIR="${ATEC_DOWNLOAD_DIR:-$ROOT/.downloads}"
FORCE=0
base_url="https://github.com/${REPO}/releases/download/${TAG}"

usage() {
  cat <<USAGE
用法：$(basename "$0") [--force] [--print-url]

下载并恢复 ATEC 真实 RGB-D、关键 Mask、中间结果和 YOLO11-seg 数据集。
数据包以公开 GitHub Release 的多个分卷发布；脚本会依次下载、合并、校验并解压。
运行前请安装 curl 和 zstd；下载数据不需要登录 GitHub。
默认不会覆盖已有 projects/atec_real/data 或 datasets；确需合并覆盖时使用 --force。

选项：
  --force      允许向已有数据目录中解压（不会主动删除已有文件）
  --print-url  输出当前数据快照的全部分卷下载地址
  -h, --help   显示本说明
USAGE
}

print_urls() {
  local suffix
  for suffix in "${PART_SUFFIXES[@]}"; do
    printf '%s/%s.%s\n' "$base_url" "$ASSET" "$suffix"
  done
}

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --print-url) print_urls; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${#PART_SUFFIXES[@]}" -eq 0 ]]; then
  echo "停止：未配置数据分卷。" >&2
  exit 1
fi

if [[ "$FORCE" -ne 1 ]]; then
  for path in "$ROOT/projects/atec_real/data" "$ROOT/projects/atec_real/datasets"; do
    if [[ -d "$path" && -n "$(find "$path" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "停止：目标目录已有数据：$path" >&2
      echo "如确认需要合并覆盖，请重新运行：$0 --force" >&2
      exit 1
    fi
  done
fi

if ! command -v zstd >/dev/null 2>&1; then
  echo "缺少 zstd。Ubuntu/Debian 可先运行：sudo apt install zstd" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "缺少 curl，无法下载公开 GitHub Release 数据快照。" >&2
  echo "Ubuntu/Debian 可先运行：sudo apt install curl" >&2
  exit 1
fi

download_asset() {
  local name="$1"
  local target="$2"
  local partial="${target}.partial"
  rm -f "$partial"
  curl --fail --location --retry 3 --retry-delay 2 \
    --output "$partial" "$base_url/$name"
  mv "$partial" "$target"
}

mkdir -p "$DOWNLOAD_DIR"
archive="$DOWNLOAD_DIR/$ASSET"
checksum="$archive.sha256"
part_paths=()

echo "下载 ATEC 数据快照分卷："
for suffix in "${PART_SUFFIXES[@]}"; do
  part_name="$ASSET.$suffix"
  part_path="$DOWNLOAD_DIR/$part_name"
  part_paths+=("$part_path")
  echo "  $base_url/$part_name"
  download_asset "$part_name" "$part_path"
done

download_asset "$ASSET.sha256" "$checksum"
: > "$archive"
for part_path in "${part_paths[@]}"; do
  cat "$part_path" >> "$archive"
done
(
  cd "$DOWNLOAD_DIR"
  sha256sum -c "$(basename "$checksum")"
)

echo "解压到仓库：$ROOT"
tar --zstd -xf "$archive" -C "$ROOT"
echo "ATEC 数据恢复完成。"
