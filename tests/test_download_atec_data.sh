#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/download_atec_data.sh"
[[ -x "$SCRIPT" ]]
urls="$($SCRIPT --print-url)"
expected=$'https://github.com/hezhou0331/yolo-annotation-pipeline/releases/download/data-20260901/atec-real-data-20260901.tar.zst.part-aa\nhttps://github.com/hezhou0331/yolo-annotation-pipeline/releases/download/data-20260901/atec-real-data-20260901.tar.zst.part-ab\nhttps://github.com/hezhou0331/yolo-annotation-pipeline/releases/download/data-20260901/atec-real-data-20260901.tar.zst.part-ac\nhttps://github.com/hezhou0331/yolo-annotation-pipeline/releases/download/data-20260901/atec-real-data-20260901.tar.zst.part-ad'
[[ "$urls" == "$expected" ]]
$SCRIPT --help | grep -q -- "--force"
$SCRIPT --help | grep -q -- "分卷"
$SCRIPT --help | grep -q -- "公开 GitHub Release"

tmp="$(mktemp -d -t atec_public_download_test.XXXXXX)"
cleanup() { find "$tmp" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/download"
cat > "$tmp/bin/curl" <<'CURL'
#!/usr/bin/env bash
set -euo pipefail
out=""; url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) out="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
case "$url" in
  *.part-aa) printf 'hello ' > "$out" ;;
  *.part-ab) printf 'wor' > "$out" ;;
  *.part-ac) printf 'l' > "$out" ;;
  *.part-ad) printf 'd' > "$out" ;;
  *.sha256) printf 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9  atec-real-data-20260901.tar.zst\n' > "$out" ;;
  *) exit 3 ;;
esac
CURL
cat > "$tmp/bin/tar" <<'TAR'
#!/usr/bin/env bash
set -euo pipefail
archive=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -xf) archive="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ "$(cat "$archive")" == "hello world" ]]
TAR
cat > "$tmp/bin/zstd" <<'ZSTD'
#!/usr/bin/env bash
exit 0
ZSTD
chmod +x "$tmp/bin/curl" "$tmp/bin/tar" "$tmp/bin/zstd"
PATH="$tmp/bin:/usr/bin:/bin" ATEC_DOWNLOAD_DIR="$tmp/download" "$SCRIPT" --force | grep -q 'ATEC 数据恢复完成'

echo "DOWNLOAD_ATEC_DATA_ASSERTIONS_PASSED"
