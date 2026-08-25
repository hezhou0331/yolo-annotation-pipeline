#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/download_atec_data.sh"
[[ -x "$SCRIPT" ]]
urls="$($SCRIPT --print-url)"
expected=$'https://github.com/hezhou0331/yolo-annotation-pipeline/releases/download/data-20260824/atec-real-data-20260824.tar.zst.part-aa\nhttps://github.com/hezhou0331/yolo-annotation-pipeline/releases/download/data-20260824/atec-real-data-20260824.tar.zst.part-ab'
[[ "$urls" == "$expected" ]]
$SCRIPT --help | grep -q -- "--force"
$SCRIPT --help | grep -q -- "分卷"
$SCRIPT --help | grep -q -- "gh auth login"

tmp="$(mktemp -d -t atec_private_download_test.XXXXXX)"
cleanup() { find "$tmp" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/download"
cat > "$tmp/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "auth status" ]]; then
  exit 0
fi
[[ "$1 $2" == "release download" ]]
pattern=""; out_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern) pattern="$2"; shift 2 ;;
    --dir) out_dir="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case "$pattern" in
  *.part-aa) printf 'hello ' > "$out_dir/$pattern" ;;
  *.part-ab) printf 'world' > "$out_dir/$pattern" ;;
  *.sha256) printf 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9  atec-real-data-20260824.tar.zst\n' > "$out_dir/$pattern" ;;
  *) exit 3 ;;
esac
GH
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
chmod +x "$tmp/bin/gh" "$tmp/bin/tar" "$tmp/bin/zstd"
PATH="$tmp/bin:/usr/bin:/bin" ATEC_DOWNLOAD_DIR="$tmp/download" "$SCRIPT" --force | grep -q 'ATEC 数据恢复完成'

echo "DOWNLOAD_ATEC_DATA_ASSERTIONS_PASSED"
