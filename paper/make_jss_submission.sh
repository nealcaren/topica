#!/usr/bin/env bash
# Assemble the JSS review attachments from the current committed source tree.
#
# The archive contains the complete package source, the JSS-formatted manuscript
# PDF and LaTeX source, the standalone reproduction driver, the latest generated
# report, and any captured reproduction logs. It deliberately excludes local
# environments and Rust/Python build products.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="$HERE/topica-jss-submission.tar.gz"
STAGE="$(mktemp -d /private/tmp/topica-jss.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

for file in "$HERE/topica.pdf" "$HERE/jsslogo.jpg" "$HERE/generated/replication_report.md"; do
  [ -f "$file" ] || { echo "ERROR: missing $file"; exit 1; }
done

PREFIX="topica-jss-submission"
git -C "$ROOT" archive --format=tar --prefix="$PREFIX/" HEAD | tar -xf - -C "$STAGE"
mkdir -p "$STAGE/$PREFIX/paper/generated"
cp "$HERE/topica.pdf" "$STAGE/$PREFIX/paper/"
cp "$HERE/generated/replication_report.md" "$STAGE/$PREFIX/paper/generated/"
if [ -d "$HERE/generated/logs" ]; then
  cp -R "$HERE/generated/logs" "$STAGE/$PREFIX/paper/generated/"
fi

tar -czf "$OUT" -C "$STAGE" "$PREFIX"
SIZE_BYTES="$(wc -c < "$OUT" | tr -d ' ')"
MAX_BYTES=$((50 * 1024 * 1024))
if [ "$SIZE_BYTES" -gt "$MAX_BYTES" ]; then
  echo "ERROR: $OUT exceeds JSS's 50 MB upload limit"
  exit 1
fi

echo "wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
echo "contents include package source, JSS PDF, replication material, and logs"
