#!/usr/bin/env bash
# Download full nuScenes v1.0-trainval 6-camera KEYFRAMES from the public
# CloudFront CDN (no login), 3 blobs in parallel (CloudFront throttles per
# connection). Each ~30 GB blob: resume-download, extract only samples/CAM_*
# (6-camera keyframes), delete the tarball. Resumable per blob via .done markers.
DATADIR="$(cd "$(dirname "$0")/../data/nuscenes_full" && pwd)"
export DATADIR
export BASE="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"

process_blob() {
  local i="$1"; cd "$DATADIR" || return 1
  [ -f ".blob${i}.done" ] && { echo "blob${i}: already done"; return 0; }
  echo "blob${i}: download start $(date +%H:%M:%S)"
  curl -L -C - --retry 5 --retry-delay 10 -sS -o "blob${i}.tgz" \
    "${BASE}/v1.0-trainval${i}_blobs.tgz" || { echo "blob${i}: dl FAIL"; return 1; }
  file "blob${i}.tgz" | grep -qi gzip || { echo "blob${i}: NOT gzip"; return 1; }
  echo "blob${i}: extracting $(date +%H:%M:%S)"
  tar -xzf "blob${i}.tgz" --wildcards 'samples/CAM*' || { echo "blob${i}: extract FAIL"; return 1; }
  rm -f "blob${i}.tgz"; touch ".blob${i}.done"
  echo "blob${i}: DONE ($(ls samples/CAM_FRONT 2>/dev/null | wc -l) front imgs, samples=$(du -sh samples 2>/dev/null | cut -f1))"
}
export -f process_blob

printf '%s\n' 01 02 03 04 05 06 07 08 09 10 \
  | xargs -P 3 -I {} bash -c 'process_blob "$1"' _ {}

echo "ALL_BLOBS_DONE"
cd "$DATADIR"
for c in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
  echo "  $c: $(ls samples/$c 2>/dev/null | wc -l)"
done
