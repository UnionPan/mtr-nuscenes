#!/usr/bin/env bash
# Download full nuScenes v1.0-trainval 6-camera KEYFRAMES from the public
# CloudFront CDN (no login). Each ~30 GB blob is downloaded, only samples/CAM_*
# (keyframes, all 6 cameras) are extracted, then the tarball is deleted — so the
# transient footprint stays ~40 GB and the kept data is ~30 GB. Resumable per
# blob via .blobNN.done markers. Metadata is assumed already present.
set -u
cd "$(dirname "$0")/../data/nuscenes_full"
BASE="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"

for i in 01 02 03 04 05 06 07 08 09 10; do
  if [ -f ".blob${i}.done" ]; then echo "blob${i}: already done, skip"; continue; fi
  url="${BASE}/v1.0-trainval${i}_blobs.tgz"
  echo "=== blob${i}: downloading $(date +%H:%M:%S) ==="
  curl -L -C - --retry 5 --retry-delay 10 -o "blob${i}.tgz" "$url"
  # sanity: must be a gzip tar, not an HTML error page
  if ! file "blob${i}.tgz" | grep -qi gzip; then
    echo "blob${i}: NOT gzip (download failed?), aborting"; exit 1
  fi
  echo "=== blob${i}: extracting samples/CAM_* ==="
  tar -xzf "blob${i}.tgz" --wildcards 'samples/CAM*' || { echo "blob${i}: extract failed"; exit 1; }
  rm -f "blob${i}.tgz"
  touch ".blob${i}.done"
  echo "blob${i}: done. CAM_FRONT=$(ls samples/CAM_FRONT 2>/dev/null | wc -l) imgs, disk=$(du -sh samples 2>/dev/null | cut -f1)"
done

echo "ALL_BLOBS_DONE. cameras present:"
for c in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
  echo "  $c: $(ls samples/$c 2>/dev/null | wc -l)"
done
