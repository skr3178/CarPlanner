#!/bin/bash
# Fast parallel download of nuPlan Vegas shard 5 (127 GB) via aria2c
# Direct S3 public URL — no auth, supports byte-range parallelism
set -e

OUT_DIR=/media/skr/storage/autoresearch/CarPlanner_Implementation/downloads
URL="https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_vegas_5.zip"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

exec aria2c \
  --max-connection-per-server=16 \
  --split=16 \
  --min-split-size=20M \
  --file-allocation=none \
  --continue=true \
  --auto-file-renaming=false \
  --console-log-level=warn \
  --summary-interval=15 \
  --download-result=full \
  --out=nuplan-v1.1_train_vegas_5.zip \
  "$URL"
