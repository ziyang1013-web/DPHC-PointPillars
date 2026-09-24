#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 checkpoint.pth" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/tools"
python test.py --cfg_file cfgs/kitti_models/pointpillar_dphc.yaml \
  --ckpt "$1" --batch_size 1 --workers 4 --eval_tag val_seed666
