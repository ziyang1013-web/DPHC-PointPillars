#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 checkpoint_epoch_33.pth" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/tools"
python test.py --cfg_file cfgs/kitti_models/pointpillar_dphc_test.yaml \
  --ckpt "$1" --batch_size 1 --workers 4 --save_to_file \
  --eval_tag kitti_test_submission
