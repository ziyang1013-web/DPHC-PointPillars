#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/tools"
python train.py --cfg_file cfgs/kitti_models/pointpillar_dpe.yaml \
  --extra_tag dpe_seed666 --fix_random_seed --seed 666
DPE_CKPT="../output/kitti_models/pointpillar_dpe/dpe_seed666/ckpt/checkpoint_epoch_80.pth"
[[ -f "$DPE_CKPT" ]] || { echo "Missing $DPE_CKPT" >&2; exit 1; }
python train.py --cfg_file cfgs/kitti_models/pointpillar_dphc.yaml \
  --pretrained_model "$DPE_CKPT" --extra_tag dphc_seed666_50e \
  --fix_random_seed --seed 666
