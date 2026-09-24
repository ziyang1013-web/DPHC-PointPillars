#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/tools"
python train.py --cfg_file cfgs/kitti_models/pointpillar_dpe_trainval.yaml \
  --extra_tag dpe_trainval_seed666_80e --fix_random_seed --seed 666
DPE_CKPT="../output/kitti_models/pointpillar_dpe_trainval/dpe_trainval_seed666_80e/ckpt/checkpoint_epoch_80.pth"
[[ -f "$DPE_CKPT" ]] || { echo "Missing $DPE_CKPT" >&2; exit 1; }
python train.py --cfg_file cfgs/kitti_models/pointpillar_dphc_trainval.yaml \
  --pretrained_model "$DPE_CKPT" --extra_tag dphc_trainval_seed666_33e \
  --fix_random_seed --seed 666
