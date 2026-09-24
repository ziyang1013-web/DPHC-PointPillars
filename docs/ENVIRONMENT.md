# Environment

| Component | Verified version |
|---|---|
| OS | Ubuntu 22.04 (WSL2) |
| Python | 3.10.20 |
| PyTorch | 2.12.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| spconv | 2.3.6 |
| GPU | NVIDIA GeForce RTX 5060 Ti |

Other OpenPCDet-compatible versions may work but are not verified here.

## Reproducibility controls

- Use **--fix_random_seed --seed 666** for every published run.
- Keep split, config, batch size and checkpoint-selection rule fixed.
- Record the parent checkpoint for every stage-2 experiment.
- Keep validation AP and official-test AP separate.
