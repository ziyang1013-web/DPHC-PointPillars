# Results and provenance

## KITTI validation snapshot

Fixed KITTI train/validation split, seed 666, 3D AP_R40. IoU thresholds are
0.7 for Car and 0.5 for Pedestrian/Cyclist.

| Method | Car Easy | Car Mod. | Car Hard | Ped. Easy | Ped. Mod. | Ped. Hard | Cyc. Easy | Cyc. Mod. | Cyc. Hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PointPillars | 84.07 | 74.73 | 71.59 | 50.89 | 44.26 | 39.90 | 73.83 | 57.86 | 54.04 |
| DPE-PointPillars | 84.53 | 74.95 | 71.65 | 51.65 | 45.14 | 40.48 | 83.17 | 62.43 | 58.43 |
| DPHC-PointPillars | 84.53 | 74.95 | 71.65 | 55.54 | 49.94 | 44.49 | 83.17 | 62.43 | 58.43 |

Strict CPF inherits Car and Cyclist from the frozen DPE path; only Pedestrian
predictions are replaced. Checkpoints are not bundled in the initial release,
so this is a manuscript snapshot until matching artifacts are published.

## Protocol

- Parent: DPE, seed 666, epoch 80.
- Stage 2: seed 666, batch size 2, 50-epoch validation budget.
- VFE, scatter, BEV backbone and main dense head are frozen.
- Report the selected epoch and selection metric.

## Official KITTI test diagnostic

Trainval stage 2 fixed at epoch 33:

| Inference | Ped. Easy | Ped. Mod. | Ped. Hard |
|---|---:|---:|---:|
| DPE main-only | 42.82 | 35.00 | 32.66 |
| DPHC strict CPF, auxiliary score 0.3 | 40.95 | 33.36 | 30.91 |

This does not support an official-test improvement claim for the current
trainval checkpoint. Do not mix it with validation results or tune repeated
test-server submissions from these values.
