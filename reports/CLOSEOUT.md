# Project closeout

Status: formal experiment work closes at **1-14** on 2026-11-11. The unnumbered
experiment-10 5x5 preview was declined by the user and archived locally. It is
excluded from formal reports and the experiment index; no experiment 15 is created.

## Retained baseline

Experiment 13 is a new four-class baseline, not a whale-only ablation:
ImageNet ResNet34 encoder, random decoder/head, kernel 3 with original stem,
640 letterbox training, ten joint epochs, batch 6, AMP, normal BN, AdamW at 1e-4
and weight decay .01, CE(ignore=255), seed 42. Historical weights are not loaded.
Whale labels map to background in a separate view; original data stays intact.

Selected epoch: **10**. Foreground animal-validation mIoU: **88.37%**. Best and last are both epoch 10, so last-minus-best mIoU is zero.

| Class | Animal-validation IoU |
|---|---:|
| background | 98.03% |
| penguin | 83.11% |
| other_bird | 88.77% |
| seal | 93.22% |

[Full baseline report and training curves](exp13/README.md) |
[Epoch CSV](exp13/metrics.csv) | [Best/last metrics](exp13/best_last_metrics.csv).

## Scale diagnosis

Experiment 14 completed 15 photos x five settings = **75/75** fixed-weight cases,
all GPU float32, AMP/TF32 disabled. Letterbox canvases are 640/960/1280/1920;
Native retains each photograph's original dimensions. No fusion, thresholds or
model settings were tuned. Native is a separate setting, not a shared resolution.

| Input | Near penguin -> bird | Far penguin point hits | Six-landscape pooled FP |
|---|---:|---:|---:|
| 640 | 0.00% | 0/16 | 0.00% |
| 960 | 0.50% | 1/16 | 2.04% |
| 1280 | 8.44% | 2/16 | 2.68% |
| 1920 | 42.45% | 7/16 | 3.37% |
| Native | 69.39% | 11/16 | 7.14% |

Larger inputs recover more selected distant penguin centers while increasing
near-photo confusion and landscape false positives. Per-image trends differ;
there is no universally best tested scale. These results cannot be attributed
solely to kernel size or crop boundaries.

[Full scale report](exp14/README.md) | [Public overview](exp14/public_overview.jpg) |
[Error curves](exp14/error_curves.png) | [Per-image metrics](exp14/image_metrics.csv).

## Evidence boundaries

- Four-class animal validation, natural-background validation and the nine-photo
  development references are distinct evaluation sets.
- Historical five-class mIoU is not directly comparable to four-class mIoU.
- benchmark_v2 includes model-assisted references; it is not an independent blind test.
- The 16 exact-pixel point hits are not instance recall. Bird/seal positive recall
  comes from animal validation, not these penguin/background references.
- NJU's six photos have no manual masks here. Their predictions are qualitative;
  their photos and visualizations stay outside tracked reports.
- Historical five-class resize comparisons use their recorded AMP protocol.
  Experiment 14's float32 results use a separate, fully recorded protocol.

## Local reproduction record

The following links require ignored local artifacts; public figures and CSVs above
remain available in reports/ without those files.

Checkpoint SHA256: `ef619766453cb7e6fa9b33dfcf09d7319258796839dca104f3952abab0e43742`.

- [Experiment-13 best weights](../artifacts/experiments/exp13_results/run_20261111_150108/best.pt).
- [Experiment-13 complete local gallery](../artifacts/experiments/exp13_results/run_20261111_150108/gallery.html).
- [Experiment-14 complete 15-photo gallery, including private NJU](../artifacts/experiments/exp14_results/run_20261111_175034/gallery.html).
- [Experiment-14 protocol and source hashes](../artifacts/experiments/exp14_results/run_20261111_175034/protocol.json).
- [Read-only closeout verification](../artifacts/maintenance/closeout/latest.json).

[Code map](../CODE_MAP.md) | [Dataset policy](../DATASETS.md) |
[Environment snapshot](environment.json) | [Cleanup record](cleanup.md).
