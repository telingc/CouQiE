# Experiment 4: Original-resolution direct inference

Check whether a fully convolutional model can segment an original-size photo without tiles.

The epoch-9 binary checkpoint is evaluated at original resolution. The final affine 1x1 head is applied before final bilinear interpolation; numerical parity is checked. CUDA AMP is used when feasible, with a recorded float32 CPU fallback after GPU OOM.

| Protocol | Mean photo IoU (3) | Pooled penguin IoU | Penguin precision | Penguin recall | Wrong animal class | Landscape FP (6) | Far penguin centers |
|---|---|---|---|---|---|---|---|
| Native | 55.25% | 44.44% | 89.95% | 46.76% | 0.00% | 0.51% | 9/16 |
| Tiles | 47.99% | 38.38% | 90.61% | 39.97% | 0.00% | 0.33% | 9/16 |
| Stretch resize | 69.23% | 77.14% | 97.30% | 78.83% | 0.00% | 5.80% | 0/16 |

| Protocol | Photo | Penguin IoU | Penguin recall | Background FP |
|---|---|---|---|---|
| Native | penguins_far | 71.92% | 79.77% | 0.93% |
| Native | penguins_near | 14.75% | 15.38% | 1.23% |
| Native | penguin_colony | 79.08% | 80.27% | 0.22% |
| Tiles | penguins_far | 60.96% | 66.56% | 0.78% |
| Tiles | penguins_near | 12.01% | 12.39% | 0.91% |
| Tiles | penguin_colony | 71.02% | 71.81% | 0.16% |
| Stretch resize | penguins_far | 71.27% | 76.88% | 0.67% |
| Stretch resize | penguins_near | 98.94% | 99.16% | 0.07% |
| Stretch resize | penguin_colony | 37.49% | 37.74% | 0.09% |

Direct inference changes object scale relative to training. It avoids tile borders but can still produce missed objects and false detections; it is not automatically equivalent to tile inference. Among the listed protocols on this set, Stretch resize has the highest mean photo IoU (69.23%); Tiles has the lowest landscape FP (0.33%).

![Photo comparison](comparison.jpg)

Fixed development references: three penguin photos and six full-background landscapes. The colony uses the user-edited mask; far uses the approved bands and near uses experiment-11 resize. These masks are diagnostic references, not an independent blind test. No positive other-bird, seal, or whale labels are present. Reported new photo scores are separate from historical 1,915-image animal validation. Standard letterbox resize uses CUDA AMP; legacy binary tile/stretch paths use float32. Native inference has a recorded float32 CPU fallback after GPU OOM. Historical models predate Wildlife source-group deduplication; no retraining was performed.

Mean photo IoU averages the three penguin photos equally. Pooled IoU/precision/recall pool pixel counts on those photos. Landscape FP is all predicted animal pixels / all pixels of the six negative landscapes. Wrong animal class is true penguin pixels assigned bird/seal/whale / all true penguin pixels. Undefined metrics are N/A.

[Reference-set protocol](../references/benchmark.md) | [All experiment reports](../README.md)
