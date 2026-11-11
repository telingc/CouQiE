# Code map

Read reports first. launch.ipynb contains only setup, links and explicitly gated
calls. The normal model path is `script.py -> animal_data.py -> experiment13.py`
for training and `experiment14.py` for scale diagnosis. Historical APIs remain
available; model classes and checkpoint keys have not been reorganized at closeout.

| Module | Responsibility |
|---|---|
| [script.py](scripts/script.py) | Shared ConvBlock, DecoderBlock, ResNetUNet, Trainer, metrics, binary dataset helpers and bird/seal preparation. Deliberately retains established imports. |
| [animal_data.py](scripts/animal_data.py) | Unified five-class datasets, joint image/mask transforms, source-group deduplication and train/valid/test assembly. |
| [project_paths.py](scripts/project_paths.py) | Canonical project, data, artifact and report locations, independent of caller cwd. |
| [gallery.py](scripts/gallery.py) | Small HTML galleries referencing external images rather than embedding image arrays. |
| [experiment13.py](scripts/experiment13.py) | Current four-class ImageNet baseline, label view, recorded sampler, training/evaluation and cached-log report generation. |
| [experiment14.py](scripts/experiment14.py) | Fixed experiment-13 weights; strict float32 scale inference, exact points, resource logs, fixed ROIs and diagnostic galleries. |
| [train_multiclass.py](scripts/train_multiclass.py) | Historical five-class training, batch throughput, BN and kernel study helpers. |
| [reference_benchmark.py](scripts/reference_benchmark.py) | Frozen historical prediction replay or explicit --rescore-only evaluation of saved masks; bare CLI may infer. |
| [report_experiments.py](scripts/report_experiments.py) | Rebuild experiments 1-14 reports from existing metadata and masks; never trains, infers or edits the notebook. |
| [check_project.py](scripts/check_project.py) | Read-only source/link/notebook/index checks; --local-results hashes completed-run inputs without loading a network. |
| [evaluate_project.py](scripts/evaluate_project.py) | Separate full-dataset/ROI audit of historical models; full outputs remain local. |
| [build_evaluation_gallery.py](scripts/build_evaluation_gallery.py) | Render a gallery from saved evaluation outputs. |
| [annotate.py](scripts/annotate.py) | Local reference-mask review editor; proposal revisions remain separate from frozen benchmarks. |
| [download_nature.py](scripts/download_nature.py) | Acquire and prepare explicit background data; network/data operation, not a closeout check. |
| [metrics.py](scripts/metrics.py) | Compatibility re-exports from script.py; preserves older notebook imports without duplicated implementations. |
| [prepare_datasets.py](scripts/prepare_datasets.py) | Compatibility CLI/re-exports from script.py; executing it prepares bird/seal data. |
| [experiment1.py](scripts/experiment1.py) | Binary overlapping sliding windows and whole-image resize inference. |
| [experiment2.py](scripts/experiment2.py) | Single weighted probability fusion of the binary inference branches. |
| [experiment3.py](scripts/experiment3.py) | Union of two differently weighted binary probability fusions. |
| [experiment4.py](scripts/experiment4.py) | Original-resolution binary inference. |
| [experiment5.py](scripts/experiment5.py) | Historical five-class baseline/tuned training launcher. |
| [experiment6.py](scripts/experiment6.py) | Historical batch-size and throughput benchmark launcher. |
| [experiment7.py](scripts/experiment7.py) | Historical normal/frozen BatchNorm comparison launcher. |
| [experiment8.py](scripts/experiment8.py) | Historical larger-kernel feasibility and training studies. |
| [experiment9.py](scripts/experiment9.py) | Historical native/resize foreground union; shared compact affine-head inference helper. |
| [experiment10.py](scripts/experiment10.py) | Historical five-group background-supervised training and comparison. |
| [experiment11.py](scripts/experiment11.py) | Historical penguin-weighted replay and shared staged-training utilities. |
| [experiment12.py](scripts/experiment12.py) | Historical requested sequential schedule using experiment-11 stage helpers. |

Experiments 1-12 preserve earlier protocols. Experiments 13 and 14 are the retained
current four-class baseline and its inference-scale diagnosis. Compatibility
modules are intentional thin wrappers, not separate metric/preparation code.
Experiment 14 reuses the existing compact-head forward from experiment 9 and
four-class scoring from experiment 13; those dependencies preserve historical
numerics. A wider package/API rewrite is deferred beyond this closed experiment set.

One-time migration/audit tools and the declined 5x5 preview source are preserved
under ignored artifacts/. They are not normal entry points. The archive and
verification locations are listed in [cleanup record](reports/cleanup.md).

After editing source or documentation, run `python scripts/check_project.py`.
To redraw cached reports, run `python scripts/report_experiments.py`.
Neither command starts training or inference. Dataset preparation, annotation,
reference revision and full evaluation remain separate explicit operations.
