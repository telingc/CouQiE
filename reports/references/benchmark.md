# Fixed development reference set

Historical binary/five-class replay protocol (experiments 1-12). Fixed development references: three penguin photos and six full-background landscapes. The colony uses the user-edited mask; far uses the approved bands and near uses experiment-11 resize. These masks are diagnostic references, not an independent blind test. No positive other-bird, seal, or whale labels are present. Reported new photo scores are separate from historical 1,915-image animal validation. Standard letterbox resize uses CUDA AMP; legacy binary tile/stretch paths use float32. Native inference has a recorded float32 CPU fallback after GPU OOM. Historical models predate Wildlife source-group deduplication; no retraining was performed.

Current reference version: **benchmark_v2**. The user explicitly authorized all six filenames containing landscape as full-background references. The people photo is excluded. The colony uses edit_002 from data/examples; originals and both edit revisions are preserved. Far uses resize above 50%, native from 50-62.5%, and resize below 62.5%. Near uses the experiment-11 best checkpoint's resize mask by explicit user selection. This foreground-focused reference omits back-row penguins, and that source prediction matches it exactly. Class-ID masks are stored under ignored data/reference_masks/benchmark_v2/. The prior benchmark_v1 masks and results are preserved; the other eight references are unchanged.

| Image | Reference origin | Positive penguin pixels |
|---|---|---|
| penguins_far.jpg | user_authorized_model_assisted_reference | 1,878,942 |
| penguins_near.jpg | user_selected_experiment11_resize_reference | 3,988,609 |
| penguin_colony.jpg | user_edited_reference | 1,884,963 |
| antarctic_landscape_mather_lake.jpg | user_declared_full_background | 0 |
| antarctic_landscape_untitled_1.jpeg | user_declared_full_background | 0 |
| antarctic_landscape_untitled_2.jpg | user_declared_full_background | 0 |
| antarctic_landscape_untitled_3.jpg | user_declared_full_background | 0 |
| antarctic_landscape_untitled_4.jpg | user_declared_full_background | 0 |
| antarctic_landscape_untitled_5.jpg | user_declared_full_background | 0 |

Metrics are full-resolution pixel metrics with label 255 ignored. Background FP = predicted animal pixels on true background / true background pixels. Penguin false discovery = false penguin pixels / predicted penguin pixels. Wrong animal class counts true penguin pixels labeled bird, seal, or whale. Recall for absent positive classes is undefined. Per-photo IoU is supplied so the largest foreground object does not dominate every conclusion.

An auxiliary distant-point check uses 16 visible penguin body centers selected from RGB before the earlier fresh predictions. It measures point accuracy, not instance recall. These points can reveal omissions in the model-assisted full reference itself. Coordinates and provenance are frozen in artifacts/reference_benchmark/benchmark_v2/point_check.json.

The images were used to design the inference methods and reference masks. Treat the set as development evaluation. The reference provenance and file hashes remain frozen; future edits should create a new benchmark version rather than silently changing this version.

[Annotation editor instructions](README.md) | [All reports](../README.md)
