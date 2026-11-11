# Dataset policy

Original data is preserved under ignored data/. Historical five-class readers
retain their existing labels. Experiment 13 adapts labels in a separate view.

| ID | Historical model | Experiment 13/14 |
|---|---|---|
| 0 | background | background |
| 1 | penguin | penguin |
| 2 | other_bird | other_bird |
| 3 | seal | seal |
| 4 | whale | mapped to background |
| 255 | ignored | ignored |

Whale-containing photographs can remain in the four-class dataset. Image sampling
groups are recomputed from converted labels, using the largest target foreground
area or background when no target animal remains. The historical sample_class=4
is not reused. Source masks and the shared five-class table are never overwritten.

| Source | Label policy | Limitation |
|---|---|---|
| dataset_penguins | Original COCO polygons | Some distant penguins are unlabeled |
| dataset_birds | CUB gray masks mapped to other bird | Mainly non-Antarctic species |
| dataset_seals | Recovered SealID silhouettes | Approximate masks, not manual ground truth |
| dataset_wildlife | COCO polygons; unknown animals ignored | Exports contain same-source variants |
| dataset_nature | Explicit natural stuff is background; other/padded pixels ignored | Not a native high-resolution Antarctic collection |

The loader retains one Wildlife record per original filename prefix before .rf.,
with deterministic ordering. Cross-split priority is test, valid, train; it also
removes byte-identical image overlaps. Raw variants remain intact. Prefix isolation
does not guarantee the absence of all visually similar images. Existing splits
are preserved after this documented deduplication view.

[Saved source counts](reports/datasets.csv) describe the deduplicated dataset:
8,420 training records, 1,915 animal-validation records and 129 separate
natural-background validation records. The source-prefix audit is in
[Wildlife audit](reports/wildlife.csv). The detailed four-class label/group audit
is summarized in [experiment-13 statistics](reports/exp13/data_statistics.csv)
and retained in its local run directory. Experiments 1-12 predate the final
deduplicated baseline and must retain their historical training metadata.

Resize/crop operations must transform images and masks together. Training uses
aspect-preserving 640x640 letterbox input, nearest-neighbor mask resizing and
ignored mask padding. RGB padding is (124,116,104), followed by ImageNet
normalization. Unknown/invalid/padded target pixels remain 255.

benchmark_v2 is a frozen nine-photo development reference set, not a training
split or independent blind test. It contains no positive bird/seal labels;
recall for these classes comes from animal validation. Its six landscape negatives
are distinct from the 129-image natural-background validation split. The 16 distant
points use exact single-pixel hits, not instance recall. NJU photos have no manual
masks here and provide visualizations only. See [reference definitions](reports/references/benchmark.md).
