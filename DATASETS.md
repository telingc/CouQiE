# Dataset policy

Class IDs: 0 background, 1 penguin, 2 other bird, 3 seal, 4 whale; 255 ignored.
All local datasets are under data/. Original files are preserved.

| Source | Label policy | Limitation |
|---|---|---|
| dataset_penguins | Original COCO polygons | Some distant penguins are unlabeled |
| dataset_birds | CUB gray masks mapped to other bird | Mainly non-Antarctic species |
| dataset_seals | Recovered SealID silhouettes | Approximate masks, not manual ground truth |
| dataset_wildlife | COCO polygons; unknown animals ignored | Training exports contain same-source variants |
| dataset_nature | Explicit natural stuff is background; other/padded pixels ignored | Not a native high-resolution Antarctic collection |

The loader keeps one Wildlife record per original filename prefix before .rf.,
with deterministic filename ordering. Cross-split priority is test, valid, train.
It also removes byte-identical image overlaps. Raw variants are not deleted.
This establishes source-prefix isolation, not a guarantee of no unrelated near-duplicates.

The fresh source-count audit is in reports/datasets.csv. Historical checkpoints
were trained before source-group deduplication; their training metadata is retained.
All resize/crop operations must transform images and masks together, using nearest
interpolation for masks. Prepared inputs preserve aspect ratio with ignored padding.
