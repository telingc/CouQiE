# Datasets and label policy

## Five-class model

| ID | Label |
| --- | --- |
| 0 | background |
| 1 | penguin |
| 2 | other_bird (excluding penguins) |
| 3 | seal |
| 4 | whale |
| 255 | ignore; excluded from loss and metric counts |

Images and labels share the same geometry: aspect-preserving letterbox, bilinear
RGB and nearest-neighbor labels. Training input is 640x640; RGB padding is
(124,116,104), mask padding is 255, with ImageNet mean/std normalization.
The retained inference setting is 960 letterbox: remove padding from continuous
logits, restore original coordinates bilinearly and only then take argmax.

## Current merged source view

| Prepared directory | Train | Valid | Test |
| --- | ---: | ---: | ---: |
| dataset_penguins | 423 | 124 | 64 |
| dataset_birds | 5,326 | 1,351 | 5,871 |
| dataset_seals | 871 | 428 | 838 |
| dataset_whales | 385 | 55 | 49 |
| dataset_nature | 1,759 | 129 | 112 |
| Source total | 8,764 | 2,087 | 6,934 |
| dataset_antarctic_background | 142 | 0 | 0 |

The four merged animal folders already include retained Wildlife records.
Do not load raw Wildlife again. Source variants and exact duplicates retain the
original deduplication policy, with test then validation then training priority.
This does not certify the absence of all visually similar photos or pretraining
contamination. The whale training group contains 344 HappyWhale/CFR records
plus 41 retained original records. Existing splits are preserved.

Prepared splits contain `images/`, original-size class-ID PNGs in `masks/`, and
`manifest.json`. Full multiclass masks are retained even when a photo is assigned
to one dominant-animal directory. `data/dataset_layout.json` identifies the
merged source view. All data live under ignored `data/`.

## Sources and annotation limitations

- [Penguin-finder-seg](https://universe.roboflow.com/penguins471/penguin-finder-seg): COCO polygons; some distant or occluded penguins are unlabeled.
- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/): non-Antarctic birds; uncertain grayscale mask values are ignored.
- [SealID](https://doi.org/10.3390/s22197602): recovered silhouettes from segmented JPEGs; approximate references, freshwater species.
- [WildlifeSegmentation](https://universe.roboflow.com/wildlife-o1yui/wildlifesegmentation): selected target animals, retained once after source-prefix deduplication; unknown categories are ignored.
- [HappyWhale](https://www.kaggle.com/competitions/happy-whale-and-dolphin) and [CFR wildlife masks](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks): automatically generated SAM3 whale masks; not independent manual ground truth. Photographs and masks retain their source terms.
- [COCO-Stuff](https://github.com/nightrome/cocostuff): selected natural stuff becomes background; other regions remain ignored.

## Antarctic background and target validation

The private target collection contains 167 images, frozen by scene/burst groups
as 142 train and 25 validation. Only the background-only version of the 142
training photos is used in experiment 16. Whole RGB images remain intact;
animals, unknown pixels and square safety margins become 255. The native
margin radius is `ceil(16 * max(width,height) / 960)`. Remaining background is 0.
The construction is in `scripts/prepare_antarctic_background.py`.

Private foreground masks are not training targets. The 25 validation references
are user-confirmed SAM3 annotations used for checkpoint selection, not independent
human annotations or a blind test. Whale coverage is only one validation image.
New source whale validation masks are also SAM3-derived. The six final course
test photographs are excluded from training and checkpoint selection; no test
IoU is claimed without masks. No private pixels or complete course report are
published in this repository.

Experiment 16 uses 8,906 available training images and 8,764 replacement draws per
epoch. Six group probabilities are 11.1% penguin, 39.5% bird, 16.0% seal,
10.6% whale, 5.7% Nature background and 17.1% Antarctic background. Each image
weight is its group probability divided by group size. These are sampling
probabilities, not loss weights or equal pixel frequencies.

## Historical views

Experiment 13 adapts labels in a separate four-class view: whale 4 becomes
background 0; ignore stays 255. Its original effective source view has 8,420
training images, 1,915 animal-validation images and 129 Nature validation images.
Original labels and historical checkpoints are preserved. Historical source
counts remain in [datasets.csv](reports/datasets.csv); current counts are given
above. Earlier five-class models and E13/E14 are not one matched class-count or
data-composition ablation.

[benchmark_v2](reports/references/benchmark.md) remains a frozen development set:
three penguin photos and six declared negative landscapes, partly model-assisted.
The sixteen far centers retain exact single-pixel hit rules, not instance recall.
Do not modify those masks or use validation/test images for training.
