"""Experiment 3: combine two differently weighted fusion predictions.

Both branches use the aligned foreground probabilities from experiment 1.
Two convex combinations favor the tiled and whole-image paths, respectively.
Each is thresholded independently and their binary masks are unioned.
The weighted scores need not be calibrated probabilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from experiment1 import predict_with_resize, predict_with_tiles
from experiment2 import fuse_probabilities


def fuse_biased_predictions(
    global_probability,
    tile_probability,
    tile_bias_alpha=0.25,
    global_bias_alpha=0.75,
    tile_threshold=0.5,
    global_threshold=0.5,
):
    """Return both weighted score maps, their masks, and the mask union.

    Both alpha parameters are weights for the whole-image prediction, as in
    experiment 2: alpha * global + (1 - alpha) * tile. Lower values favor the
    tiled branch; higher values favor the resized whole-image branch.
    Each score is classified using a strict ``>`` with its own threshold.

    Returns ``(tile_score, global_score, tile_biased_mask,
    global_biased_mask, union_mask)``. The score maps have original image
    resolution; masks are NumPy uint8 arrays containing zero or one.
    """
    tile_score, tile_biased_mask = fuse_probabilities(
        global_probability, tile_probability,
        alpha=tile_bias_alpha, threshold=tile_threshold,
    )
    global_score, global_biased_mask = fuse_probabilities(
        global_probability, tile_probability,
        alpha=global_bias_alpha, threshold=global_threshold,
    )
    union_mask = np.logical_or(tile_biased_mask, global_biased_mask).astype(np.uint8)
    return tile_score, global_score, tile_biased_mask, global_biased_mask, union_mask


def show_dual_fusion_comparison(
    image, tile_mask, global_mask, tile_biased_mask, global_biased_mask,
    union_mask, *, title="", max_display_side=1600,
):
    """Compare original, individual branches, biased masks, and final union.

    Predictions remain at original resolution; only the displayed image and
    masks are reduced to make high-resolution comparisons manageable.
    """
    width, height = image.size
    masks = (tile_mask, global_mask, tile_biased_mask, global_biased_mask, union_mask)
    if any(np.asarray(mask).shape != (height, width) for mask in masks):
        raise ValueError("each prediction must match the original image dimensions")
    if max_display_side <= 0:
        raise ValueError("max_display_side must be positive")

    preview = image.copy()
    preview.thumbnail((max_display_side, max_display_side), Image.Resampling.LANCZOS)
    previews = [
        np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(
            preview.size, Image.Resampling.NEAREST
        ))
        for mask in masks
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    axes[0].imshow(preview)
    axes[0].set_title("Original")
    labels = ("Sliding window", "Whole-image resize", "Tile-biased fusion",
              "Resize-biased fusion", "Union of biased fusions")
    for ax, mask, label in zip(axes[1:], previews, labels):
        ax.imshow(preview)
        ax.imshow(mask, cmap="Reds", vmin=0, vmax=1,
                  alpha=mask.astype(np.float32) * 0.5)
        ax.set_title(label)
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_experiment3(
    model,
    device,
    image_paths,
    tile_size=640,
    stride=480,
    tile_bias_alpha=0.25,
    global_bias_alpha=0.75,
    tile_threshold=0.5,
    global_threshold=0.5,
):
    """Display per-image comparisons using the same model and preprocessing.

    Alpha parameters represent the whole-image weight. By default the tiled
    branch gets 75% weight in the first fusion, and the whole-image branch gets
    75% weight in the second. No files are written.
    """
    for name, value in (
        ("tile_bias_alpha", tile_bias_alpha),
        ("global_bias_alpha", global_bias_alpha),
        ("tile_threshold", tile_threshold),
        ("global_threshold", global_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    for image_path in map(Path, image_paths):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        print(f"Experiment 3: {image_path.name} ({image.width} x {image.height})")
        tile_mask, _, tile_probability = predict_with_tiles(
            image, model, device, tile_size=tile_size, stride=stride, return_prob=True
        )
        _, global_mask, global_probability = predict_with_resize(
            image, model, device, tile_size=tile_size, return_prob=True
        )
        tile_score, global_score, tile_biased_mask, global_biased_mask, union_mask = (
            fuse_biased_predictions(
                global_probability, tile_probability,
                tile_bias_alpha=tile_bias_alpha,
                global_bias_alpha=global_bias_alpha,
                tile_threshold=tile_threshold,
                global_threshold=global_threshold,
            )
        )
        show_dual_fusion_comparison(
            image, tile_mask, global_mask, tile_biased_mask, global_biased_mask,
            union_mask,
            title=(f"{image_path.name} | global weights: {tile_bias_alpha:g}, "
                   f"{global_bias_alpha:g}; thresholds: {tile_threshold:g}, "
                   f"{global_threshold:g}"),
        )
        del image, tile_mask, global_mask, tile_probability, global_probability
        del tile_score, global_score, tile_biased_mask, global_biased_mask, union_mask
