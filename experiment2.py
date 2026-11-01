"""Experiment 2: fuse whole-image and tiled penguin predictions.

Both inputs use the same model and are mapped back to the original image.
The tiled path averages overlapping logits before converting to probabilities;
the whole-image path resizes logits before converting to probabilities. The
two foreground probabilities are then averaged, and thresholded only once.
These scores are not necessarily calibrated probabilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from experiment1 import predict_with_resize, predict_with_tiles


def fuse_probabilities(global_probability, tile_probability, alpha=0.5, threshold=0.5):
    """Fuse aligned foreground probabilities and return (probability, mask).

    ``alpha`` weights the resized whole-image path. ``1 - alpha`` weights the
    sliding-window path. The binary mask uses a strict ``>`` threshold, so an
    exact 0.5 tie is background, as with argmax over [background, penguin].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    global_probability = np.asarray(global_probability, dtype=np.float32)
    tile_probability = np.asarray(tile_probability, dtype=np.float32)
    if global_probability.ndim != 2 or tile_probability.ndim != 2:
        raise ValueError("probability maps must be two-dimensional")
    if global_probability.shape != tile_probability.shape:
        raise ValueError("probability maps must have the same original-image shape")
    if not np.isfinite(global_probability).all() or not np.isfinite(tile_probability).all():
        raise ValueError("probability maps must contain finite values")
    if (global_probability < 0).any() or (global_probability > 1).any():
        raise ValueError("global probabilities must be in [0, 1]")
    if (tile_probability < 0).any() or (tile_probability > 1).any():
        raise ValueError("tile probabilities must be in [0, 1]")

    fused_probability = alpha * global_probability + (1.0 - alpha) * tile_probability
    fused_mask = (fused_probability > threshold).astype(np.uint8)
    return fused_probability, fused_mask


def show_fusion_comparison(image, tile_mask, global_mask, fused_mask, *, title="", max_display_side=1600):
    """Show the original image and three overlays; resize only display copies."""
    width, height = image.size
    if any(np.asarray(mask).shape != (height, width) for mask in (tile_mask, global_mask, fused_mask)):
        raise ValueError("each prediction must match the original image dimensions")
    if max_display_side <= 0:
        raise ValueError("max_display_side must be positive")

    preview = image.copy()
    preview.thumbnail((max_display_side, max_display_side), Image.Resampling.LANCZOS)
    masks = [
        np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(
            preview.size, Image.Resampling.NEAREST
        ))
        for mask in (tile_mask, global_mask, fused_mask)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(preview)
    axes[0].set_title("Original")
    for ax, mask, label in zip(
        axes[1:], masks, ("Sliding window", "Whole-image resize", "Probability fusion")
    ):
        ax.imshow(preview)
        ax.imshow(mask, cmap="Reds", vmin=0, vmax=1, alpha=mask.astype(np.float32) * 0.5)
        ax.set_title(label)
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_experiment2(
    model,
    device,
    image_paths,
    tile_size=640,
    stride=480,
    alpha=0.5,
    threshold=0.5,
):
    """Compare both methods and their probability fusion for each image.

    No files are written. Each original-resolution result is released after
    its figure is displayed, to bound memory use on high-resolution images.
    """
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("alpha and threshold must be in [0, 1]")

    for image_path in image_paths:
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        print(f"Experiment 2: {image_path.name} ({image.width} x {image.height})")
        tile_mask, _, tile_probability = predict_with_tiles(
            image, model, device, tile_size=tile_size, stride=stride, return_prob=True
        )
        _, global_mask, global_probability = predict_with_resize(
            image, model, device, tile_size=tile_size, return_prob=True
        )
        fused_probability, fused_mask = fuse_probabilities(
            global_probability, tile_probability, alpha=alpha, threshold=threshold
        )
        show_fusion_comparison(
            image,
            tile_mask,
            global_mask,
            fused_mask,
            title=f"{image_path.name} | global weight={alpha:g}, threshold={threshold:g}",
        )
        del image, tile_mask, tile_probability, global_mask, global_probability
        del fused_probability, fused_mask
