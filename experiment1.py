"""Experiment 1: compare tiled and whole-image-resize segmentation.

Both methods use the same binary model and ImageNet normalization.  Tiled
inference averages overlapping logits; whole-image inference resizes logits
back to the original image before choosing a class.  The optional probability
maps allow experiment 2 to combine the two methods without changing them.
"""

from pathlib import Path, WindowsPath

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from script import ResNetUNet


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_model(checkpoint_path, device):
    """Load the project's binary checkpoint without initializing new weights.

    The current project checkpoint contains a WindowsPath metadata value. The
    restricted safe-globals context permits that type while retaining
    ``weights_only=True``. Only load checkpoints from a trusted source.
    """
    with torch.serialization.safe_globals([WindowsPath]):
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )

    if checkpoint.get("num_classes") != 2:
        raise ValueError("This experiment requires a two-class checkpoint.")
    model = ResNetUNet(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model


def get_tile_starts(length, tile_size, stride):
    """Cover one image axis, aligning the last window with its far edge."""
    if length <= 0 or tile_size <= 0 or not 0 < stride <= tile_size:
        raise ValueError("length and tile_size must be positive; 0 < stride <= tile_size")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.inference_mode()
def predict_with_tiles(image, model, device, tile_size=640, stride=480, return_prob=False):
    """Predict from overlapping tiles and optionally return penguin probability.

    Returns ``(mask, tiles)`` or ``(mask, tiles, probability)``. Masks and
    probabilities are NumPy arrays in original image coordinates. Each tile is
    ``(left, top, width, height)``. Predictions from overlapping windows are
    combined as equally weighted logits before sigmoid/argmax.
    """
    model.eval()
    image = image.convert("RGB")
    width, height = image.size
    xs = get_tile_starts(width, tile_size, stride)
    ys = get_tile_starts(height, tile_size, stride)
    logit_sum = torch.zeros((2, height, width), dtype=torch.float32)
    count = torch.zeros((height, width), dtype=torch.float32)
    tiles = []
    total_tiles = len(xs) * len(ys)
    print(f"  Number of tiles: {total_tiles}")

    for top in ys:
        for left in xs:
            patch_width = min(tile_size, width - left)
            patch_height = min(tile_size, height - top)
            patch = image.crop((left, top, left + patch_width, top + patch_height))
            patch_tensor = TF.to_tensor(patch)
            patch_tensor = F.pad(
                patch_tensor, (0, tile_size - patch_width, 0, tile_size - patch_height)
            )
            patch_tensor = TF.normalize(patch_tensor, mean=MEAN, std=STD)
            logits = model(patch_tensor.unsqueeze(0).to(device))[0].float().cpu()
            if logits.shape != (2, tile_size, tile_size):
                raise ValueError(f"Expected model output (2, {tile_size}, {tile_size}); got {tuple(logits.shape)}")
            logits = logits[:, :patch_height, :patch_width]
            logit_sum[:, top:top + patch_height, left:left + patch_width] += logits
            count[top:top + patch_height, left:left + patch_width] += 1
            tiles.append((left, top, patch_width, patch_height))
            if len(tiles) % 25 == 0:
                print(f"  Processed {len(tiles)}/{total_tiles} tiles")

    if (count == 0).any():
        raise RuntimeError("Some image pixels were not covered by a tile.")
    logit_sum /= count.unsqueeze(0)
    mask = logit_sum.argmax(dim=0).to(torch.uint8).numpy()
    if return_prob:
        probability = torch.sigmoid(logit_sum[1] - logit_sum[0]).numpy()
        return mask, tiles, probability
    return mask, tiles


@torch.inference_mode()
def predict_with_resize(image, model, device, tile_size=640, return_prob=False):
    """Predict one stretched image and restore logits to original resolution.

    Returns ``(resized_image, mask)`` or ``(resized_image, mask, probability)``.
    Both output arrays use the original image coordinates; probabilities are
    computed only after bilinearly restoring the class logits.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    model.eval()
    image = image.convert("RGB")
    width, height = image.size
    resized = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    inputs = TF.normalize(TF.to_tensor(resized), mean=MEAN, std=STD)
    logits = model(inputs.unsqueeze(0).to(device)).float().cpu()
    if logits.shape != (1, 2, tile_size, tile_size):
        raise ValueError(f"Expected model output (1, 2, {tile_size}, {tile_size}); got {tuple(logits.shape)}")
    restored = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    mask = restored.argmax(dim=1)[0].to(torch.uint8).numpy()
    if return_prob:
        probability = torch.sigmoid(restored[0, 1] - restored[0, 0]).numpy()
        return resized, mask, probability
    return resized, mask


def show_overlay(image, prediction, title, tiles=None, max_display_side=1600):
    """Display an overlay and optional tile borders in original coordinates.

    Only the displayed image and mask are downsampled; inference and tile
    positions remain at original resolution. Nearest-neighbor mask resizing
    preserves class labels.
    """
    width, height = image.size
    if prediction.shape != (height, width):
        raise ValueError("Prediction shape does not match the image.")
    if max_display_side <= 0:
        raise ValueError("max_display_side must be positive")
    scale = min(1.0, max_display_side / max(width, height))
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if display_size == image.size:
        displayed_image = image
        displayed_mask = prediction
    else:
        displayed_image = image.resize(display_size, Image.Resampling.LANCZOS)
        displayed_mask = np.asarray(
            Image.fromarray(prediction.astype(np.uint8)).resize(
                display_size, Image.Resampling.NEAREST
            )
        )
    extent = (-0.5, width - 0.5, height - 0.5, -0.5)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(displayed_image, extent=extent)
    ax.imshow(displayed_mask, cmap="Reds", vmin=0, vmax=1, extent=extent,
              interpolation="nearest", alpha=displayed_mask.astype(np.float32) * 0.5)
    for left, top, tile_width, tile_height in (() if tiles is None else tiles):
        ax.add_patch(Rectangle((left - 0.5, top - 0.5), tile_width, tile_height,
                               fill=False, edgecolor="cyan", linewidth=0.8, alpha=0.8))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_experiment1(model, device, image_paths, tile_size=640, stride=480):
    """Display one tiled and one resized overlay per image, without saving files."""
    for image_path in map(Path, image_paths):
        print(f"Experiment 1: {image_path.name}")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tiled_mask, tiles = predict_with_tiles(image, model, device, tile_size, stride)
        show_overlay(image, tiled_mask, f"{image_path.name} | Sliding window", tiles=tiles)
        del tiled_mask, tiles
        _, resized_mask = predict_with_resize(image, model, device, tile_size)
        show_overlay(image, resized_mask, f"{image_path.name} | Whole-image resize")
        del image, resized_mask
