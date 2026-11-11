"""Experiment 4: direct inference at the original image resolution.

No input resizing, tiling, or padding. CUDA autocast is enabled by default;
display copies are reduced only after prediction. No files are written.
LLM assistance: generated with Codex on 2026-11-09 and tested locally.
"""

from project_paths import PROJECT_ROOT

from contextlib import nullcontext
import gc
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF

from experiment1 import MEAN, STD


@torch.inference_mode()
def predict_full_image(image, model, device, use_amp=True):
    """Return an original-resolution binary mask and timing/memory statistics.

    AMP changes numerical precision, not image dimensions. CPU inference uses
    float32. CUDA peak allocation includes all live tensors in this process,
    not just the model supplied here; it is not total system GPU memory usage.
    CUDA out-of-memory errors propagate to the caller.
    """
    device = torch.device(device)
    model.eval()
    image = image.convert("RGB")
    amp_enabled = use_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    inputs = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD)
    inputs = inputs.unsqueeze(0).to(device)
    autocast = (torch.amp.autocast("cuda", dtype=torch.float16)
                if amp_enabled else nullcontext())
    with autocast:
        logits = model(inputs)
    expected = (1, 2, image.height, image.width)
    if tuple(logits.shape) != expected:
        raise ValueError(f"Expected model output {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite model output; retry with use_amp=False")
    prediction = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    statistics = {
        "width": image.width,
        "height": image.height,
        "seconds": perf_counter() - started,
        "precision": "AMP float16" if amp_enabled else "float32",
        "device": str(device),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
    }
    return prediction, statistics


def show_full_image_prediction(image, prediction, *, title="", max_display_side=1600):
    """Show original and prediction; downsample display copies only."""
    if prediction.shape != (image.height, image.width):
        raise ValueError("prediction must match the original image dimensions")
    if max_display_side <= 0:
        raise ValueError("max_display_side must be positive")
    preview = image.copy()
    preview.thumbnail((max_display_side, max_display_side), Image.Resampling.LANCZOS)
    mask_preview = np.asarray(Image.fromarray(prediction).resize(
        preview.size, Image.Resampling.NEAREST,
    ))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax in axes:
        ax.imshow(preview)
        ax.axis("off")
    axes[0].set_title("Original")
    axes[1].imshow(mask_preview, cmap="Reds", vmin=0, vmax=1,
                   alpha=mask_preview.astype(np.float32) * 0.5)
    axes[1].set_title("Native-resolution direct inference")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_experiment4(model, device, image_paths, use_amp=True):
    """Predict one full image at a time, visualize successes, and report OOMs.

    A failed image is explicitly skipped without resizing or CPU fallback.
    Return a small per-image log; full-resolution masks are released after use.
    """
    device = torch.device(device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU: {torch.cuda.get_device_name(device)} | "
              f"free {free / 1024**3:.2f} / total {total / 1024**3:.2f} GiB")
    results = []
    for image_path in map(Path, image_paths):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        print(f"Experiment 4: {image_path.name} ({image.width} x {image.height})", flush=True)
        failed = False
        try:
            prediction, statistics = predict_full_image(image, model, device, use_amp)
        except torch.cuda.OutOfMemoryError:
            failed = True
        # Leave the exception handler before releasing memory held by its traceback.
        if failed:
            gc.collect()
            torch.cuda.empty_cache()
            print("  CUDA OOM: native-resolution inference skipped. "
                  "No resized result was substituted.")
            results.append({"image": image_path.name, "status": "cuda_oom",
                            "width": image.width, "height": image.height})
            del image
            continue
        peak = statistics["peak_allocated_gib"]
        memory_log = f" | process peak tensor allocation {peak:.2f} GiB" if peak is not None else ""
        print(f"  {statistics['precision']} | {statistics['seconds']:.2f} s{memory_log}")
        show_full_image_prediction(
            image, prediction,
            title=(f"{image_path.name} | input {image.width} x {image.height} | "
                   f"{statistics['precision']}"),
        )
        results.append({"image": image_path.name, "status": "ok", **statistics})
        del image, prediction
    successes = sum(result["status"] == "ok" for result in results)
    print(f"Experiment 4 completed: {successes}/{len(results)} original-resolution images succeeded.")
    return results
