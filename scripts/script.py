from project_paths import PROJECT_ROOT
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from torchvision.models import resnet34, ResNet34_Weights

from collections import Counter
import csv
import io
import random
from zipfile import ZipFile
from contextlib import nullcontext
from time import perf_counter
from scipy import ndimage
from animal_data import CLASS_NAMES, fit_image_and_mask

# --- shared evaluation (moved from metrics.py; LLM-assisted 2026-11-09) ---

def binary_confusion_counts(predictions, targets, ignore_index=255):
    """Count TP/FP/FN/TN from aligned class-ID masks, not logits/probabilities.

    Accepts tensors or NumPy arrays of the same shape, e.g. [B, H, W].
    Pixels whose target equals ignore_index contribute to no count.
    """
    predictions = torch.as_tensor(predictions)
    targets = torch.as_tensor(targets, device=predictions.device)
    if predictions.shape != targets.shape:
        raise ValueError("prediction and target masks must have the same shape")
    valid = targets != ignore_index
    predictions, targets = predictions[valid], targets[valid]
    if not (((predictions == 0) | (predictions == 1)).all()
            and ((targets == 0) | (targets == 1)).all()):
        raise ValueError("non-ignored masks must contain only class IDs 0 and 1")
    predicted, labeled = predictions == 1, targets == 1
    values = torch.stack([
        (predicted & labeled).sum(),
        (predicted & ~labeled).sum(),
        (~predicted & labeled).sum(),
        (~predicted & ~labeled).sum(),
    ]).tolist()
    return dict(zip(("tp", "fp", "fn", "tn"), values))


def metrics_from_counts(counts):
    """Compute pooled pixel metrics; foreground Dice is also binary F1."""
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    penguin_iou = ratio(tp, tp + fp + fn)
    background_iou = ratio(tn, tn + fp + fn)
    class_ious = [v for v in (background_iou, penguin_iou) if math.isfinite(v)]
    return {
        "penguin_iou": penguin_iou,
        "background_iou": background_iou,
        "miou": sum(class_ious) / len(class_ious) if class_ious else float("nan"),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def compute_binary_metrics(predictions, targets, ignore_index=255):
    """Evaluate one mask pair; use pooled counts for multi-image evaluation."""
    return metrics_from_counts(binary_confusion_counts(predictions, targets, ignore_index))


def metrics_from_confusion(confusion):
    """Rows are true labels, columns predicted labels; absent ratios are NaN."""
    matrix = torch.as_tensor(confusion, dtype=torch.float64)
    tp = matrix.diag()
    fp, fn = matrix.sum(0) - tp, matrix.sum(1) - tp

    def divide(numerator, denominator):
        return torch.where(denominator > 0, numerator / denominator, float("nan"))

    iou = divide(tp, tp + fp + fn)
    dice = divide(2 * tp, 2 * tp + fp + fn)
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    result = {
        "iou_per_class": iou.tolist(), "dice_per_class": dice.tolist(),
        "precision_per_class": precision.tolist(), "recall_per_class": recall.tolist(),
        "miou": torch.nanmean(iou).item(),
        "foreground_miou": torch.nanmean(iou[1:]).item(),
        "dice": torch.nanmean(dice[1:]).item(),
        "precision": torch.nanmean(precision[1:]).item(),
        "recall": torch.nanmean(recall[1:]).item(),
        "pixel_accuracy": divide(tp.sum(), matrix.sum()).item(),
        "background_iou": iou[0].item(), "penguin_iou": iou[1].item(),
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }
    if len(matrix) == 2:
        result.update(tn=int(matrix[0, 0]), fp=int(matrix[0, 1]),
                      fn=int(matrix[1, 0]), tp=int(matrix[1, 1]))
    return result


@torch.no_grad()
def evaluate_segmentation(model, loader, device, criterion=None, ignore_index=255,
                          num_classes=None, amp=False, channels_last=False,
                          progress_every=0):
    """Evaluate direct inference on a labeled loader without training/saving.

    The loader supplies normalized images and class-ID masks. The optional loss
    should use mean reduction and the same ignore_index. Its batch means are
    image-weighted, matching Trainer's existing validation-loss convention.
    The model's previous training/eval mode is restored even if evaluation fails.
    """
    training_states = {layer: layer.training for layer in model.modules()}
    model.eval()
    confusion = None
    total_images = loss_images = 0
    total_loss = 0.0
    try:
        for step, (images, targets) in enumerate(loader, 1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=amp and torch.device(device).type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets) if criterion is not None else None
            if logits.ndim != 4 or logits.shape[1] < 2:
                raise ValueError("expected model logits with shape [B, C, H, W], C >= 2")
            classes = logits.shape[1]
            if num_classes is not None and classes != num_classes:
                raise ValueError("model output does not match num_classes")
            if confusion is None:
                confusion = torch.zeros((classes, classes), dtype=torch.int64)
            elif len(confusion) != classes:
                raise ValueError("model class count changed during evaluation")
            predictions = logits.argmax(dim=1)
            if predictions.shape != targets.shape:
                raise ValueError("predictions and targets must have matching dimensions")
            valid = targets != ignore_index
            labeled = targets[valid]
            if ((labeled < 0) | (labeled >= classes)).any():
                raise ValueError("target contains an unknown class ID")
            valid_count = labeled.numel()
            batch_confusion = torch.bincount(
                (labeled * classes + predictions[valid]).reshape(-1),
                minlength=classes * classes,
            ).reshape(classes, classes)
            confusion += batch_confusion.cpu()
            batch_size = images.shape[0]
            total_images += batch_size
            # CrossEntropyLoss(mean) is undefined for a fully ignored batch.
            if criterion is not None and valid_count > 0:
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite validation loss: {loss.item()}")
                total_loss += loss.item() * batch_size
                loss_images += batch_size
            if progress_every and step % progress_every == 0:
                print(f"  validation {total_images}/{len(loader.dataset)}", flush=True)
    finally:
        for layer, training in training_states.items():
            layer.training = training
    if total_images == 0 or confusion.sum() == 0:
        raise ValueError("evaluation requires images with non-ignored labeled pixels")
    result = metrics_from_confusion(confusion)
    result["num_images"] = total_images
    result["num_valid_pixels"] = int(confusion.sum())
    if criterion is not None:
        result["loss"] = total_loss / loss_images
    return result


@torch.inference_mode()
def evaluate_background(model, loader, device, *, amp=False, channels_last=False):
    """Measure false animals on explicitly labeled background (0), ignoring 255.

    Image alarm means at least one non-ignored pixel predicted as an animal.
    This does not measure animal recall or certify unlabeled pixels as empty.
    LLM-assisted experiment 10, 2026-11-10.
    """
    states = {layer: layer.training for layer in model.modules()}
    model.eval()
    predicted_counts = torch.zeros(5, dtype=torch.int64)
    image_fractions = []
    alarms = 0
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if not ((targets == 0) | (targets == 255)).all():
                raise ValueError("Background evaluation expects only labels 0/255")
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(torch.device(device).type, dtype=torch.float16,
                                enabled=amp and torch.device(device).type == "cuda"):
                logits = model(images)
            if logits.shape[:2] != (len(images), 5) or logits.shape[-2:] != targets.shape[-2:]:
                raise ValueError("Expected five-class, spatially aligned background logits")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite background logits")
            prediction = logits.argmax(1)
            valid = targets == 0
            predicted_counts += torch.bincount(prediction[valid], minlength=5).cpu()
            for pred, keep in zip(prediction, valid):
                denominator = int(keep.sum())
                if denominator == 0:
                    raise ValueError("Background image has no explicitly labeled pixels")
                errors = int(((pred != 0) & keep).sum())
                image_fractions.append(errors / denominator)
                alarms += int(errors > 0)
    finally:
        for layer, state in states.items():
            layer.training = state
    total = int(predicted_counts.sum())
    if not total:
        raise ValueError("Empty background evaluation")
    errors = int(predicted_counts[1:].sum())
    return dict(false_positive_fraction=errors/total, false_positive_pixels=errors,
        num_valid_pixels=total, num_images=len(image_fractions),
        false_alarm_image_fraction=alarms/len(image_fractions),
        mean_image_false_positive_fraction=sum(image_fractions)/len(image_fractions),
        predicted_pixels_per_class=predicted_counts.tolist())


# --- resnet blocks ---

def enlarge_spatial_convolutions(module, kernel_size):
    """Enlarge 3x3 convolutions only; retain stem 7x7 and projection/head 1x1.

    Center-copy weights and zero the new outer ring: at initialization this
    preserves the original function (within floating point precision).
    LLM-assisted experimental implementation, 2026-11-09.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            expanded = nn.Conv2d(
                child.in_channels, child.out_channels, kernel_size,
                stride=child.stride, padding=kernel_size // 2,
                dilation=child.dilation, groups=child.groups, bias=child.bias is not None,
                padding_mode=child.padding_mode,
                device=child.weight.device, dtype=child.weight.dtype,
            )
            offset = (kernel_size - 3) // 2
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :, offset:offset+3, offset:offset+3].copy_(child.weight)
                if child.bias is not None:
                    expanded.bias.copy_(child.bias)
            setattr(module, name, expanded)
        else:
            enlarge_spatial_convolutions(child, kernel_size)


def load_initial_animal_model(checkpoint_path, num_classes=5, kernel_size=3, seed=42):
    """Reuse binary features, fresh classification head, matching kernels by padding.

    Construct the original model first so all kernel candidates have exactly
    the same fresh head and center weights, independent of random-number draws.
    """
    from pathlib import WindowsPath
    torch.manual_seed(seed)
    model = ResNetUNet(num_classes=num_classes, pretrained=False)
    with torch.serialization.safe_globals([WindowsPath]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    weights = {key: value for key, value in checkpoint["model_state"].items()
               if not key.startswith("head.")}
    result = model.load_state_dict(weights, strict=False)
    if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
        raise ValueError(f"Unexpected initial checkpoint mismatch: {result}")
    if kernel_size != 3:
        if kernel_size not in (5, 7):
            raise ValueError("kernel_size must be 3, 5 or 7")
        enlarge_spatial_convolutions(model, kernel_size)
        model.kernel_size = kernel_size
    return model

class ConvBlock(nn.Module):
    """
    Fuse features with two convolution layers.

    Input:  [B, in_channels,  H, W]
    Output: [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size, padding=kernel_size // 2), 
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DecoderBlock(nn.Module):
    """
    Upsample the decoder feature, concatenate a skip feature,
    then fuse them with ConvBlock.

    x:    [B, x_channels,    H1, W1]
    skip: [B, skip_channels, H2, W2]

    output: [B, out_channels, H2, W2]
    """

    def __init__(
        self,
        x_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.fuse = ConvBlock(
            in_channels=x_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        
        upsample = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        conv_input = torch.cat([upsample, skip], dim=1)

        return self.fuse(conv_input)

class ResNetUNet(nn.Module):
    """
    A U-Net-style segmentation model with a ResNet-34 encoder.

    Input:  [B, 3, H, W]
    Output: [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()

        if kernel_size not in (3, 5, 7):
            raise ValueError("kernel_size must be 3, 5 or 7")
        self.kernel_size = kernel_size

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        if kernel_size != 3:
            enlarge_spatial_convolutions(backbone, kernel_size)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.dec4 = DecoderBlock(512, 256, 256, kernel_size)
        self.dec3 = DecoderBlock(256, 128, 128, kernel_size)
        self.dec2 = DecoderBlock(128, 64, 64, kernel_size)
        self.dec1 = DecoderBlock(64, 64, 64, kernel_size)

        self.head = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        deepest = self.enc4(s3)

        d4 = self.dec4(deepest, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        upsample = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.head(upsample)

# --- trainer ---

class Trainer:
    """Reusable binary/multiclass training with optional AMP and frozen BN.

    Original call signature remains valid. LLM-assisted consolidation, 2026-11-09.
    model.train() changes layer mode; Trainer.fit() performs optimization.
    """
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer,
                 device, checkpoint_dir, *, class_names=("background", "penguin"),
                 checkpoint_name="best.pt", dataset_metadata=None, amp=False,
                 bn_policy="normal", channels_last=False, gradient_clip=None,
                 log_every=100, background_loader=None):
        if bn_policy not in ("normal", "frozen_stats"):
            raise ValueError("bn_policy must be normal or frozen_stats")
        self.model, self.train_loader, self.valid_loader = model, train_loader, valid_loader
        self.background_loader = background_loader
        self.criterion, self.optimizer = criterion, optimizer
        self.device, self.checkpoint_dir = torch.device(device), Path(checkpoint_dir)
        self.class_names, self.checkpoint_name = tuple(class_names), checkpoint_name
        self.dataset_metadata = dataset_metadata or {}
        self.selection_metric = "penguin_iou" if len(self.class_names) == 2 else "foreground_miou"
        self.amp = bool(amp and self.device.type == "cuda")
        self.bn_policy, self.channels_last = bn_policy, channels_last
        self.gradient_clip, self.log_every = gradient_clip, log_every
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp, init_scale=1024)
        self.best_iou, self.epoch, self.history = -float("inf"), 0, []
        self.config = {"amp": self.amp, "bn_policy": bn_policy, "channels_last": channels_last,
                       "kernel_size": getattr(model, "kernel_size", 3),
                       "batch_size": train_loader.batch_size, "gradient_clip": gradient_clip}
        if channels_last:
            self.model.to(memory_format=torch.channels_last)

    def set_training_mode(self):
        self.model.train()
        if self.bn_policy == "frozen_stats":
            for layer in self.model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()  # Affine gamma/beta still learn.

    def train_step(self, images, masks):
        images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(images)
            loss = self.criterion(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        self.scaler.scale(loss).backward()
        if self.gradient_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.detach().item()

    def train_one_epoch(self):
        self.set_training_mode()
        started, total_loss, total_images = perf_counter(), 0.0, 0
        epoch_size = len(self.train_loader.sampler)
        for step, (images, masks) in enumerate(self.train_loader, 1):
            loss = self.train_step(images, masks)
            total_loss += loss * len(images)
            total_images += len(images)
            if self.log_every and (step == 1 or step % self.log_every == 0 or step == len(self.train_loader)):
                print(f"Epoch {self.epoch:02d} | train {total_images}/{epoch_size} "
                      f"| loss={total_loss/total_images:.4f} "
                      f"| {total_images/(perf_counter()-started):.2f} img/s", flush=True)
        if not total_images:
            raise ValueError("Training dataset is empty")
        return total_loss / total_images

    @torch.no_grad()
    def validate(self):
        self.validation_metrics = evaluate_segmentation(
            self.model, self.valid_loader, self.device, criterion=self.criterion,
            ignore_index=getattr(self.criterion, "ignore_index", 255),
            num_classes=len(self.class_names), amp=self.amp,
            channels_last=self.channels_last, progress_every=self.log_every,
        )
        if self.background_loader is not None:
            background = evaluate_background(self.model,self.background_loader,self.device,
                amp=self.amp,channels_last=self.channels_last)
            self.validation_metrics["background_holdout"] = background
            print(f"Background valid | false-positive pixels={background['false_positive_fraction']:.4%} "
                  f"| images with any false positive={background['false_alarm_image_fraction']:.2%}",flush=True)
        return self.validation_metrics["loss"], self.validation_metrics[self.selection_metric]

    def _checkpoint(self):
        sampler = getattr(self.train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        return dict(epoch=self.epoch, model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(), scaler_state=self.scaler.state_dict(),
                    best_iou=self.best_iou, history=self.history, config=self.config,
                    valid_loss=self.validation_metrics["loss"],
                    valid_penguin_iou=self.validation_metrics["penguin_iou"],
                    valid_score=self.validation_metrics[self.selection_metric],
                    selection_metric=self.selection_metric, valid_metrics=self.validation_metrics,
                    num_classes=len(self.class_names), class_names=list(self.class_names),
                    dataset_metadata=self.dataset_metadata,
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_states=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                    sampler_rng_state=generator.get_state() if generator is not None else None)

    def _write_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(self._checkpoint(), temporary)
        temporary.replace(path)

    def save_checkpoint(self, valid_loss, valid_iou):
        if math.isfinite(valid_iou) and valid_iou > self.best_iou:
            self.best_iou = valid_iou
            self._write_checkpoint(self.checkpoint_dir / self.checkpoint_name)
            print("Saved new best:", self.checkpoint_dir / self.checkpoint_name, flush=True)

    def fit(self, num_epochs):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(num_epochs):
            self.epoch += 1
            started = perf_counter()
            train_loss = self.train_one_epoch()
            train_seconds = perf_counter() - started
            valid_loss, valid_score = self.validate()
            scores = self.validation_metrics
            self.history.append(dict(
                epoch=self.epoch, train_loss=train_loss, valid_loss=valid_loss,
                validate_loss=valid_loss, validate_iou=valid_score,
                selection_metric=self.selection_metric, foreground_miou=scores["foreground_miou"],
                iou_per_class=scores["iou_per_class"], valid_metrics=scores,
                train_seconds=train_seconds, epoch_seconds=perf_counter()-started,
                **{f"validate_{key}": scores[key] for key in ("dice", "precision", "recall", "pixel_accuracy", "miou")},
            ))
            (self.checkpoint_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            print(f"EPOCH {self.epoch:02d} | train loss={train_loss:.4f} | valid loss={valid_loss:.4f} "
                  f"| {self.selection_metric}={valid_score:.4f} | Dice={scores['dice']:.4f} "
                  f"| precision={scores['precision']:.4f} | recall={scores['recall']:.4f} "
                  f"| pixel accuracy={scores['pixel_accuracy']:.4f}", flush=True)
            print("Class IoU: " + " | ".join(f"{n}={v:.4f}" for n, v in zip(self.class_names, scores["iou_per_class"])), flush=True)
            self.save_checkpoint(valid_loss, valid_score)
            self._write_checkpoint(self.checkpoint_dir / "last.pt")
        return self.history

    def resume(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        for key in ("kernel_size", "bn_policy", "batch_size", "channels_last"):
            if checkpoint["config"].get(key) != self.config.get(key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.epoch, self.history = checkpoint["epoch"], checkpoint["history"]
        self.best_iou = checkpoint["best_iou"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        sampler_state = checkpoint.get("sampler_rng_state")
        if sampler_state is not None:
            self.train_loader.sampler.generator.set_state(sampler_state.cpu())
        return self


# --- dataset ---

class PenguinDataset(Dataset):
    """
    Read one Roboflow COCO split and build binary semantic masks.

    Each sample:
        image: [3, H, W], float32, ImageNet normalized
        mask:  [H, W],    int64, values 0 or 1

    Important:
        0 currently means "not annotated as penguin".
        Some of those pixels may contain unannotated penguins.
    """

    def __init__(self, split_dir: str | Path):
        self.split_dir = Path(split_dir)
        annotation_path = self.split_dir / "_annotations.coco.json"

        with annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]

        # Build the index once, instead of scanning every annotation
        # whenever __getitem__ is called.
        self.annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(
                annotation
            )

    def __len__(self) -> int:
        return len(self.images)

    def get_raw(self, index: int):
        """
        Return an RGB PIL image and a PIL mask before normalization.
        Useful for visual inspection.
        """
        info = self.images[index]
        image_path = self.split_dir / info["file_name"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        width, height = image.size

        if (width, height) != (info["width"], info["height"]):
            raise ValueError(f"Image size does not match JSON: {image_path}")

        # Start with zeros. These are provisional background pixels.
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        annotations = self.annotations_by_image[info["id"]]

        for annotation in annotations:
            # In this downloaded dataset, category_id 1 is Penguin.
            if annotation["category_id"] != 1:
                continue

            # This dataset stores segmentation as polygon lists.
            for polygon in annotation["segmentation"]:
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.polygon(points, fill=1)

        return image, mask

    def __getitem__(self, index: int):
        image, mask = self.get_raw(index)

        # Image: uint8 RGB -> float32 tensor in [0, 1].
        image_tensor = TF.to_tensor(image)

        # Match the normalization associated with the pretrained encoder.
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Mask contains class IDs: preserve 0 and 1 as integers.
        mask_tensor = torch.from_numpy(
            np.array(mask, dtype=np.int64, copy=True)
        )

        return image_tensor, mask_tensor

# --- visualizer ---

def visualize_penguin_image(image_path: str | Path):
    """
    Show an image, its polygon-derived mask, and an overlay.

    The image must belong to a split directory containing
    _annotations.coco.json.
    """
    image_path = Path(image_path)
    dataset = PenguinDataset(image_path.parent)

    # Match by filename within this split's JSON.
    matches = [
        i
        for i, info in enumerate(dataset.images)
        if info["file_name"] == image_path.name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one matching image, found {len(matches)}: {image_path}"
        )

    image, mask = dataset.get_raw(matches[0])
    mask_array = np.asarray(mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")

    axes[1].imshow(mask_array, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Annotated mask")

    axes[2].imshow(image)
    axes[2].imshow(
        mask_array,
        cmap="Reds",
        alpha=mask_array * 0.5,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def show_validation_prediction(model, dataset, index, device):
    # Raw image and annotation mask for display.
    image, raw_mask = dataset.get_raw(index)

    # Normalized tensor for the model.
    image_tensor, _ = dataset[index]

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")

    axes[1].imshow(image)
    axes[1].imshow(raw_mask, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Existing annotation")

    axes[2].imshow(image)
    axes[2].imshow(prediction, cmap="Blues", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Model prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# --- dataset preparation (moved from prepare_datasets.py; never run on import) ---

SEED = 42
SIZE = 640


def save_sample(root, split, name, image, mask, **metadata):
    folder = root / split
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "masks").mkdir(exist_ok=True)
    image_path, mask_path = folder / "images" / f"{name}.jpg", folder / "masks" / f"{name}.png"
    if image_path.exists() or mask_path.exists():
        raise FileExistsError(f"Prepared output exists; refusing to overwrite {name}")
    image.save(image_path, quality=95)
    mask.save(mask_path)
    return {"image": image_path.relative_to(folder).as_posix(),
            "mask": mask_path.relative_to(folder).as_posix(), **metadata}


def save_manifests(root, records, policy):
    for split in ("train", "valid", "test"):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        data = {"class_names": list(CLASS_NAMES), "size": SIZE,
                "split_policy": policy, "samples": records[split]}
        data["mask_policy"] = (
            "raw>=128 -> bird=2; raw==0 -> background=0; 0<raw<128 -> ignore=255"
            if root.name == "dataset_birds" else
            "JPEG silhouette >8 support, >4 outer support; largest component+hole fill; "
            "erode inner 2px, dilate outer 1px; boundary/padding=255 (approximate)"
        )
        (folder / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {split: len(records[split]) for split in records}


def bird_mask(raw):
    """Threshold supplied soft masks; ignore nonzero low-confidence pixels."""
    labels = np.full(raw.shape, 255, dtype=np.uint8)
    labels[raw == 0], labels[raw >= 128] = 0, 2
    return Image.fromarray(labels)


def prepare_birds(root):
    cub = root / "CUB_200_2011"
    images = {int(line.split()[0]): line.split(maxsplit=1)[1]
              for line in (cub / "images.txt").read_text().splitlines()}
    original_split = dict(map(lambda line: map(int, line.split()),
                              (cub / "train_test_split.txt").read_text().splitlines()))
    species = dict(map(lambda line: map(int, line.split()),
                       (cub / "image_class_labels.txt").read_text().splitlines()))
    groups = defaultdict(list)
    for index in images:
        if original_split[index] == 1:
            groups[species[index]].append(index)
    valid_ids = set()
    rng = random.Random(SEED)
    for key in sorted(groups):
        group = sorted(groups[key])
        rng.shuffle(group)
        valid_ids.update(group[:max(1, round(len(group) * .2))])
    records = defaultdict(list)
    for count, (index, name) in enumerate(sorted(images.items()), 1):
        split = "test" if not original_split[index] else "valid" if index in valid_ids else "train"
        source_path = cub / "images" / name
        mask_path = root / "segmentations" / Path(name).with_suffix(".png")
        with Image.open(source_path) as source, Image.open(mask_path) as source_mask:
            image = source.convert("RGB")
            raw = np.asarray(source_mask.convert("L"))
            image, mask = fit_image_and_mask(image, bird_mask(raw), SIZE)
        record = save_sample(root, split, f"cub_{index:05d}", image, mask,
                             sample_class=2, source="CUB-200-2011", mask_kind="provided_gray_mask_threshold128",
                             original_image=source_path.relative_to(root).as_posix(), species_id=species[index])
        records[split].append(record)
        if count % 1000 == 0:
            print(f"Birds: {count}/{len(images)}", flush=True)
    return save_manifests(root, records, "Official test retained; seeded per-species 20% of official train used as valid (seed=42).")


def largest_filled_region(binary):
    components, count = ndimage.label(binary)
    if not count:
        raise ValueError("Segmented seal JPEG has no foreground")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(components == sizes.argmax())


def recover_seal_mask(segmented):
    """Conservative trimap from JPEG appearance, not an original human mask.

    JPEG black-background artifacts and dark animal pixels prevent exact recovery.
    Fill enclosed holes; retain the largest body component; ignore boundary band.
    """
    intensity = np.asarray(segmented.convert("RGB")).max(axis=2)
    # A high foreground threshold erases dark fur. Recover the silhouette first,
    # then exclude its edge, rather than selecting only the brighter body pixels.
    support = largest_filled_region(intensity > 8)
    weak = largest_filled_region(intensity > 4) | support
    inner = ndimage.binary_erosion(support, iterations=2)
    outer = ndimage.binary_dilation(weak, iterations=1)
    labels = np.zeros(intensity.shape, dtype=np.uint8)
    labels[outer], labels[inner] = 255, 3
    return Image.fromarray(labels)


def prepare_seals(root):
    records = defaultdict(list)
    archive = root / "SealID" / "full images.zip"
    with ZipFile(archive) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("full images/annotation.csv").decode("utf-8-sig"))))
        for count, row in enumerate(rows, 1):
            split = {"training": "train", "validation": "valid", "testing": "test"}[row["segmentation_split"]]
            prefix = f"full images/{{}}_{row['reid_split']}/{row['file']}"
            with Image.open(io.BytesIO(z.read(prefix.format("source")))) as source, \
                 Image.open(io.BytesIO(z.read(prefix.format("segmented")))) as segmented:
                if source.size != segmented.size:
                    raise ValueError(f"Source/segmented image mismatch: {row['file']}")
                # Both JPEGs use the same reduced decoding size, preserving alignment.
                source.draft("RGB", (SIZE, SIZE))
                segmented.draft("RGB", (SIZE, SIZE))
                image = source.convert("RGB")
                segmented = segmented.convert("RGB")
                # Recover at working resolution; source archive retains original pixels.
                placeholder = Image.new("L", image.size)
                image, _ = fit_image_and_mask(image, placeholder, SIZE)
                resized, _ = fit_image_and_mask(segmented, placeholder, SIZE)
                # Ignore padded image space rather than label padding as scene background.
                _, valid = fit_image_and_mask(segmented, Image.new("L", segmented.size, 0), SIZE)
                padded = np.asarray(valid) == 255
                segmented_pixels = np.array(resized, copy=True)
                segmented_pixels[padded] = 0
                mask = recover_seal_mask(Image.fromarray(segmented_pixels))
                labels = np.array(mask, copy=True)
                labels[padded] = 255
                mask = Image.fromarray(labels)
            records[split].append(save_sample(
                root, split, Path(row["file"]).stem, image, mask, sample_class=3,
                source="SealID", mask_kind="recovered_black_background_jpeg",
                original_image=f"SealID/full images.zip::{prefix.format('source')}",
                individual_id=row["class_id"],
            ))
            if count % 200 == 0:
                print(f"Seals: {count}/{len(rows)}", flush=True)
    return save_manifests(root, records, "Official segmentation_split retained: training/validation/testing -> train/valid/test; reid_split is not used for splitting.")


def prepare_all_datasets():
    root = PROJECT_ROOT / "data"
    report = {"class_names": CLASS_NAMES, "size": SIZE, "seed": SEED}
    for name, prepare in (("dataset_birds", prepare_birds), ("dataset_seals", prepare_seals)):
        folder = root / name
        if all((folder / split / "manifest.json").exists() for split in ("train", "valid", "test")):
            report[name] = {split: len(json.loads((folder / split / "manifest.json").read_text())["samples"])
                            for split in ("train", "valid", "test")}
            print(f"{name}: existing manifests found; no files overwritten.", flush=True)
        else:
            report[name] = prepare(folder)
    (root / "data_preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


