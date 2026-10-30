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

# --- resnet blocks ---

class ConvBlock(nn.Module):
    """
    Fuse features with two convolution layers.

    Input:  [B, in_channels,  H, W]
    Output: [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1), 
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
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
    ):
        super().__init__()
        self.fuse = ConvBlock(
            in_channels=x_channels + skip_channels,
            out_channels=out_channels,
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
    ):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

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

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

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
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.valid_loader   = valid_loader
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_iou       = float("-inf")
        self.epoch          = 0 # completed epoches
        self.history        = [] # store loss history and matrices for visualization

    def train_one_epoch(self):
        """
        1. model.train()
        2. Iterate over train_loader
        3. Forward, loss, backward, optimizer.step()
        4. Return average training loss
        """
        self.model.train()
        
        total_loss = 0.0
        total_images = 0
    
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True) # note: care care care.

            logits = self.model(images)
            loss = self.criterion(logits, masks)
    
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {loss.item()}")

            loss.backward()
            self.optimizer.step()
    
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size
            avg = total_loss / total_images

        return avg

    @torch.no_grad()
    def validate(self):
        """
        1. model.eval()
        2. Iterate over valid_loader
        3. Calculate validation loss and penguin IoU
        4. Return both values
        """
        self.model.eval()

        total_loss     = 0.0
        total_images   = 0
        true_positive  = 0
        false_positive = 0
        false_negative = 0

        for images, masks in self.valid_loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            logits = self.model(images)
            loss   = self.criterion(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss: {loss.item()}")

            predictions = logits.argmax(dim=1)

            valid_pixels = masks != 255

            predicted_penguin = (predictions == 1) & valid_pixels
            labeled_penguin = (masks == 1) & valid_pixels

            true_positive += (predicted_penguin & labeled_penguin).sum().item()
            false_positive += (predicted_penguin & ~labeled_penguin).sum().item()
            false_negative += (~predicted_penguin & labeled_penguin).sum().item()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size

        union = true_positive + false_positive + false_negative
        penguin_iou = true_positive / union if union > 0 else float("nan")

        return total_loss / total_images, penguin_iou

    def fit(self, num_epochs: int):
        """
        1. Increase self.epoch
        2. Call train_one_epoch()
        3. Call validate()
        4. Record and print results
        5. Save checkpoint if validation IoU improves
        """
        for _ in range(1, num_epochs + 1):
            self.epoch += 1
            train_loss = self.train_one_epoch()
            validate_loss, validate_iou = self.validate()

            self.history.append({
                "epoch": self.epoch,
                "train_loss": train_loss,
                "validate_loss": validate_loss,
                "validate_iou": validate_iou,
            })

            print(
                f"[logs] === epoch {self.epoch:02d} ===\n",
                f"train loss: {train_loss:.4f}\n",
                f"valid loss: {validate_loss:.4f}\n",
                f"penguin iou: {validate_iou:.4f}\n",
            )

            self.save_checkpoint(validate_loss, validate_iou)

    def save_checkpoint(self, valid_loss, valid_iou):
        """
        - current epoch
        - model.state_dict()
        - optimizer.state_dict()
        - validation loss and IoU
        """
        if math.isfinite(valid_iou) and valid_iou > self.best_iou:
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "valid_loss": valid_loss,
                    "valid_penguin_iou": valid_iou,
                    "num_classes": 2,
                    "excluded_train_file": self.checkpoint_dir,
                },
                self.checkpoint_dir / "best.pt",
            )

            self.best_iou = valid_iou
        
            print("[save] === best checkpoint saved ===\n")

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