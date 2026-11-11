"""Unified multiclass masks and loaders. LLM-assisted: Codex, 2026-11-09.

Prepared bird/seal splits: images/, masks/, manifest.json. Existing Roboflow
splits remain readable directly from _annotations.coco.json.
"""

from project_paths import PROJECT_ROOT

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import functional as TF

CLASS_NAMES = ("background", "penguin", "other_bird", "seal", "whale")
IGNORE_INDEX = 255
BIRD_NAMES = {
    "bird", "crow", "duck", "eagle", "flamingo", "hummingbird", "owl",
    "parrot", "pigeon", "sandpiper", "sparrow", "swan", "turkey", "woodpecker",
}
CATEGORY_MAP = {"penguin": 1, **{name: 2 for name in BIRD_NAMES}, "seal": 3, "whale": 4}


def fit_image_and_mask(image, mask, size=640):
    """Preserve aspect ratio; use nearest masks and ignore padded pixels."""
    if image.size != mask.size:
        raise ValueError(f"Image/mask dimensions differ: {image.size}, {mask.size}")
    width, height = image.size
    scale = min(size / width, size / height)
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(output_size, Image.Resampling.BILINEAR)
    mask = mask.resize(output_size, Image.Resampling.NEAREST)
    offset = ((size - output_size[0]) // 2, (size - output_size[1]) // 2)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    labels = Image.new("L", (size, size), IGNORE_INDEX)
    canvas.paste(image, offset)
    labels.paste(mask, offset)
    return canvas, labels


class AnimalDataset(Dataset):
    def __init__(self, root, split, image_size=640):
        self.root, self.split = Path(root), split
        self.split_dir = self.root / split
        self.image_size = image_size
        self.annotations = defaultdict(list)
        manifest = self.split_dir / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if tuple(data["class_names"]) != CLASS_NAMES:
                raise ValueError(f"Incompatible class mapping: {manifest}")
            self.records = data["samples"]
            self.prepared = True
        else:
            path = self.split_dir / "_annotations.coco.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            self.prepared = False
            self.category_map = {
                c["id"]: CATEGORY_MAP.get(c["name"].strip().lower())
                for c in data["categories"]
            }
            for ann in data["annotations"]:
                self.annotations[ann["image_id"]].append(ann)
            self.records = []
            for info in data["images"]:
                areas = Counter()
                for ann in self.annotations[info["id"]]:
                    target = self.category_map.get(ann["category_id"])
                    if target is not None:
                        areas[target] += max(ann.get("area", 1), 1)
                if areas:
                    self.records.append({
                        "image": info["file_name"], "image_id": info["id"],
                        "sample_class": areas.most_common(1)[0][0],
                        "mask_kind": "coco_polygon", "source": self.root.name,
                    })

    def __len__(self):
        return len(self.records)

    def image_path(self, index):
        return self.split_dir / self.records[index]["image"]

    def get_raw(self, index):
        record = self.records[index]
        with Image.open(self.image_path(index)) as source:
            image = source.convert("RGB")
        if self.prepared:
            with Image.open(self.split_dir / record["mask"]) as source:
                mask = source.convert("L")
        else:
            labels = np.zeros((image.height, image.width), dtype=np.uint8)
            for ann in self.annotations[record["image_id"]]:
                if not isinstance(ann.get("segmentation"), list):
                    raise ValueError("This loader expects COCO polygon segmentations")
                region = Image.new("1", image.size)
                draw = ImageDraw.Draw(region)
                for polygon in ann["segmentation"]:
                    if len(polygon) >= 6:
                        draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
                region = np.asarray(region, dtype=bool)
                target = self.category_map.get(ann["category_id"])
                if target is None:
                    labels[region] = IGNORE_INDEX
                else:
                    conflict = region & (labels != 0) & (labels != target)
                    labels[region & (labels == 0)] = target
                    labels[conflict] = IGNORE_INDEX
            mask = Image.fromarray(labels)
        if image.size != mask.size:
            raise ValueError(f"Image/mask mismatch: {self.image_path(index)}")
        return image, mask

    def __getitem__(self, index):
        image, mask = self.get_raw(index)
        image, mask = fit_image_and_mask(image, mask, self.image_size)
        values = np.array(mask, dtype=np.int64, copy=True)
        if not np.isin(values, [0, 1, 2, 3, 4, IGNORE_INDEX]).all():
            raise ValueError("Mask contains an unknown class ID")
        tensor = TF.normalize(TF.to_tensor(image),
                              mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return tensor, torch.from_numpy(values)


def build_datasets(root, image_size=640, *, include_background=False):
    """Keep source splits; exclude byte-identical leakage, preferring test/valid.

    Hashing detects exact copies, including the known penguin train/valid copy.
    It does not certify that differently cropped/compressed photos are unrelated.
    """
    root = Path(root)
    sources = ("dataset_penguins", "dataset_birds", "dataset_seals", "dataset_wildlife")
    datasets = {split: [AnimalDataset(root / "data" / s, split, image_size) for s in sources]
                for split in ("train", "valid", "test")}
    seen, excluded = {}, []
    wildlife_seen, wildlife_excluded = {}, []
    for split in ("test", "valid", "train"):
        for dataset in datasets[split]:
            retained = []
            ordered = sorted(enumerate(dataset.records), key=lambda pair: pair[1]["image"])
            for index, record in ordered:
                if dataset.root.name == "dataset_wildlife":
                    group = Path(record["image"]).name.split(".rf.", 1)[0]
                    if group in wildlife_seen:
                        wildlife_excluded.append({"split":split,"image":record["image"],
                            "source_group":group,"kept":wildlife_seen[group]})
                        continue
                digest = hashlib.sha256(dataset.image_path(index).read_bytes()).hexdigest()
                if digest in seen:
                    excluded.append({"split": split, "source": dataset.root.name,
                                     "image": record["image"], "kept": seen[digest]})
                else:
                    seen[digest] = f"{dataset.root.name}/{split}/{record['image']}"
                    if dataset.root.name == "dataset_wildlife":
                        wildlife_seen[group] = f"{split}/{record['image']}"
                    retained.append(record)
            dataset.records = retained
    background_excluded = []
    if include_background:
        # Preserve every existing animal split. New background records lose any
        # exact-byte overlap, including original-image hashes when available.
        for split in ("test", "valid", "train"):
            background = AnimalDataset(root / "data" / "dataset_nature", split, image_size)
            retained = []
            for index, record in enumerate(background.records):
                if record["sample_class"] != 0:
                    raise ValueError("Nature records must belong to sampling group 0")
                digest = hashlib.sha256(background.image_path(index).read_bytes()).hexdigest()
                if digest != record["prepared_image_sha256"]:
                    raise ValueError(f"Changed background image: {background.image_path(index)}")
                digests = {digest, record["original_image_sha256"]}
                collision = next((key for key in sorted(digests) if key in seen), None)
                if collision is not None:
                    background_excluded.append(dict(split=split, source="dataset_nature",
                        image=record["image"], kept=seen[collision]))
                else:
                    with Image.open(background.split_dir / record["mask"]) as mask:
                        values = np.asarray(mask)
                        if mask.size != (image_size, image_size) or not np.isin(values, [0, IGNORE_INDEX]).all() or not (values == 0).any():
                            raise ValueError(f"Invalid background mask: {record['mask']}")
                    mask_hash = hashlib.sha256((background.split_dir / record["mask"]).read_bytes()).hexdigest()
                    if mask_hash != record["prepared_mask_sha256"]:
                        raise ValueError(f"Changed background mask: {record['mask']}")
                    for key in digests:
                        seen[key] = f"dataset_nature/{split}/{record['image']}"
                    retained.append(record)
            background.records = retained
            datasets[split].append(background)
        sources = (*sources, "dataset_nature")
    summary = {
        "class_names": list(CLASS_NAMES), "image_size": image_size,
        "sources": {split: {d.root.name: len(d) for d in values}
                    for split, values in datasets.items()},
        "class_samples": {split: dict(Counter(
            CLASS_NAMES[r['sample_class']] for d in values for r in d.records
        )) for split, values in datasets.items()},
        "excluded_exact_duplicates": excluded,
        "wildlife_source_group_policy": "One deterministic record per Roboflow source prefix; test then valid then train. Raw exports are preserved.",
        "wildlife_excluded_source_variants": wildlife_excluded,
        "annotation_manifest_sha256": {
            f"{source}/{split}": hashlib.sha256((root / "data" / source / split / (
                "manifest.json" if (root / "data" / source / split / "manifest.json").exists()
                else "_annotations.coco.json"
            )).read_bytes()).hexdigest()
            for source in sources for split in ("train", "valid", "test")
        },
        "seal_mask_note": "Recovered from black-background JPEGs; approximate, not original binary masks.",
    }
    if include_background:
        summary.update(background_excluded_exact_duplicates=background_excluded,
            background_overlap_policy="Existing animal splits unchanged; exclude overlapping nature records. Prepared-byte and recorded original-byte hashes only, not a near-duplicate guarantee.",
            background_mask_policy="Explicit natural stuff=0; other and padded pixels=255 ignored")
    return {split: ConcatDataset(values) for split, values in datasets.items()}, summary


@torch.inference_mode()
def show_animal_prediction(model, dataset, index, device):
    """Visualize a sample from AnimalDataset or the combined validation dataset."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    child, local_index = dataset, index
    while isinstance(child, ConcatDataset):
        for candidate in child.datasets:
            if local_index < len(candidate):
                child = candidate
                break
            local_index -= len(candidate)
        else:
            raise IndexError(index)
    image, mask = child.get_raw(local_index)
    image, mask = fit_image_and_mask(image, mask, child.image_size)
    tensor, _ = dataset[index]
    model.eval()
    prediction = model(tensor.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
    colors = ["black", "red", "orange", "deepskyblue", "limegreen"]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    for ax, values, title in zip(axes[1:], (np.asarray(mask), prediction),
                                 ("Training annotation", "Prediction")):
        visible = (values > 0) & (values != IGNORE_INDEX)
        ax.imshow(np.where(visible, values, 0), cmap=cmap, vmin=0, vmax=4,
                  alpha=visible.astype(float) * .5)
        ax.set_title(title)
    axes[0].set_title(child.root.name)
    fig.legend(handles=[Patch(color=colors[i], label=CLASS_NAMES[i]) for i in range(1, 5)],
               loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, .06, 1, 1))
    plt.show()
    plt.close(fig)
