"""Evaluate saved experiments against a fixed, user-authorized development set.

No training. Landscape filenames are negative labels by explicit user instruction.
References include model-assisted masks and are not an independent blind test.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path, WindowsPath
import csv
import gc
import hashlib
import json
import shutil
import time

import numpy as np
from PIL import Image, ImageOps
import psutil
import torch

from project_paths import PROJECT_ROOT as ROOT
from script import ResNetUNet
from experiment1 import predict_with_tiles, predict_with_resize
from experiment2 import fuse_probabilities
from experiment3 import fuse_biased_predictions
from experiment9 import infer_branch, foreground_union, verify_compact_graph

CLASS_NAMES = ("background", "penguin", "other_bird", "seal", "whale")
ANIMAL_PHOTOS = ("penguins_far.jpg", "penguins_near.jpg", "penguin_colony.jpg")
BENCHMARK_VERSION = "benchmark_v2"
PREDICTION_CACHE_VERSION = "benchmark_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def class_mask(path):
    with Image.open(path) as raw:
        values = np.asarray(raw)
        if values.ndim == 3:
            if not (np.array_equal(values[:, :, 0], values[:, :, 1]) and np.array_equal(values[:, :, 0], values[:, :, 2])):
                raise ValueError(f"Expected grayscale class IDs: {path}")
            if values.shape[2] == 4 and not (values[:, :, 3] == 255).all():
                raise ValueError(f"Unexpected transparent mask: {path}")
            values = values[:, :, 0]
        values = np.array(values, dtype=np.uint8, copy=True)
    if not np.isin(values, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError(f"Unknown class IDs: {path}")
    return values


def prepare_reference_set():
    folder = ROOT / "data/reference_masks" / BENCHMARK_VERSION
    path = folder / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            if sha(ROOT / entry["image_path"]) != entry["image_sha256"] or sha(folder / entry["mask"]) != entry["mask_sha256"]:
                raise ValueError("The frozen reference set changed; create a new benchmark version")
        return folder, manifest
    if BENCHMARK_VERSION == "benchmark_v2":
        return revise_near_reference(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed_folder = ROOT / "data/reference_masks/review_v1"
    entries = []
    names = list(ANIMAL_PHOTOS) + sorted(p.name for p in (ROOT / "data/examples").iterdir()
        if "landscape" in p.stem.lower() and p.suffix.lower() in {".jpg", ".jpeg", ".webp"})
    for name in names:
        image_path = ROOT / "data/examples" / name
        with Image.open(image_path) as raw:
            width, height = ImageOps.exif_transpose(raw).size
        stem = image_path.stem
        if "landscape" in stem.lower():
            values = np.zeros((height, width), dtype=np.uint8)
            origin = "user_declared_full_background"
            source = None
        else:
            candidates = [p for parent in (ROOT / "data/examples", seed_folder) for p in parent.glob(stem + ".edit_*.png")]
            if candidates:
                candidates.sort(key=lambda p: (int(p.stem.rsplit("_", 1)[1]), p.stat().st_mtime_ns), reverse=True)
                source = candidates[0]
                origin = "user_edited_reference"
            else:
                source = seed_folder / (stem + ".seed.png")
                origin = "user_authorized_model_assisted_reference"
            values = class_mask(source)
        if values.shape != (height, width):
            raise ValueError(f"Reference/image shape mismatch: {name}")
        target = folder / (stem + ".png")
        Image.fromarray(values).save(target)
        ids, counts = np.unique(values, return_counts=True)
        entries.append(dict(image=name, image_path=image_path.relative_to(ROOT).as_posix(),
            image_sha256=sha(image_path), width=width, height=height, mask=target.name,
            mask_sha256=sha(target), annotation_origin=origin,
            source_annotation=source.relative_to(ROOT).as_posix() if source else None,
            source_annotation_sha256=sha(source) if source else None,
            pixel_counts={str(k): int(v) for k, v in zip(ids, counts)},
            group="landscapes" if "landscape" in stem.lower() else "penguin_photos"))
    manifest = dict(version=BENCHMARK_VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        authorization="User requested these reference images for metrics; all landscape filenames are full background.",
        status="frozen_development_reference", class_names=list(CLASS_NAMES), ignore_index=255,
        limitations="Only background and penguin positive labels. Some references originate from experiment-11 predictions; this is not a blind independent test. No positive bird/seal/whale recall can be measured here.",
        people_image="Excluded: no explicit full-background instruction or edited mask for the people image.",
        entries=entries)
    write_json(path, manifest)
    return folder, manifest


def revise_near_reference(folder):
    """Freeze v2 from v1, changing only the user-selected near-photo reference."""
    previous = ROOT / "data/reference_masks/benchmark_v1"
    manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    result_path = ROOT / "artifacts/reference_benchmark" / PREDICTION_CACHE_VERSION / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = next(m for m in result["models"] if m["spec"]["id"] == "exp11_best")
    record = next(r for r in model["records"] if r["image"] == "penguins_near.jpg")
    artifact = record["masks"]["resize"]
    source = ROOT / artifact["path"]
    if sha(source) != artifact["sha256"]:
        raise ValueError("The selected resize prediction changed")
    folder.mkdir(parents=True, exist_ok=True)
    for entry in manifest["entries"]:
        if sha(previous / entry["mask"]) != entry["mask_sha256"] or sha(ROOT / entry["image_path"]) != entry["image_sha256"]:
            raise ValueError("The previous frozen reference set changed")
        target = folder / entry["mask"]
        if entry["image"] != "penguins_near.jpg":
            shutil.copy2(previous / entry["mask"], target)
            continue
        if entry["image_sha256"] != record["image_sha256"]:
            raise ValueError("Resize prediction/source image mismatch")
        values = class_mask(source)
        if values.shape != (entry["height"], entry["width"]):
            raise ValueError("Resize reference dimensions differ")
        Image.fromarray(values).save(target)
        labels, counts = np.unique(values, return_counts=True)
        entry.update(mask_sha256=sha(target), annotation_origin="user_selected_experiment11_resize_reference",
            source_annotation=source.relative_to(ROOT).as_posix(), source_annotation_sha256=sha(source),
            reference_checkpoint_sha256=model["model"]["checkpoint_sha256"],
            pixel_counts={str(k):int(v) for k,v in zip(labels,counts)})
    manifest.update(version=BENCHMARK_VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        previous_version="benchmark_v1", previous_manifest_sha256=sha(previous / "manifest.json"),
        revision_reason="User selected the experiment-11 best checkpoint's resize prediction as the penguins_near reference. All eight other masks remain byte-identical.",
        near_reference_limitations="Model-assisted foreground-focused mask; back-row penguins omitted by resize remain background. Matching this source prediction gives 100% IoU by construction.")
    write_json(folder / "manifest.json", manifest)
    return folder, manifest


def model_specs():
    def selected(number, filename="latest.json"):
        return json.loads((ROOT / f"artifacts/experiments/exp{number}_results" / filename).read_text())["selected"]["checkpoint"]
    seven = json.loads((ROOT / "artifacts/experiments/exp7_results/latest.json").read_text())
    twelve = json.loads((ROOT / "artifacts/experiments/exp12_results/latest.json").read_text())
    result = [dict(id="binary", experiment=1, checkpoint=str(ROOT / "artifacts/runs/penguin_baseline/best.pt")),
        dict(id="exp5_initial", experiment=5, checkpoint=str(ROOT / "artifacts/runs/antarctic_multiclass/best_antarctic_multiclass.pt")),
        dict(id="exp5_tuned", experiment=5, checkpoint=str(ROOT / "artifacts/runs/antarctic_batch_test_20261109_192421/best_antarctic_multiclass_tuned.pt"))]
    result += [dict(id="exp7_" + row["config"]["bn_policy"], experiment=7, checkpoint=row["checkpoint"]) for row in seven["results"]]
    result += [dict(id="exp8_kernel" + str(k), experiment=8, checkpoint=selected(8, f"latest_kernel{k}.json")) for k in (5, 7)]
    result += [dict(id=f"exp{n}_best", experiment=n, checkpoint=selected(n)) for n in (10, 11, 12)]
    result += [dict(id="exp12_final", experiment=12, checkpoint=twelve["final_checkpoint"])]
    return result


def confusion(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and reference dimensions differ")
    keep = target != 255
    if not np.isin(prediction[keep], range(5)).all():
        raise ValueError("Prediction contains unknown IDs")
    return np.bincount((target[keep].astype(np.int64) * 5 + prediction[keep]).ravel(), minlength=25).reshape(5, 5)


def metric_row(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    ratio = lambda a, b: float(a / b) if b else None
    total = int(matrix.sum())
    tp = int(matrix[1, 1]); fp = int(matrix[:, 1].sum() - tp); fn = int(matrix[1].sum() - tp)
    animal_tp = int(matrix[1:, 1:].sum()); animal_fp = int(matrix[0, 1:].sum()); animal_fn = int(matrix[1:, 0].sum())
    bg = int(matrix[0].sum()); fg = int(matrix[1:].sum())
    return dict(pixels=total, penguin_tp=tp, penguin_fp=fp, penguin_fn=fn,
        penguin_iou=ratio(tp, tp + fp + fn), penguin_dice=ratio(2 * tp, 2 * tp + fp + fn),
        penguin_precision=ratio(tp, tp + fp), penguin_recall=ratio(tp, tp + fn),
        penguin_false_discovery=ratio(fp, tp + fp), pixel_accuracy=ratio(np.trace(matrix), total),
        animal_iou=ratio(animal_tp, animal_tp + animal_fp + animal_fn),
        background_fp=ratio(animal_fp, bg), background_specificity=ratio(int(matrix[0, 0]), bg),
        animal_class_confusion=ratio(int(matrix[1:, 1:].sum() - np.trace(matrix[1:, 1:])), fg),
        animal_miss=ratio(animal_fn, fg),
        **{name + "_on_background": ratio(int(matrix[0, c]), bg) for c, name in enumerate(CLASS_NAMES) if c},
        **{name + "_area": ratio(int(matrix[:, c].sum()), total) for c, name in enumerate(CLASS_NAMES)})


def get_model(spec):
    with torch.serialization.safe_globals([WindowsPath]):
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model_state"]
    kernel = state["enc1.0.conv1.weight"].shape[-1]
    classes = int(checkpoint["num_classes"])
    model = ResNetUNet(num_classes=classes, pretrained=False, kernel_size=kernel)
    model.load_state_dict(state, strict=True)
    metadata = dict(epoch=int(checkpoint["epoch"]), num_classes=classes, kernel_size=kernel,
        historical_validation=checkpoint.get("valid_metrics"), binary_validation_iou=checkpoint.get("valid_penguin_iou"),
        config=checkpoint.get("config", {}))
    del state, checkpoint
    return model.eval(), metadata


def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def native_scores(image, model):
    try:
        return infer_branch(image, model, "cuda:0")
    except torch.OutOfMemoryError:
        pass
    model.cpu()
    clean()
    estimated = image.width * image.height * 350 + 2 * 1024**3
    if estimated > psutil.virtual_memory().available * .95:
        raise MemoryError("Insufficient RAM for original-resolution inference; no hidden resize was applied")
    values, info = infer_branch(image, model, "cpu")
    info["fallback_reason"] = "CUDA out of memory at original resolution"
    return values, info


def cache_record(folder, entry, methods):
    path = folder / entry["image"].rsplit(".", 1)[0] / "prediction.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["image_sha256"] != entry["image_sha256"]:
        raise ValueError("Source image changed")
    for method in methods:
        output = ROOT / record["masks"][method]["path"]
        if not output.exists() or sha(output) != record["masks"][method]["sha256"]:
            return None
    return record


def save_prediction(folder, entry, masks, details):
    output = folder / Path(entry["image"]).stem
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for method, values in masks.items():
        path = output / (method + ".png")
        Image.fromarray(values.astype(np.uint8)).save(path)
        paths[method] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    record = dict(image=entry["image"], image_sha256=entry["image_sha256"], masks=paths, details=details)
    write_json(output / "prediction.json", record)
    return record


@torch.inference_mode()
def run_model(spec, references, output):
    folder = output / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha(spec["checkpoint"])
    metadata_path = folder / "model.json"
    if metadata_path.exists():
        stored = json.loads(metadata_path.read_text())
        if stored["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("A surviving checkpoint changed; use a new benchmark output")
    methods = ["native", "resize", "union"]
    if spec["id"] == "binary":
        methods += ["tiles", "stretch_resize", "fusion_060", "tile_biased_020", "resize_biased_080", "dual_union"]
    cached = [cache_record(folder, entry, methods) for entry in references["entries"]]
    if all(cached):
        print("Verified cached model:", spec["id"], flush=True)
        return dict(spec=spec, model=json.loads(metadata_path.read_text()), records=cached)
    model, metadata = get_model(spec)
    metadata.update(checkpoint_sha256=checkpoint_sha, checkpoint=spec["checkpoint"],
        graph_parity=verify_compact_graph(model), prediction_code_sha256=sha(Path(__file__)),
        inference_code_sha256=sha(ROOT / "scripts/experiment9.py"), training_performed=False)
    write_json(metadata_path, metadata)
    records = []
    fresh11 = json.loads((ROOT / "artifacts/evaluation/latest.json").read_text())
    known11 = {r["image"]: r for r in fresh11["records"]}
    for index, (entry, existing) in enumerate(zip(references["entries"], cached), 1):
        print(f"{spec['id']} | {index}/{len(cached)} | {entry['image']}", flush=True)
        if existing:
            records.append(existing)
            continue
        old = known11.get(entry["image"])
        if spec["id"] == "exp11_best" and fresh11["config"]["checkpoint_sha256"] == checkpoint_sha and old and old["source_sha256"] == entry["image_sha256"]:
            masks = {method: class_mask(Path(old["output"]) / (method + ".png")) for method in ("native", "resize", "union")}
            records.append(save_prediction(folder, entry, masks, dict(source="Verified saved experiment-11 inference", source_run=fresh11["config"]["run"], native=old["native"], resize=old["resize"])))
            del masks
            continue
        with Image.open(ROOT / entry["image_path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        native, native_info = native_scores(image, model)
        clean()
        resized, resize_info = infer_branch(image, model, "cuda:0", resize=True)
        a, b, union, union_info = foreground_union(native, resized)
        masks = dict(native=a, resize=b, union=union)
        del native, resized
        clean()
        if spec["id"] == "binary":
            print("  Replaying 640/480 tiles and historical probability fusion", flush=True)
            model.cuda()
            tile, _, tile_probability = predict_with_tiles(image, model, "cuda:0", tile_size=640, stride=480, return_prob=True)
            _, stretch, resize_probability = predict_with_resize(image, model, "cuda:0", tile_size=640, return_prob=True)
            _, fused = fuse_probabilities(resize_probability, tile_probability, alpha=.6, threshold=.5)
            _, _, tile_biased, resize_biased, dual = fuse_biased_predictions(resize_probability, tile_probability,
                tile_bias_alpha=.2, global_bias_alpha=.8, tile_threshold=.5, global_threshold=.5)
            masks.update(tiles=tile, stretch_resize=stretch, fusion_060=fused, tile_biased_020=tile_biased,
                resize_biased_080=resize_biased, dual_union=dual)
            del tile_probability, resize_probability, tile_biased, resize_biased, dual, fused, tile, stretch
        records.append(save_prediction(folder, entry, masks, dict(source="Fresh inference", native=native_info,
            resize=resize_info, union=union_info, protocol="Letterbox 640 for resize. Binary historical replay additionally uses stretch 640, tile 640/stride 480, alpha .6 and .2/.8.")))
        del masks, image, a, b, union
        clean()
    model.cpu()
    del model
    clean()
    return dict(spec=spec, model=metadata, records=records)


def summarize(result, references, ref_folder):
    rows, per_class, matrices = [], [], {}
    for model in result["models"]:
        variant = model["spec"]["id"]
        entries = {e["image"]: e for e in references["entries"]}
        for record in model["records"]:
            entry = entries[record["image"]]
            target = class_mask(ref_folder / entry["mask"])
            for method, artifact in record["masks"].items():
                predicted = class_mask(ROOT / artifact["path"])
                matrix = confusion(predicted, target)
                rows.append(dict(variant=variant, method=method, scope="image", image=entry["image"],
                    image_group=entry["group"], num_images=1, **metric_row(matrix)))
                for group in ("all", entry["group"]):
                    key = (variant, method, group)
                    matrices.setdefault(key, []).append(matrix)
                del predicted
            del target
    for (variant, method, group), values in matrices.items():
        matrix = np.sum(values, axis=0)
        photo_ious = [metric_row(value)["penguin_iou"] for value in values if value[1].sum() > 0]
        rows.append(dict(variant=variant, method=method, scope="pooled", image="", image_group=group,
            num_images=len(values), mean_penguin_photo_iou=float(np.mean(photo_ious)) if photo_ious else None,
            **metric_row(matrix)))
        for c, name in enumerate(CLASS_NAMES):
            tp = int(matrix[c, c]); fp = int(matrix[:, c].sum() - tp); fn = int(matrix[c].sum() - tp)
            ratio = lambda a, b: float(a / b) if b else None
            support = int(matrix[c].sum())
            per_class.append(dict(variant=variant, method=method, image_group=group, class_name=name,
                target_pixels=support, tp=tp, fp=fp, fn=fn, iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                note="Absent positive class: recall undefined; any prediction is a false positive" if not support else ""))
    return rows, per_class


def run():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this finite experiment replay")
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    ref_folder, references = prepare_reference_set()
    output = ROOT / "artifacts/reference_benchmark" / BENCHMARK_VERSION
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = dict(reference_manifest=str(ref_folder / "manifest.json"), reference_manifest_sha256=sha(ref_folder / "manifest.json"),
        status="running", training_performed=False, models=[])
    for spec in model_specs():
        prediction_cache = ROOT / "artifacts/reference_benchmark" / PREDICTION_CACHE_VERSION
        result["models"].append(run_model(spec, references, prediction_cache))
        write_json(output / "progress.json", result)
    rows, per_class = summarize(result, references, ref_folder)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "class_metrics.csv", per_class)
    result.update(status="complete", seconds=time.perf_counter() - started,
        num_reference_images=len(references["entries"]), metrics=str(output / "metrics.csv"), class_metrics=str(output / "class_metrics.csv"))
    write_json(output / "result.json", result)
    write_json(ROOT / "artifacts/reference_benchmark/latest.json", result)
    print("COMPLETE", output, result["seconds"], flush=True)
    return result


def rescore_only():
    """Recalculate reference metrics from saved masks without loading a network."""
    started = time.perf_counter()
    folder, references = prepare_reference_set()
    source = ROOT / "artifacts/reference_benchmark" / PREDICTION_CACHE_VERSION / "result.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    for model in result["models"]:
        for record in model["records"]:
            for artifact in record["masks"].values():
                if sha(ROOT / artifact["path"]) != artifact["sha256"]:
                    raise ValueError("A cached prediction changed")
    output = ROOT / "artifacts/reference_benchmark" / BENCHMARK_VERSION
    output.mkdir(parents=True, exist_ok=True)
    result.update(status="rescoring", reference_manifest=str(folder / "manifest.json"),
        reference_manifest_sha256=sha(folder / "manifest.json"), prediction_source_result=str(source),
        prediction_source_result_sha256=sha(source), inference_performed=False, training_performed=False)
    old_points = source.parent / "point_check.json"
    if old_points.exists() and not (output / old_points.name).exists():
        shutil.copy2(old_points, output / old_points.name)
    print("Rescoring verified prediction masks against", BENCHMARK_VERSION, flush=True)
    rows, per_class = summarize(result, references, folder)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "class_metrics.csv", per_class)
    result.update(status="complete", seconds=time.perf_counter()-started,
        num_reference_images=len(references["entries"]), metrics=str(output/"metrics.csv"), class_metrics=str(output/"class_metrics.csv"))
    write_json(output / "result.json", result)
    write_json(ROOT / "artifacts/reference_benchmark/latest.json", result)
    print("RESCORE COMPLETE", result["seconds"], flush=True)
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rescore-only", action="store_true", help="Reuse saved predictions without network inference")
    arguments = parser.parse_args()
    rescore_only() if arguments.rescore_only else run()
