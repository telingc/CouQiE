"""Download a reproducible COCO-Stuff nature subset; never start training.

Raw PNG values are labels.txt IDs minus one; 255 is unlabeled. Only a fixed
allowlist of annotated natural stuff becomes background. Everything else is
ignored, even on an image without an annotated animal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import threading
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

ROOT = Path(__file__).resolve().parents[1] / "data/dataset_nature"
RAW = ROOT / "raw"
ZIP_SHA256 = "da7f213d6a581bd6912a957573d7b4e5061de0d41f733bb27249b0fa981239d7"
ZIP_NAME = "stuffthingmaps_trainval2017.zip"
ZIP_URL = "https://huggingface.co/datasets/JPShi/COCO-Stuff/resolve/main/" + ZIP_NAME
OFFICIAL = "https://github.com/nightrome/cocostuff"
CLASS_NAMES = ["background", "penguin", "other_bird", "seal", "whale"]
# Values here are the actual zero-based PNG values, not the one-based IDs.
GROUPS = {
    "sky_clouds": [105, 119, 156],
    "snow": [158],
    "rock_ground": [110, 124, 125, 126, 134, 135, 149, 153, 161],
    "sea_water": [147, 154, 177],
}
ALLOWED = np.array(sorted({v for ids in GROUPS.values() for v in ids}))
ANIMALS = list(range(15, 25))  # bird through giraffe in the original PNG map
HEADERS = {"User-Agent": "semseg-nature-dataset/1.0 (research download)", "Accept-Encoding": "identity"}
SESSIONS = threading.local()


def session():
    if not hasattr(SESSIONS, "value"):
        SESSIONS.value = requests.Session()
    return SESSIONS.value


def log(message):
    print(message, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def download(url, path, expected_sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_sha is None or sha256(path) == expected_sha):
        return path
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if expected_sha and partial.exists() else 0
        headers = {**HEADERS, **({"Range": f"bytes={offset}-"} if offset else {})}
        try:
            with session().get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected resume offset")
                current = offset if append else 0
                total = int(response.headers.get("Content-Length", 0)) + current
                started, last = time.monotonic(), time.monotonic()
                initial = current
                with partial.open("ab" if append else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        current += len(block)
                        if time.monotonic() - last >= 10:
                            speed = (current - initial) / max(time.monotonic() - started, .01)
                            log(f"[download] {path.name}: {current / 1e6:.1f}/{total / 1e6:.1f} MB; {speed / 1e6:.2f} MB/s")
                            last = time.monotonic()
                if total and response.headers.get("Content-Encoding", "identity") == "identity" and partial.stat().st_size != total:
                    raise ValueError("Incomplete download")
            if expected_sha and sha256(partial) != expected_sha:
                # Retain the failed file for diagnosis instead of resuming it.
                partial.replace(partial.with_suffix(".bad"))
                raise ValueError("SHA-256 mismatch")
            partial.replace(path)
            return path
        except (requests.RequestException, OSError, ValueError) as error:
            log(f"[retry {attempt + 1}/4] {path.name}: {error}")
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


def acquire_annotations():
    RAW.mkdir(parents=True, exist_ok=True)
    archive = download(ZIP_URL, RAW / ZIP_NAME, ZIP_SHA256)
    for name in ("labels.txt", "README.md"):
        download(f"https://raw.githubusercontent.com/nightrome/cocostuff/master/{name}", RAW / ("official_" + name))
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith(".png")]
        log(f"[annotations] verified SHA-256; {len(names)} label maps; examples: {names[:3]}")
    return archive


def index_annotations(archive):
    cache = ROOT / "annotation_index.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    candidates, reasons = [], Counter()
    with zipfile.ZipFile(archive) as z:
        entries = [i for i in z.infolist() if i.filename.endswith(".png")]
        for index, entry in enumerate(entries, 1):
            if index % 5000 == 0:
                log(f"[index] {index}/{len(entries)}; candidates={len(candidates)}")
            if not Path(entry.filename).stem.isdigit():
                raise ValueError(f"Unexpected source image ID: {entry.filename}")
            # ZipFile.read checks the CRC of each decoded annotation.
            with Image.open(BytesIO(z.read(entry))) as image:
                w, h = image.size
                if min(w, h) < 320 or max(w, h) < 600:
                    reasons["too_small"] += 1
                    continue
                array = np.array(image)
            if array.ndim != 2:
                raise ValueError(f"Expected indexed labels, got {entry.filename}: {array.shape}")
            histogram = np.bincount(array.ravel(), minlength=256)
            area = array.size
            if histogram[ANIMALS].sum():
                reasons["annotated_animal_present"] += 1
                continue
            if histogram[:91].sum() / area > .10:
                reasons["too_many_things"] += 1
                continue
            fractions = {key: float(histogram[ids].sum() / area) for key, ids in GROUPS.items()}
            nature_fraction = float(histogram[ALLOWED].sum() / area)
            if nature_fraction < .65:
                reasons["too_little_natural_background"] += 1
                continue
            group = max(fractions, key=fractions.get)
            if fractions[group] < .20:
                reasons["no_dominant_group"] += 1
                continue
            source_split = "val2017" if "val2017/" in entry.filename else "train2017"
            candidates.append({
                "image_id": Path(entry.filename).stem, "annotation": entry.filename,
                "source_split": source_split, "group": group,
                "width": w, "height": h, "nature_fraction": nature_fraction,
                "group_fractions": fractions,
            })
        output = {"archive_sha256": ZIP_SHA256, "scanned": len(entries),
                  "rejected": dict(reasons), "candidates": candidates}
    write_json(cache, output)
    log(f"[index complete] candidates={len(candidates)}; groups={dict(Counter(c['group'] for c in candidates))}")
    return output


def choose_candidates(index, target):
    # Preserve official COCO train/val separation. Subdivide val into valid/test.
    # Stable hashing happens on source image IDs BEFORE any resizing/cropping.
    buckets = defaultdict(list)
    exclusion_file = ROOT / "manual_exclusions.json"
    exclusions = json.loads(exclusion_file.read_text(encoding="utf-8")) if exclusion_file.exists() else {}
    for candidate in index["candidates"]:
        c = dict(candidate)
        if c["image_id"] in exclusions:
            continue
        digest = hashlib.sha256(("semseg-nature-v1:" + c["image_id"]).encode()).hexdigest()
        c["split"] = "train" if c["source_split"] == "train2017" else ("valid" if int(digest[:8], 16) % 2 == 0 else "test")
        c["selection_key"] = digest
        buckets[(c["split"], c["group"])].append(c)
    for values in buckets.values():
        values.sort(key=lambda c: c["selection_key"])
    chosen = []
    per_group, remainder = divmod(target, len(GROUPS))
    for number, group in enumerate(GROUPS):
        count = per_group + (number < remainder)
        train = round(count * .8)
        valid = (count - train) // 2 + (number % 2 if (count - train) % 2 else 0)
        quotas = {"train": train, "valid": valid, "test": count - train - valid}
        # Keep all available holdout samples up to their quotas. Fill holdout
        # shortfalls only with additional train2017 images, never by moving an
        # original validation image into training (or a train image into val).
        holdout_count = sum(min(quotas[s], len(buckets[(s, group)])) for s in ("valid", "test"))
        quotas["train"] = count - holdout_count
        for split, quota in quotas.items():
            available = buckets[(split, group)]
            if len(available) < quota:
                log(f"[shortfall] {split}/{group}: requested {quota}, available {len(available)}; using available only")
            chosen.extend(available[:quota])
    return chosen


def prepare_one(candidate):
    c = dict(candidate)
    image_id = c["image_id"]
    original = RAW / "images" / c["source_split"] / f"{image_id}.jpg"
    # The path-style S3 endpoint has valid HTTPS certificates; the original
    # images.cocodataset.org hostname currently fails hostname verification.
    url = f"https://s3.amazonaws.com/images.cocodataset.org/{c['source_split']}/{image_id}.jpg"
    download(url, original)
    with Image.open(original) as image:
        image.load()
        image = image.convert("RGB")
    if image.size != (c["width"], c["height"]):
        raise ValueError(f"Image/annotation dimension mismatch: {image_id}")
    original_mask = RAW / "masks" / c["source_split"] / f"{image_id}.png"
    annotation_bytes = original_mask.read_bytes()
    with Image.open(BytesIO(annotation_bytes)) as m:
        source_labels = np.array(m)
    mask = Image.fromarray(np.where(np.isin(source_labels, ALLOWED), 0, 255).astype(np.uint8))
    # Same aspect-preserving 640 fit and ignore padding as animal_data.py.
    scale = min(640 / image.width, 640 / image.height)
    shape = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    offset = ((640 - shape[0]) // 2, (640 - shape[1]) // 2)
    output_image = Image.new("RGB", (640, 640), (124, 116, 104))
    output_mask = Image.new("L", (640, 640), 255)
    output_image.paste(image.resize(shape, Image.Resampling.BILINEAR), offset)
    output_mask.paste(mask.resize(shape, Image.Resampling.NEAREST), offset)
    split_dir = ROOT / c["split"]
    for directory in ("images", "masks"):
        (split_dir / directory).mkdir(parents=True, exist_ok=True)
    output_image.save(split_dir / "images" / f"{image_id}.jpg", quality=95)
    output_mask.save(split_dir / "masks" / f"{image_id}.png")
    return {**c, "image": f"images/{image_id}.jpg", "mask": f"masks/{image_id}.png",
            "sample_class": 0, "source": "COCO-Stuff-2017/nature-subset",
            "mask_kind": "annotated_natural_stuff_only_background_else_ignore",
            "source_url": url, "original_image_sha256": sha256(original),
            "original_mask_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "prepared_image_sha256": sha256(split_dir / "images" / f"{image_id}.jpg"),
            "prepared_mask_sha256": sha256(split_dir / "masks" / f"{image_id}.png")}


def make_previews(records):
    preview = ROOT / "previews"
    preview.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=16)
    ordered = sorted(records, key=lambda r: (r["group"], r["split"], r["image_id"]))
    for page, start in enumerate(range(0, len(ordered), 60), 1):
        items = ordered[start:start + 60]
        sheet = Image.new("RGB", (6 * 250, 10 * 195), "white")
        draw = ImageDraw.Draw(sheet)
        for position, record in enumerate(items):
            x, y = (position % 6) * 250, (position // 6) * 195
            with Image.open(RAW / "images" / record["source_split"] / f"{record['image_id']}.jpg") as image:
                image = image.convert("RGB")
                image.thumbnail((246, 165))
                sheet.paste(image, (x, y))
            draw.text((x + 3, y + 166), f"{record['image_id']} {record['split']}", fill="black", font=font)
            draw.text((x + 3, y + 181), record["group"], fill="black", font=font)
        sheet.save(preview / f"contact_sheet_{page:02}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    archive = acquire_annotations()
    if args.annotations_only:
        return
    index = index_annotations(archive)
    exclusion_file = ROOT / "manual_exclusions.json"
    exclusions = json.loads(exclusion_file.read_text(encoding="utf-8")) if exclusion_file.exists() else {}
    for image_id in exclusions:
        if not image_id.isdigit():
            raise ValueError("Invalid manually excluded image ID")
        for split in ("train", "valid", "test"):
            for directory, suffix in (("images", ".jpg"), ("masks", ".png")):
                path = ROOT / split / directory / (image_id + suffix)
                if not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Excluded prepared path outside dataset root")
                # Raw originals remain available for review; remove prepared
                # copies generated by this script for excluded source images.
                path.unlink(missing_ok=True)
    chosen = choose_candidates(index, args.count)
    write_json(ROOT / "selection.json", {"requested": args.count, "selected": chosen})
    # Read the large ZIP directory once, rather than once per worker/image.
    with zipfile.ZipFile(archive) as z:
        for c in chosen:
            original_mask = RAW / "masks" / c["source_split"] / f"{c['image_id']}.png"
            original_mask.parent.mkdir(parents=True, exist_ok=True)
            original_mask.write_bytes(z.read(c["annotation"]))
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_one, c): c for c in chosen}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as error:
                c = futures[future]
                errors.append({"image_id": c["image_id"], "error": str(error)})
                log(f"[image failed] {c['image_id']}: {error}")
            if number % 25 == 0 or number == len(chosen):
                log(f"[prepare] {number}/{len(chosen)}; successful={len(records)}; failed={len(errors)}")
    # Exact-byte duplicates are retained in raw only, never across prepared splits.
    seen, duplicates = set(), []
    unique = []
    for record in sorted(records, key=lambda r: ({"test": 0, "valid": 1, "train": 2}[r["split"]], r["image_id"])):
        digest = record["original_image_sha256"]
        if digest in seen:
            duplicates.append(record["image_id"])
            # Remove only the duplicate copies generated by this preparation.
            for key in ("image", "mask"):
                (ROOT / record["split"] / record[key]).unlink()
            continue
        seen.add(digest)
        unique.append(record)
    for split in ("train", "valid", "test"):
        write_json(ROOT / split / "manifest.json", {
            "class_names": CLASS_NAMES, "image_size": 640, "ignore_index": 255,
            "source": OFFICIAL, "samples": sorted([r for r in unique if r["split"] == split], key=lambda r: r["image_id"]),
        })
    make_previews(unique)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_images": args.count, "prepared_images": len(unique),
        "splits": dict(Counter(r["split"] for r in unique)),
        "groups": dict(Counter(r["group"] for r in unique)),
        "by_split_and_group": {s: dict(Counter(r["group"] for r in unique if r["split"] == s)) for s in ("train", "valid", "test")},
        "errors": errors, "excluded_exact_duplicates": duplicates,
        "manual_exclusions": exclusions,
        "archive_url": ZIP_URL, "archive_sha256": ZIP_SHA256,
        "official_source": OFFICIAL, "png_label_policy": "labels.txt ID minus one; original void=255",
        "background_png_values": GROUPS,
        "annotation_index_scanned": index["scanned"],
        "limitations": ["Subset, not the full COCO-Stuff image collection.",
                       "Automatic filtering uses annotations, not a guarantee of animal absence.",
                       "Only allowlisted annotated natural stuff is background; other pixels and padding are ignored.",
                       "COCO source photos are commonly around 640 pixels on their longer side, not native high-resolution Antarctic imagery.",
                       "Not connected to training loaders; no training was run.",
                       "Internal original-image split isolation does not establish independence from other existing project datasets.",
                       "Annotations CC-BY-4.0; image rights remain with their original owners."],
    }
    write_json(ROOT / "preparation_report.json", report)
    log(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("Some images failed. Rerun to resume from existing raw downloads.")


if __name__ == "__main__":
    main()
