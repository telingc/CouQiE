"""Build a private, whole-image background-only supervision dataset.

LLM-assisted: Codex, 2026-11-12. RGB is copied unchanged. Only private training
images are used. Animals, existing ignore regions and a safety margin become
255; remaining background becomes 0. No model inference or training occurs.
"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter1d

from project_paths import PROJECT_ROOT
from animal_data import CLASS_NAMES, AnimalDataset, fit_image_and_mask

ROOT = Path(PROJECT_ROOT)
SOURCE = ROOT / "data/dataset_private_prepared"
DEFAULT_OUTPUT = ROOT / "data/dataset_antarctic_background"
PROTOCOL = "antarctic_background_whole_image_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def background_mask(labels, radius):
    """Exact square/Chebyshev safety margin around every non-background pixel."""
    if labels.ndim != 2 or not np.isin(labels,[0,1,2,3,4,255]).all():
        raise ValueError("Expected a five-class mask with ignore=255")
    excluded = labels != 0
    if radius:
        excluded = maximum_filter1d(excluded, size=2*radius+1, axis=0, mode="constant", cval=0)
        excluded = maximum_filter1d(excluded, size=2*radius+1, axis=1, mode="constant", cval=0)
    output = np.zeros(labels.shape, dtype=np.uint8)
    output[excluded] = 255
    return output


def make_preview(image, labels, path, title):
    image = image.copy(); image.thumbnail((440,310),Image.Resampling.LANCZOS)
    small = np.asarray(Image.fromarray(labels).resize(image.size,Image.Resampling.NEAREST))
    colors = np.full((*small.shape,3),[170,170,170],dtype=np.uint8)
    colors[small==0] = [65,38,90]
    overlay = np.asarray(image).copy(); valid = small==0
    overlay[valid] = (.55*overlay[valid] + .45*np.array([115,68,185])).astype(np.uint8)
    canvas = Image.new("RGB",(image.width*3,image.height+48),"white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",15)
    except OSError:
        font = ImageFont.load_default()
    for i,(shown,label) in enumerate([(image,"Original RGB"),(Image.fromarray(colors),"Background 0 | Ignore 255"),(Image.fromarray(overlay),"Background supervision overlay")]):
        canvas.paste(shown,(i*image.width,48)); draw.text((i*image.width+4,25),label,font=font,fill="black")
    draw.text((4,3),title,font=font,fill="black")
    canvas.save(path,quality=91)
    return canvas


def run(output=DEFAULT_OUTPUT, margin=16, reference_size=960):
    output = Path(output).resolve()
    data_root = (ROOT/"data").resolve()
    if not output.is_relative_to(data_root) or output == data_root or output == SOURCE.resolve():
        raise ValueError("Output must be a new dataset directory inside project data/")
    if margin < 0 or reference_size <= 0:
        raise ValueError("Invalid safety-margin configuration")
    train_manifest = SOURCE/"train/manifest.json"; valid_manifest = SOURCE/"valid/manifest.json"
    train = read_json(train_manifest)["samples"]; valid = read_json(valid_manifest)["samples"]
    train_scenes = {r["scene_group"] for r in train}; valid_scenes = {r["scene_group"] for r in valid}
    if train_scenes & valid_scenes:
        raise ValueError("Existing source scene split is not disjoint")
    validation_hashes = {r.get("original_image_sha256",r["image_sha256"]) for r in valid}
    test_images = sorted(p for p in (ROOT/"data/nju_testset").iterdir() if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"})
    test_hashes = {sha(p) for p in test_images}
    protected = [train_manifest,valid_manifest,SOURCE/"conversion.json",ROOT/"scripts/animal_data.py"]
    protected += [ROOT/"data/dataset_nature"/split/"manifest.json" for split in ["train","valid","test"]]
    protected_before = {p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    identity = dict(protocol=PROTOCOL,source_train_manifest_sha256=sha(train_manifest),
        source_valid_manifest_sha256=sha(valid_manifest),margin_reference_pixels=margin,
        margin_reference_canvas=reference_size,margin_shape="square/Chebyshev",
        native_radius="ceil(margin * max(original_width, original_height) / reference_canvas)",
        image_policy="byte-identical independent file copies; no RGB erasing/cropping/resizing",
        labels={"background":0,"animal_and_safety_margin":255,"existing_ignore_and_margin":255},
        source_quality_basis="User confirmed high-quality SAM3 labels; no extra per-pixel manual correction is claimed")
    request = output/"request.json"
    if output.exists() and any(output.iterdir()) and (not request.exists() or read_json(request)!=identity):
        raise ValueError("Existing output differs; choose another dataset directory")
    output.mkdir(parents=True,exist_ok=True); save_json(request,identity)
    for split in ["train","valid","test"]:
        (output/split/"images").mkdir(parents=True,exist_ok=True)
        (output/split/"masks").mkdir(exist_ok=True)
    (output/"records").mkdir(exist_ok=True); (output/"previews").mkdir(exist_ok=True)
    records = []; skipped = []; statistics = []; previews = []; seen = set(); representative_scenes = set()
    for index,parent in enumerate(train,1):
        image_path = SOURCE/"train"/parent["image"]; mask_path = SOURCE/"train"/parent["mask"]
        image_digest = sha(image_path); mask_digest = sha(mask_path)
        if image_digest != parent["image_sha256"] or mask_digest != parent["mask_sha256"]:
            raise ValueError("Source image or SAM3 mask changed after conversion")
        original_digest = parent.get("original_image_sha256",image_digest)
        if original_digest in validation_hashes or original_digest in test_hashes:
            skipped.append(dict(id=parent["id"],reason="source_hash_overlaps_validation_or_final_test")); continue
        if original_digest in seen:
            skipped.append(dict(id=parent["id"],reason="duplicate_source_image")); continue
        seen.add(original_digest)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            original_labels = np.asarray(source,dtype=np.uint8)
        if original_labels.shape != (image.height,image.width):
            raise ValueError("Source dimensions do not match")
        radius = math.ceil(margin * max(image.size) / reference_size)
        labels = background_mask(original_labels,radius)
        if not np.all(labels[original_labels != 0]==255):
            raise AssertionError("An animal/ignored source pixel became background")
        fitted_image,fitted_mask = fit_image_and_mask(image,Image.fromarray(labels),640)
        fitted = np.asarray(fitted_mask)
        valid_640 = int((fitted==0).sum())
        fitted_image.close(); fitted_mask.close()
        if valid_640 < 1024:
            skipped.append(dict(id=parent["id"],reason="fewer_than_1024_valid_background_pixels_at_training_640"));image.close();continue
        identifier = parent["id"]
        if Path(identifier).name != identifier or any(c in identifier for c in ["/","\\"]):
            raise ValueError("Unsafe source identifier")
        destination_image = output/"train/images"/image_path.name
        destination_mask = output/"train/masks"/(identifier+".png")
        if destination_image.exists():
            if sha(destination_image)!=image_digest:
                raise ValueError("Existing copied RGB differs from source")
        else:
            shutil.copy2(image_path,destination_image)
        Image.fromarray(labels).save(destination_mask)
        counts = {"0":int((labels==0).sum()),"255":int((labels==255).sum())}
        record = dict(id=identifier,image="images/"+destination_image.name,mask="masks/"+destination_mask.name,
            source="dataset_private_prepared/background_view",sample_class=0,sampling_group="antarctic_background",
            mask_kind="background_only_animals_and_margin_ignored",split="train",scene_group=parent["scene_group"],
            width=image.width,height=image.height,pixel_counts=counts,
            pixel_counts_640={"0":valid_640,"255":int((fitted==255).sum())},
            parent_image=image_path.relative_to(ROOT).as_posix(),parent_mask=mask_path.relative_to(ROOT).as_posix(),
            parent_image_sha256=image_digest,parent_mask_sha256=mask_digest,original_image_sha256=original_digest,
            image_sha256=sha(destination_image),mask_sha256=sha(destination_mask),
            prepared_image_sha256=image_digest,prepared_mask_sha256=sha(destination_mask),
            margin_native_pixels=radius,margin_reference_pixels=margin,margin_reference_canvas=reference_size,
            annotation_origin="Background complement of user-confirmed-high-quality SAM3 labels",
            human_ground_truth=False,original_rgb_unchanged=True)
        records.append(record);save_json(output/"records"/(identifier+".json"),record)
        statistics.append(dict(id=identifier,scene_group=parent["scene_group"],width=image.width,height=image.height,
            margin_native_pixels=radius,background_pixels=counts["0"],ignored_pixels=counts["255"],
            background_fraction=counts["0"]/labels.size,training_640_background_pixels=valid_640))
        preview_path = output/"previews"/(identifier+".jpg")
        canvas = make_preview(image,labels,preview_path,identifier+" | "+parent["scene_group"]+f" | background {counts['0']/labels.size:.1%}")
        if parent["scene_group"] not in representative_scenes:
            representative_scenes.add(parent["scene_group"])
            thumb = canvas.copy();thumb.thumbnail((760,240),Image.Resampling.LANCZOS);previews.append((identifier,thumb))
        canvas.close();image.close()
        if index % 10 == 0 or index==len(train):
            print("BACKGROUND",index,"/",len(train),"prepared:",len(records),flush=True)
            save_json(output/"progress.json",dict(processed=index,total=len(train),prepared=len(records),skipped=skipped))
    for split,items in [("train",records),("valid",[]),("test",[])]:
        save_json(output/split/"manifest.json",dict(class_names=list(CLASS_NAMES),ignore_index=255,samples=items,
            source="antarctic_background",sampling_group="antarctic_background",semantic_class=0,
            annotation_origin="Derived SAM3 background supervision",human_ground_truth=False,
            evaluation_note="Training-only derivative; use the frozen private validation images for evaluation"))
    save_csv(output/"statistics.csv",statistics)
    for start in range(0,len(previews),12):
        page = Image.new("RGB",(1540,260*((min(12,len(previews)-start)+1)//2)),"white")
        for j,(_,thumb) in enumerate(previews[start:start+12]):
            page.paste(thumb,(10+(j%2)*770,10+(j//2)*260))
        page.save(output/"previews"/f"overview_{start//12+1:02d}.jpg",quality=92)
    cards = []
    for row in records:
        cards.append('<figure><img loading="lazy" src="previews/'+html.escape(row["id"])+'.jpg"><figcaption>'+html.escape(row["id"]+' | '+row["scene_group"])+"</figcaption></figure>")
    (output/"gallery.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Antarctic background supervision</title>'
        '<style>body{font-family:Arial;margin:24px;background:#eee}figure{background:white;padding:12px;margin:16px 0}img{max-width:100%;height:auto}</style>'
        '<h1>Antarctic background-only supervision</h1><p>Whole RGB remains unchanged. Purple: background label 0. Gray: ignored animals and safety margin (255). '
        'Private training images only; no independent new photographs or evaluation set.</p>'+''.join(cards)+'</html>',encoding="utf-8")
    save_json(output/"sampling_plan.json",dict(active=False,purpose="Proposed source-aware background sampling for a future training run",
        background_total_probability="inherit the baseline; keep all animal-group probabilities unchanged",
        conditional_background_probabilities={"nature_background":.25,"antarctic_background":.75},
        per_image_weight="target_group_probability / number_of_images_in_that_group",semantic_labels_unchanged=True,
        note="Sampling groups describe data sources, not extra network output classes; training code was not modified."))
    protected_after = {name:sha(ROOT/name) for name in protected_before}
    if protected_before != protected_after:
        raise AssertionError("A protected source file changed during preparation")
    summary = dict(protocol=PROTOCOL,status="prepared",created_utc=datetime.now(timezone.utc).isoformat(),
        source_images=len(train),prepared_images=len(records),splits={"train":len(records),"valid":0,"test":0},
        scenes=len({r["scene_group"] for r in records}),labels=[0,255],class_names=list(CLASS_NAMES),skipped=skipped,
        margin_reference_pixels=margin,margin_reference_canvas=reference_size,
        valid_background_native_pixels=sum(r["background_pixels"] for r in statistics),
        valid_background_training_640_pixels=sum(r["training_640_background_pixels"] for r in statistics),
        protected_source_sha256=protected_before,source_images_and_masks_preserved=True,
        builder_sha256=sha(Path(__file__)),training_performed=False,training_configuration_changed=False,
        sample_unit="One derived supervision view per retained original photograph; not a new independent image",
        annotation_basis="User-confirmed-high-quality SAM3 annotations; masks are model-assisted, not independent human ground truth")
    save_json(output/"preparation.json",summary)
    (output/"README.md").write_text("# Antarctic background-only dataset\n\n"
        f"{len(records)} whole-image training views from {len(train)} private training photographs. No validation or final test photograph is used.\n\n"
        "| Split | Images |\n|---|---:|\n"+f"| Train | {len(records)} |\n| Valid | 0 |\n| Test | 0 |\n\n"
        "RGB files are byte-identical independent copies of the prepared private originals. Animal RGB remains visible: this is loss masking, not erasing animals. "
        "Background=0; all four animal classes, existing ignore regions, and their safety margin=255. "
        f"The square safety radius is ceil({margin} * max(width,height) / {reference_size}) native pixels, corresponding to {margin} pixels at the {reference_size}-pixel inference canvas. "
        "This is a conservative exclusion margin, not a measured receptive field.\n\n"
        "train/images/, train/masks/ and train/manifest.json follow AnimalDataset's prepared format. Five semantic output classes remain unchanged. "
        "Sampling group antarctic_background is distinct from Nature; both supervise semantic background 0. "
        "sampling_plan.json records an inactive Nature:Antarctic=1:3 proposal within the existing total background sampling mass. No training configuration was edited.\n\n"
        "These are derivatives of user-confirmed-high-quality SAM3 masks. No additional per-pixel human labeling is claimed. "
        "The collection changes background supervision/exposure, not the number of independent source photographs. "
        "Use the existing frozen 25-image private validation split to compare background FP and animal precision/recall; it is deliberately not copied into this training collection.\n\n"
        "Open gallery.html or previews/overview_01.jpg to inspect RGB, label maps and overlays. "
        "Class-ID PNGs contain 0 and 255, so background appears black and ignored regions white in a normal viewer; gray/purple previews explain the labels. "
        "statistics.csv and records/ retain original dimensions, coordinates policy, parent hashes and valid-pixel counts. "
        "Everything is private course material and covered by the project's ignored data/ directory.\n\n"
        "Reproduce from the project root with:\n\n```text\npython scripts/prepare_antarctic_background.py --execute\n```\n"
        "The script never infers or trains and preserves source annotations, Nature and old experiment outputs.\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["status","source_images","prepared_images","splits","scenes","skipped"]},indent=2),flush=True)
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--margin",type=int,default=16)
    parser.add_argument("--reference-size",type=int,default=960)
    args = parser.parse_args()
    if not args.execute:
        print("Plan only: private TRAIN images -> whole-RGB/background-only masks; no training.")
        return
    run(args.output,args.margin,args.reference_size)


if __name__ == "__main__":
    main()
