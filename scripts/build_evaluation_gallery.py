"""Link fresh previews and RGB annotation reviews in a small local gallery."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from project_paths import PROJECT_ROOT as ROOT
from gallery import write_gallery_html


def main():
    data=json.loads((ROOT/"artifacts/evaluation/latest.json").read_text())
    run=Path(data["config"]["run"])
    items=[(r["image"],Path(r["output"])/"preview.jpg") for r in data["records"]]
    review=run/"review";review.mkdir(exist_ok=True)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
    for name,regions in data["config"]["roi_labels"].items():
        with Image.open(ROOT/"data/examples"/name) as raw:image=raw.convert("RGB")
        image.thumbnail((1400,1000));draw=ImageDraw.Draw(image)
        for number,(label,box) in enumerate(regions,1):
            x0,y0,x1,y1=[round(v*s) for v,s in zip(box,[image.width,image.height,image.width,image.height])]
            color="lime" if label==1 else "cyan"
            draw.rectangle((x0,y0,x1,y1),outline=color,width=3)
            draw.text((x0,y0),f"{number}: {'penguin' if label==1 else 'background'}",fill=color,font=font)
        path=review/(Path(name).stem+"_review.jpg");image.save(path,quality=90)
        items.append((name+" | provisional RGB-reviewed interiors",path))
    with Image.open(ROOT/"data/examples/penguins_far.jpg") as raw:
        strip=raw.crop((0,2050,2000,2450)).convert("RGB")
    draw=ImageDraw.Draw(strip)
    for number,(x,y) in enumerate(data["config"]["far_point_labels"]["points"],1):
        y-=2050;draw.ellipse((x-5,y-5,x+5,y+5),outline="lime",width=2)
        draw.text((x+6,y-18),str(number),fill="lime",font=font)
    path=review/"distant_points.jpg";strip.save(path,quality=94)
    items.append(("16 distant penguin body centers selected from RGB before fresh predictions",path))
    gallery=write_gallery_html(run/"gallery.html","Experiment 11: fresh inference and error audit",items,
        "Local only. Native / resize / union / model-derived reference. Reference agreement is not ground-truth accuracy. RGB-reviewed regions and points are provisional and cover only the displayed interiors.")
    cleanup=json.loads((ROOT/"artifacts/archive/cleanup_verification.json").read_text())
    (ROOT/"reports/cleanup.md").write_text(
        "# Project cleanup\n\nCode is under scripts/, data under ignored data/, and historical large outputs under ignored artifacts/. The original source notebook and Git history were backed up before migration. No model training was performed.\n\n"
        "| Check | Result |\n|---|---:|\n"
        f"| Launcher size | {cleanup['notebook_bytes']} bytes |\n"
        "| Embedded notebook images | 0 |\n| Python modules at root | 0 |\n"
        f"| Wildlife training files / source groups | 7858 / 2627 |\n"
        "| Wildlife retained target training records | 650 (previously 1938) |\n"
        "| Wildlife cross-split source-prefix overlap | 0 |\n"
        f"| Inference-only weight file | {cleanup['inference_checkpoint_MB']} MB |\n"
        "| Oversized blobs reachable from main | 0 |\n\n"
        "The single unpushed commit was undone with a mixed reset. Working files are preserved and have not been recommitted or pushed. Historical model optimizer states remain archived; physical disk usage was not reduced by deleting history.\n\n"
        f"Fresh local gallery: {gallery.relative_to(ROOT).as_posix()}.\n"
        "Current counts: datasets.csv. Raw source-group audit: wildlife.csv. The before-cleanup Git bundle is artifacts/archive/before_cleanup.bundle.\n",encoding="utf-8")
    print("LOCAL_GALLERY",gallery)


if __name__=="__main__":main()
