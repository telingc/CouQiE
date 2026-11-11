"""Write a small English project entry point and summarize preserved artifacts."""
from pathlib import Path
import csv
import json
import zipfile
import nbformat
from project_paths import PROJECT_ROOT as ROOT


def main():
    report=ROOT/"reports";report.mkdir(exist_ok=True)
    # Preserve previous small report versions before the fresh evaluation replaces them.
    archive=ROOT/"artifacts/archive"
    backup=archive/"pre_cleanup_reports.zip"
    if not backup.exists():
        with zipfile.ZipFile(backup,"w",zipfile.ZIP_DEFLATED) as z:
            for p in report.rglob("*"):
                if p.is_file():z.write(p,str(p.relative_to(report)))
    (ROOT/".gitignore").write_text("/data/\n/artifacts/\n**/__pycache__/\n**/.ipynb_checkpoints/\n*.pyc\n*.tmp\n",encoding="utf-8")
    (ROOT/"README.md").write_text('''# Antarctic Animal Segmentation

Hand-written ResNet-34 U-Net for background, penguin, other bird, seal, and whale.
Code, lightweight reports, and large local artifacts have separate homes.

| Path | Purpose | Git |
|---|---|---|
| scripts/ | Model, datasets, training, experiments, evaluation, and maintenance | Track |
| launch.ipynb | Small entry point; no embedded pictures or saved outputs | Track |
| reports/ | Short summaries, CSV metrics, configurations, and selected previews | Track |
| data/ | Original/prepared datasets, example photographs, and private course images | Ignore |
| artifacts/ | Checkpoints, complete predictions, historical runs, logs, and backups | Ignore |

## Use

Run from the project root with the existing Python environment:

```powershell
& E:\\workspace\\pyscripts\\.venv\\Scripts\\python.exe scripts\\evaluate_project.py
```

Open launch.ipynb after restarting its kernel. Its default cells do not train.
Historical experiment functions remain in scripts/experiment1.py through
scripts/experiment13.py. Use a new run directory for new training: the dataset
view now deduplicates Wildlife source variants. Existing runs are historical.

## Read results

- [Experiment index](reports/index.csv)
- [Fresh experiment-11 audit](reports/exp11/README.md)
- [Dataset policy](DATASETS.md)
- Full local results: artifacts/evaluation/latest.json and its gallery.html.

Model-derived stitched references measure agreement, not independent accuracy.
ROI audits are provisional visual-review interiors; original validation labels
have known limitations. No course-image pixels are stored in tracked reports.

## Storage policy

Keep class-ID masks once; render overlays on demand. Use external image links,
not notebook attachments. Keep only small curated previews in reports. Full
checkpoints and archived notebooks remain local. The migration backup is
artifacts/archive/before_cleanup.bundle; source/metadata snapshots are beside it.
''',encoding="utf-8")
    (ROOT/"DATASETS.md").write_text('''# Dataset policy

Class IDs: 0 background, 1 penguin, 2 other bird, 3 seal, 4 whale; 255 ignored.
All local datasets are under data/. Original files are preserved.

| Source | Label policy | Limitation |
|---|---|---|
| dataset_penguins | Original COCO polygons | Some distant penguins are unlabeled |
| dataset_birds | CUB gray masks mapped to other bird | Mainly non-Antarctic species |
| dataset_seals | Recovered SealID silhouettes | Approximate masks, not manual ground truth |
| dataset_wildlife | COCO polygons; unknown animals ignored | Training exports contain same-source variants |
| dataset_nature | Explicit natural stuff is background; other/padded pixels ignored | Not a native high-resolution Antarctic collection |

The loader keeps one Wildlife record per original filename prefix before .rf.,
with deterministic filename ordering. Cross-split priority is test, valid, train.
It also removes byte-identical image overlaps. Raw variants are not deleted.
This establishes source-prefix isolation, not a guarantee of no unrelated near-duplicates.

The fresh source-count audit is in reports/datasets.csv. Historical checkpoints
were trained before source-group deduplication; their training metadata is retained.
All resize/crop operations must transform images and masks together, using nearest
interpolation for masks. Prepared inputs preserve aspect ratio with ignored padding.
''',encoding="utf-8")
    cells=[nbformat.v4.new_markdown_cell("# Antarctic Segmentation\n\nA lightweight launcher. Restart the kernel after migration. Images stay in external files; no attachments are embedded.\n\n[Project README](README.md) | [Experiment index](reports/index.csv) | [Experiment 11 audit](reports/exp11/README.md)"),
     nbformat.v4.new_code_cell('from pathlib import Path\nimport sys\nROOT = Path.cwd().resolve()\nif not (ROOT / "scripts").is_dir():\n    raise RuntimeError("Start this notebook from the project root")\nsys.path.insert(0, str(ROOT / "scripts"))\n'),
     nbformat.v4.new_markdown_cell("## Reports\n\nThe preview below is a relative file link, not an embedded attachment. It contains public examples only.\n\n![Experiment 11 preview](reports/exp11/preview.jpg)"),
     nbformat.v4.new_code_cell('import pandas as pd\npd.read_csv(ROOT / "reports/index.csv")'),
     nbformat.v4.new_markdown_cell("## Fresh evaluation\n\nThis reruns inference and metrics without training. Model-reference agreement and provisional ROI accuracy are reported separately."),
     nbformat.v4.new_code_cell('RUN_EVALUATION = False\nif RUN_EVALUATION:\n    from evaluate_project import run_evaluation\n    evaluation = run_evaluation()'),
     nbformat.v4.new_markdown_cell("## Training\n\nTraining is explicit. Select a protocol from scripts/experiment*.py and a new output directory. Do not reuse a completed historical run after changing the dataset policy. Full states and private outputs belong under ignored storage.")]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    nbformat.validate(nb);nbformat.write(nb,ROOT/"launch.ipynb")
    rows=[]
    for number in range(1,14):
        folder=ROOT/"artifacts/experiments"/f"exp{number}_results"
        candidates=[]
        for p in folder.glob("latest*.json"):
            d=json.loads(p.read_text(encoding="utf-8"));s=d.get("selected",{})
            if isinstance(s,dict) and isinstance(s.get("metrics"),dict) and "foreground_miou" in s["metrics"]:
                candidates.append((s["metrics"]["foreground_miou"],p,d,s))
        if candidates:
            _,p,d,s=max(candidates,key=lambda v:v[0])
            rows.append(dict(experiment=number,status="historical_labeled_result",foreground_miou=s["metrics"]["foreground_miou"],best_epoch=s.get("best_epoch"),checkpoint=s.get("checkpoint"),metadata=str(p.relative_to(ROOT)),report="reports/exp11/README.md" if number==11 else ""))
        else:
            rows.append(dict(experiment=number,status="historical_qualitative_or_resource_test" if folder.exists() else "not_run",metadata=str(folder.relative_to(ROOT))))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (report/"index.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)
    dataset=json.loads((ROOT/"artifacts/data_reports/current_datasets.json").read_text())
    with (report/"datasets.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.writer(stream);w.writerow(["source","train","valid","test"])
        for source in dataset["sources"]["train"]:w.writerow([source]+[dataset["sources"][s][source] for s in ("train","valid","test")])
    print("Notebook bytes:",(ROOT/"launch.ipynb").stat().st_size)
    print("Project entry point and report index updated.")


if __name__=="__main__":main()
