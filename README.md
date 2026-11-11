# Antarctic Animal Segmentation

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
& E:\workspace\pyscripts\.venv\Scripts\python.exe scripts\reference_benchmark.py
& E:\workspace\pyscripts\.venv\Scripts\python.exe scripts\report_experiments.py
```

Open launch.ipynb after restarting its kernel. Its default cells do not train.
Historical experiment functions remain in scripts/experiment1.py through
scripts/experiment13.py. Use a new run directory for new training: the dataset
view now deduplicates Wildlife source variants. Existing runs are historical.

## Read results

- [Experiment index](reports/index.csv)
- [All experiment reports](reports/README.md)
- [Experiment 11 report](reports/exp11/README.md)
- [Fixed development references](reports/references/benchmark.md)
- [Dataset policy](DATASETS.md)
- [Reference mask editor](reports/references/README.md)
- Full local results: artifacts/evaluation/latest.json and its gallery.html.

The fixed development set contains three penguin references and six landscapes
declared background by the user. Model-assisted reference masks are not an
independent blind test. No course-image pixels are stored in tracked reports.
The active reference version is benchmark_v2: penguins_near uses experiment-11
resize; the eight other references are unchanged. To rescore saved predictions
without running networks, use scripts/reference_benchmark.py --rescore-only.
An additional full-dataset/ROI audit is available through scripts/evaluate_project.py;
its complete outputs stay under artifacts/evaluation/ and do not overwrite reports.

## Storage policy

Keep class-ID masks once; render overlays on demand. Use external image links,
not notebook attachments. Keep only small curated previews in reports. Full
checkpoints and archived notebooks remain local. The migration backup is
artifacts/archive/before_cleanup.bundle; source/metadata snapshots are beside it.
