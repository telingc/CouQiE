# Antarctic Animal Segmentation

The formal project closes at experiments **1-14**. The retained baseline is the
four-class experiment-13 hand-written ResNet34 U-Net: background, penguin, other
bird and seal. Experiment 14 diagnoses inference-scale sensitivity with those
fixed weights. Historical binary and five-class experiments remain reproducible.

Start with [project closeout](reports/CLOSEOUT.md), [all experiment reports](reports/README.md),
[code map](CODE_MAP.md), and [dataset policy](DATASETS.md).

| Location | Content | Git |
|---|---|---|
| scripts/ | Active model, data, experiment, reporting and checking code | Track |
| launch.ipynb | Small launcher with explicit execution switches; no saved outputs or images | Track |
| reports/ | English reports, metric CSVs and selected public figures | Track |
| data/ | Original/prepared datasets, references and private NJU photos | Ignore |
| artifacts/ | Checkpoints, full predictions, galleries, backups and rejected previews | Ignore |

## Read and verify

Open [launch.ipynb](launch.ipynb) from the project root in the existing Python
environment. All execution switches default to False. Reports can be read without
running the notebook or loading a model.

```powershell
python scripts/check_project.py
python scripts/check_project.py --local-results
```

The first command checks source syntax, local imports, documentation links, the
formal index, notebook defaults and private-file tracking. The second also hashes
local result sources and checks experiment-13/14 metadata. Neither command imports
torch, trains, runs inference or downloads data.

To regenerate reports from existing logs and saved predictions:

```powershell
python scripts/report_experiments.py
```

This redraws reports, including experiment-13 training curves; it does not run a
network or edit launch.ipynb. Local data and artifact caches are required. The
observed environment is recorded in [environment.json](reports/environment.json).
The local interpreter used for completed runs is
`E:\workspace\pyscripts\.venv\Scripts\python.exe`.

## Retained results

- [Experiment 13: four-class baseline and training curves](reports/exp13/README.md).
- [Experiment 14: fixed-weight scale sensitivity](reports/exp14/README.md).
- [Formal experiment index](reports/index.csv) and [historical comparison](reports/README.md).
- [Frozen benchmark_v2 definitions](reports/references/benchmark.md).
- [Reference review editor](reports/references/README.md).

The nine-photo development benchmark contains three penguin references and six
user-declared background landscapes. Some masks are model-assisted; these are not
independent blind-test scores. Animal validation, natural-background validation,
and development-photo metrics are separate. NJU has no manual masks here and is
visualization only. Course photos and their visualizations stay in ignored storage.

Four-class mIoU cannot be compared directly with historical five-class mIoU.
Historical cross-model reports use their recorded AMP protocol; experiment 14 uses
fresh float32 predictions. Scale sensitivity does not establish a universally best
input resolution or a kernel-only causal effect.

## Reproduction boundary

Training and full inference remain available through explicit notebook switches
and experiment APIs. Use new run directories when changing model/data settings;
verify checkpoint, image, preprocessing and precision hashes before reusing caches.
`reference_benchmark.py --rescore-only` rescales no images and scores saved class
masks on the CPU; its bare command can run historical inference and is not needed
for closeout. `prepare_datasets.py` and `download_nature.py` intentionally modify
local data and are not part of routine verification.

The unnumbered experiment-10 5x5 scale preview was declined and archived locally;
it is not experiment 15. One-time migration tools were also archived. See
[cleanup record](reports/cleanup.md). Checkpoints, source data, frozen reference
masks and historical predictions are preserved.
