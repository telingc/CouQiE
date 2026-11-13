"""Build concise English experiment reports from fixed-reference evaluations."""
from pathlib import Path
import csv
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

from project_paths import PROJECT_ROOT as ROOT
from reference_benchmark import CLASS_NAMES, BENCHMARK_VERSION, class_mask, write_csv, write_json

COLORS = np.zeros((256, 3), dtype=np.uint8)
for value, color in {0:(166,133,245),1:(255,65,77),2:(255,167,38),3:(0,190,220),4:(61,190,90),255:(155,163,175)}.items():
    COLORS[value] = color
PHOTOS = ("penguins_far.jpg", "penguins_near.jpg", "penguin_colony.jpg", "antarctic_landscape_untitled_2.jpg")
FONT = "C:/Windows/Fonts/arial.ttf"
COMMON = ("Historical binary/five-class replay protocol (experiments 1-12). Fixed development references: three penguin photos and six full-background landscapes. "
    "The colony uses the user-edited mask; far uses the approved bands and near uses experiment-11 resize. "
    "These masks are diagnostic references, not an independent blind test. No positive other-bird, seal, or whale labels are present. "
    "Reported new photo scores are separate from historical 1,915-image animal validation. "
    "Standard letterbox resize uses CUDA AMP; legacy binary tile/stretch paths use float32. Native inference has a recorded float32 CPU fallback after GPU OOM. "
    "Historical models predate Wildlife source-group deduplication; no retraining was performed.")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(value):
    return "N/A" if value in (None, "") else f"{float(value)*100:.2f}%"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)])


class Reports:
    def __init__(self):
        self.folder = ROOT / "artifacts/reference_benchmark" / BENCHMARK_VERSION
        self.result = read(self.folder / "result.json")
        if self.result["status"] != "complete":
            raise ValueError("Complete the reference benchmark first")
        self.reference_folder = ROOT / "data/reference_masks" / BENCHMARK_VERSION
        self.references = read(self.reference_folder / "manifest.json")
        self.entries = {e["image"]: e for e in self.references["entries"]}
        self.models = {m["spec"]["id"]: m for m in self.result["models"]}
        with (self.folder / "metrics.csv").open(encoding="utf-8-sig") as stream:
            self.rows = list(csv.DictReader(stream))
        # Derive the equal-photo average from existing per-image scores.
        for row in self.rows:
            if row['scope'] == 'pooled' and not row.get('mean_penguin_photo_iou') and row['image_group'] != 'landscapes':
                values = [float(r['penguin_iou']) for r in self.rows if r['scope']=='image'
                    and r['variant']==row['variant'] and r['method']==row['method']
                    and r['image_group']=='penguin_photos' and r['penguin_iou']!='']
                row['mean_penguin_photo_iou'] = float(np.mean(values)) if values else None
            # This frozen set contains only background and penguin positives.
            # Recover exact conditional class counts from the saved pixel ratios.
            positive = int(row['penguin_tp']) + int(row['penguin_fn'])
            if positive:
                total = int(row['pixels'])
                background = total - positive
                for name in CLASS_NAMES[2:]:
                    predicted = round(float(row[name+'_area'])*total)
                    on_background = round(float(row[name+'_on_background'])*background)
                    row[name+'_on_penguin'] = (predicted-on_background)/positive
                partition = float(row['penguin_recall']) + float(row['animal_miss']) + sum(row[name+'_on_penguin'] for name in CLASS_NAMES[2:])
                if abs(partition-1)>1e-8:raise ValueError('Penguin error counts do not partition the reference pixels')
        point_path = self.folder / 'point_check.json'
        if not point_path.exists():
            previous = read(ROOT / 'artifacts/evaluation/latest.json')['config']['far_point_labels']
            write_json(point_path, dict(**previous, image='penguins_far.jpg',
                image_sha256=self.entries['penguins_far.jpg']['image_sha256'],
                note='Auxiliary positive body centers selected from RGB before earlier fresh predictions; not instance recall.'))
        self.point_check = read(point_path)
        if self.point_check['image_sha256'] != self.entries['penguins_far.jpg']['image_sha256']:
            raise ValueError('Distant-point source photo changed')
        for row in self.rows:
            if row['scope']=='image' and row['image']=='penguins_far.jpg':
                mask = class_mask(self.prediction(row['variant'],row['method'],row['image']))
                points = self.point_check['points']
                row['distant_penguin_points'] = sum(int(mask[y,x])==1 for x,y in points)
                row['distant_point_count'] = len(points)
                row['distant_point_accuracy'] = row['distant_penguin_points']/len(points)
        self.lookup = {(r["variant"], r["method"], r["image_group"]):r for r in self.rows if r["scope"] == "pooled"}
        self.index = []

    def model(self, variant):
        return self.models[variant]["model"]

    def prediction(self, variant, method, name):
        record = next(r for r in self.models[variant]["records"] if r["image"] == name)
        return ROOT / record["masks"][method]["path"]

    def score_table(self, protocols):
        rows = []
        for title, variant, method in protocols:
            animals = self.lookup[(variant, method, "penguin_photos")]
            negatives = self.lookup[(variant, method, "landscapes")]
            points = next(r for r in self.rows if r['scope']=='image' and r['variant']==variant and r['method']==method and r['image']=='penguins_far.jpg')
            rows.append([title, percent(animals.get("mean_penguin_photo_iou")), percent(animals["penguin_iou"]),
                percent(animals["penguin_precision"]), percent(animals["penguin_recall"]),
                percent(animals["animal_class_confusion"]), percent(negatives["background_fp"]),
                f"{points['distant_penguin_points']}/{points['distant_point_count']}"])
        return md_table(["Protocol", "Mean photo IoU (3)", "Pooled penguin IoU", "Penguin precision", "Penguin recall", "Wrong animal class", "Landscape FP (6)", "Far penguin centers"], rows)

    def photo_table(self, protocols):
        rows = []
        for title, variant, method in protocols:
            for name in ("penguins_far.jpg", "penguins_near.jpg", "penguin_colony.jpg"):
                row = next(r for r in self.rows if r["scope"] == "image" and r["variant"] == variant and r["method"] == method and r["image"] == name)
                rows.append([title, Path(name).stem, percent(row["penguin_iou"]), percent(row["penguin_recall"]), percent(row["background_fp"])])
        return md_table(["Protocol", "Photo", "Penguin IoU", "Penguin recall", "Background FP"], rows)

    def landscape_class_table(self, protocols):
        rows=[]
        for title,variant,method in protocols:
            row=self.lookup[(variant,method,'landscapes')]
            rows.append([title,*[percent(row[name+'_on_background']) for name in CLASS_NAMES[1:]]])
        text = ('False-positive pixels by predicted class, divided by all pixels of the six landscapes:\n\n'
            +md_table(['Protocol','Penguin FP','Other bird FP','Seal FP','Whale FP'],rows))
        errors=[]
        for title,variant,method in protocols:
            row=self.lookup[(variant,method,'penguin_photos')]
            errors.append([title,*[percent(row[name+'_on_penguin']) for name in CLASS_NAMES[2:]],percent(row['animal_miss'])])
        return text+'\n\nTrue penguin pixels assigned another class, divided by all true penguin pixels on the three penguin photos:\n\n'+md_table(['Protocol','Penguin to bird','Penguin to seal','Penguin to whale','Penguin to background'],errors)

    def historical_table(self, variants):
        rows = []
        for label, variant in variants:
            model = self.model(variant)
            metrics = model.get("historical_validation") or {}
            config = model.get("config") or {}
            rows.append([label, model["epoch"], config.get("epochs", "unrecorded"), config.get("batch_size", "unrecorded"),
                config.get("bn_policy", "unrecorded"), model["kernel_size"], percent(metrics.get("foreground_miou")),
                *[percent(v) for v in metrics.get("iou_per_class", [None]*5)[1:]]])
        return md_table(["Checkpoint", "Selected epoch", "Epoch budget", "Batch", "BN policy", "Kernel", "Animal mIoU", "Penguin IoU", "Bird IoU", "Seal IoU", "Whale IoU"], rows)

    def comparison(self, target, columns, names=PHOTOS):
        width, height = 380, 300
        canvas = Image.new("RGB", (width * (len(columns)+2), 72 + height * len(names)), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(FONT, 18)
        small = ImageFont.truetype(FONT, 16)
        draw.text((12, 7), "Development references | purple: background; red: penguin; orange: bird; cyan: seal; green: whale", font=font, fill="black")
        for col, title in enumerate(["Original", "Fixed reference"] + [c[0] for c in columns]):
            draw.text((col*width+8, 34), title, font=font, fill="black")
        for row, name in enumerate(names):
            with Image.open(ROOT / self.entries[name]["image_path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((width-10, height-38))
            paths = [None, self.reference_folder / self.entries[name]["mask"]] + [self.prediction(variant, method, name) for _,variant,method in columns]
            for col, path in enumerate(paths):
                preview = image
                if path:
                    with Image.open(path) as raw:
                        values = np.asarray(raw.resize(image.size, Image.Resampling.NEAREST))
                    preview = Image.blend(image, Image.fromarray(COLORS[values]), .43)
                x = col*width+(width-preview.width)//2
                y = 72 + row*height + 20
                canvas.paste(preview, (x,y))
                if col == 0:
                    draw.text((col*width+8,72+row*height),Path(name).stem.replace("antarctic_landscape_", "landscape_"),font=small,fill="black")
        canvas.save(target, quality=86, optimize=True)

    def write(self, number, title, purpose, procedure, protocols, columns, conclusion, extra="", historical=(), status="completed"):
        folder = ROOT / "reports" / f"exp{number:02d}"
        # Reuse the existing experiment-11 path, keeping its detailed audit CSVs.
        if number == 11:
            folder = ROOT / "reports/exp11"
        folder.mkdir(exist_ok=True)
        selected = {(variant, method) for _,variant,method in protocols}
        rows = [r for r in self.rows if (r["variant"],r["method"]) in selected]
        if rows:
            write_csv(folder / "metrics.csv", rows)
        lines = [f"# Experiment {number}: {title}", "", purpose, "", procedure, ""]
        if protocols:
            lines += [self.score_table(protocols), "", self.photo_table(protocols), ""]
        if historical:
            lines += ["Historical labeled-validation results:", "", self.historical_table(historical), ""]
        if extra:
            lines += [extra, ""]
        if protocols:
            best_iou = max(protocols,key=lambda p:float(self.lookup[(p[1],p[2],'penguin_photos')]['mean_penguin_photo_iou']))
            lowest_fp = min(protocols,key=lambda p:float(self.lookup[(p[1],p[2],'landscapes')]['background_fp']))
            conclusion += (f" Among the listed protocols on this set, {best_iou[0]} has the highest mean photo IoU "
                f"({percent(self.lookup[(best_iou[1],best_iou[2],'penguin_photos')]['mean_penguin_photo_iou'])}); "
                f"{lowest_fp[0]} has the lowest landscape FP ({percent(self.lookup[(lowest_fp[1],lowest_fp[2],'landscapes')]['background_fp'])}).")
        lines += [conclusion, ""]
        if columns:
            self.comparison(folder / "comparison.jpg", columns)
            lines += ["![Photo comparison](comparison.jpg)", ""]
        lines += [COMMON, "", "Mean photo IoU averages the three penguin photos equally. Pooled IoU/precision/recall pool pixel counts on those photos. Landscape FP is all predicted animal pixels / all pixels of the six negative landscapes. Wrong animal class is true penguin pixels assigned bird/seal/whale / all true penguin pixels. Undefined metrics are N/A.", "", "[Reference-set protocol](../references/benchmark.md) | [All experiment reports](../README.md)"]
        if number == 11:
            lines += ["", "The previous partial-ROI audit is preserved under artifacts/reference_benchmark/previous_exp11_report/. Its provisional-ROI denominators differ from this full-reference evaluation."]
        (folder / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        row = dict(experiment=number, title=title, status=status, report=(folder / "README.md").relative_to(ROOT).as_posix(), reference_version=BENCHMARK_VERSION)
        if protocols:
            _, variant, method = protocols[0]
            score = self.lookup[(variant, method, "penguin_photos")]
            row.update(first_protocol=protocols[0][0], first_protocol_mean_photo_iou=score.get("mean_penguin_photo_iou"),
                first_protocol_landscape_fp=self.lookup[(variant,method,"landscapes")]["background_fp"])
        self.index.append(row)
        return folder

    def build(self):
        old_report = ROOT / 'reports/exp11'
        archive = ROOT / 'artifacts/reference_benchmark/previous_exp11_report'
        if (old_report / 'README.md').exists() and 'fresh error audit' in (old_report / 'README.md').read_text(encoding='utf-8').splitlines()[0]:
            archive.mkdir(parents=True, exist_ok=True)
            for source in old_report.iterdir():
                target = archive / source.name
                if not source.resolve().is_relative_to(ROOT.resolve()) or not target.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Report archive path escapes the workspace')
                if target.exists():
                    raise FileExistsError(target)
                source.rename(target)
        index_backup = ROOT / 'artifacts/reference_benchmark/previous_report_index.csv'
        if not index_backup.exists():
            shutil.copy2(ROOT / 'reports/index.csv', index_backup)
        binary = lambda label,method:(label,"binary",method)
        self.write(1,"Sliding windows and image resize",
            "Compare local high-resolution detail with whole-image context for binary penguin segmentation.",
            "Retrospective replay uses the surviving epoch-9 binary checkpoint. Tiles are 640 pixels with stride 480; overlapping logits are averaged. The historical resize path stretches the input to 640x640 and restores logits before classification. Original screenshots may have used an earlier overwritten best checkpoint.",
            [binary("Tiles 640/480","tiles"),binary("Stretch resize 640","stretch_resize")],
            [binary("Tiles 640/480","tiles"),binary("Stretch resize","stretch_resize")],
            "The per-photo rows quantify the near/far tradeoff. Background errors on negative landscapes are reported separately from penguin recall.")
        self.write(2,"Single probability fusion",
            "Test whether averaging tile and whole-image scores balances detail and context.",
            "Replay uses the same binary checkpoint and branch predictions as experiment 1. The preserved notebook parameter is alpha=0.6 for whole-image resize and 0.4 for tiles; the final threshold is strictly greater than 0.5.",
            [binary("Fusion alpha .6","fusion_060"),binary("Tiles","tiles"),binary("Stretch resize","stretch_resize")],
            [binary("Tiles","tiles"),binary("Resize","stretch_resize"),binary("Fusion .6","fusion_060")],
            "Averaging can suppress a weak branch response; it does not guarantee the union of detections. These are uncalibrated model scores.")
        self.write(3,"Union of two biased fusions",
            "Preserve detections favored by either a local or global weighted fusion.",
            "The preserved notebook used resize weights 0.2 and 0.8, not the module defaults 0.25 and 0.75. Each weighted map is thresholded at >0.5; their binary masks are unioned.",
            [binary("Dual union .2/.8","dual_union"),binary("Tile-biased .2","tile_biased_020"),binary("Resize-biased .8","resize_biased_080")],
            [binary("Tile-biased .2","tile_biased_020"),binary("Resize-biased .8","resize_biased_080"),binary("Dual union","dual_union")],
            "The union retains the foreground accepted by either fusion and also retains its false positives. The reference-set table measures both consequences.")
        self.write(4,"Original-resolution direct inference",
            "Check whether a fully convolutional model can segment an original-size photo without tiles.",
            "The epoch-9 binary checkpoint is evaluated at original resolution. The final affine 1x1 head is applied before final bilinear interpolation; numerical parity is checked. CUDA AMP is used when feasible, with a recorded float32 CPU fallback after GPU OOM.",
            [binary("Native","native"),binary("Tiles","tiles"),binary("Stretch resize","stretch_resize")],
            [binary("Native","native"),binary("Tiles","tiles"),binary("Stretch resize","stretch_resize")],
            "Direct inference changes object scale relative to training. It avoids tile borders but can still produce missed objects and false detections; it is not automatically equivalent to tile inference.")
        self.write(5,"Five-class baseline and tuned configuration",
            "Extend binary penguin segmentation to background, penguin, other bird, seal, and whale.",
            "Compare the surviving original multiclass best checkpoint with the three-epoch tuned run (batch 4, frozen BN statistics, AMP). Both are evaluated with identical 640 letterbox-resize inference. Several settings and training budgets changed together.",
            [("Original","exp5_initial","resize"),("Tuned","exp5_tuned","resize")],
            [("Original resize","exp5_initial","resize"),("Tuned resize","exp5_tuned","resize")],
            "Historical validation improves substantially with the combined tuned configuration. This comparison cannot isolate batch size as the sole cause. The development references measure penguin/background transfer, not positive bird/seal/whale recall.",
            historical=[("Original","exp5_initial"),("Tuned","exp5_tuned")])
        self.resources()
        self.write(7,"BatchNorm policy comparison",
            "Compare updating BN running statistics with freezing them during multiclass training.",
            "The recorded controlled runs use the same binary initialization, batch 6, seed 42, and ten-epoch budget. Compare each policy's best checkpoint using the same fixed-reference inference protocol.",
            [("Normal BN","exp7_normal","resize"),("Frozen statistics","exp7_frozen_stats","resize")],
            [("Normal BN resize","exp7_normal","resize"),("Frozen BN resize","exp7_frozen_stats","resize")],
            "Normal BN has the higher historical full-animal validation mIoU. The new photo results test whether that ordering transfers to these high-resolution scenes.",
            historical=[("Normal BN","exp7_normal"),("Frozen statistics","exp7_frozen_stats")])
        architecture=read(ROOT/'artifacts/experiments/exp8_results/architecture_comparison.json')
        architecture_table=md_table(["Kernel","Parameters","Deepest theoretical receptive field"],[(r['kernel'],f"{r['parameters']:,}",r['deepest_theoretical_receptive_field']) for r in architecture])
        architecture_table+='\n\nThe first combined resource/pilot attempt was interrupted after the 3x3 pilot and part of the 5x5 pilot, before full larger-kernel training. The requested 5x5 model was then trained separately on November 9; the 7x7 continuation was completed on November 10. Pilot results are not counted as completed ten-epoch runs.'
        self.write(8,"Larger convolution kernels",
            "Test 5x5 and 7x7 convolutions against the 3x3 baseline.",
            "Use completed ten-epoch runs with the recorded feasible batch sizes. Encoder/decoder 3x3 convolutions were enlarged; the stem and 1x1 layers stayed unchanged. Initial weights were center-copied with zero rings. Compare best checkpoints under identical resize inference.",
            [("3x3","exp7_normal","resize"),("5x5","exp8_kernel5","resize"),("7x7","exp8_kernel7","resize")],
            [("3x3 resize","exp7_normal","resize"),("5x5 resize","exp8_kernel5","resize"),("7x7 resize","exp8_kernel7","resize")],
            "Historical animal mIoU is 83.56%, 84.62%, and 79.90% for 3x3, 5x5, and 7x7. Larger kernels do not produce a monotonic improvement. All three completed comparisons use batch 6. Enlarging kernels also increases parameter count and changes optimization, so the results do not isolate receptive field as the cause of errors.",extra=architecture_table,
            historical=[("3x3","exp7_normal"),("5x5","exp8_kernel5"),("7x7","exp8_kernel7")])
        protocols=[(f"{k}x{k} {method}",variant,method) for k,variant in [(3,"exp7_normal"),(5,"exp8_kernel5"),(7,"exp8_kernel7")] for method in ("native","resize","union")]
        self.write(9,"Native/resize foreground union",
            "Measure the recall and false-positive effects of combining native and resized five-class predictions.",
            "For each 3x3/5x5/7x7 checkpoint, infer both original resolution and 640 letterbox resize. Keep foreground from either branch; resolve conflicting animal classes using the branch's selected-class score. No model is trained in this experiment. The figure shows the 5x5 run; the table includes all three kernels.",
            protocols,[("5x5 native","exp8_kernel5","native"),("5x5 resize","exp8_kernel5","resize"),("5x5 union","exp8_kernel5","union")],
            "A foreground union cannot remove a foreground false positive already present in either branch. Class-conflict resolution may also change strict penguin accuracy. Use the table to judge recall against background error.")
        self.write(10,"Explicit background supervision",
            "Test whether adding natural-background examples suppresses animal false positives.",
            "Compare the experiment-8 5x5 baseline with the completed ten-epoch background-supervised run. The recorded sampler gives equal probability mass to background and each of the four animal groups. Use identical reference photos and inference protocols.",
            [("Baseline resize","exp8_kernel5","resize"),("Background resize","exp10_best","resize"),("Background native","exp10_best","native"),("Background union","exp10_best","union")],
            [("Baseline resize","exp8_kernel5","resize"),("Background resize","exp10_best","resize"),("Background native","exp10_best","native")],
            "Historical full-animal mIoU fell from 84.62% to 83.28%. Background supervision is a distribution tradeoff; improvements on a natural-background holdout do not establish improvement on every Antarctic landscape. Equal sampling-group mass does not balance pixels or equalize per-image exposure.",
            historical=[("Baseline","exp8_kernel5"),("Background supervised","exp10_best")])
        self.write(11,"Penguin-weighted replay",
            "Recover penguin performance while retaining explicit background and other-animal replay.",
            "The five-epoch run starts from the full experiment-8 5x5 model. The sampler assigns penguin 50%, background 20%, and each other group 10%, with 2,000 draws per epoch. Epoch 3 is the best checkpoint. Verified saved branch masks are reused and rescored against the newly frozen full reference masks.",
            [("Native","exp11_best","native"),("Resize","exp11_best","resize"),("Union","exp11_best","union")],
            [("Native","exp11_best","native"),("Resize","exp11_best","resize"),("Union","exp11_best","union")],
            "Historical full-animal mIoU reaches 86.94%. Far/near reference masks originate from this model, so these development scores may favor it. Near uses this checkpoint's own resize mask: its 100% IoU is a self-consistency result, not an independent accuracy estimate. The negative landscapes and manually revised colony provide additional error checks; they do not turn the set into a blind benchmark.",
            extra=self.landscape_class_table([('Native','exp11_best','native'),('Resize','exp11_best','resize'),('Union','exp11_best','union')]),
            historical=[("Best epoch 3","exp11_best")])
        twelve=read(ROOT/'artifacts/experiments/exp12_results/latest.json')
        stage_table=md_table(["Stage","Epoch budget","Best animal mIoU","End-of-stage animal mIoU"],[(r['stage_name'],r['completed_epochs'],percent(r['metrics']['foreground_miou']),percent(r['final_metrics']['foreground_miou'])) for r in twelve['stages']])
        self.write(12,"Sequential training without replay",
            "Evaluate the requested background/wildlife, bird, whale, seal, and penguin training sequence.",
            "The recorded schedule is 10+15+5+5+5 epochs, initialized from binary features plus a fresh five-class head. Each stage passes its last checkpoint to the next stage without replay. Evaluate the globally selected best checkpoint and the final checkpoint separately.",
            [("Overall best resize","exp12_best","resize"),("Final resize","exp12_final","resize"),("Overall best native","exp12_best","native"),("Final native","exp12_final","native")],
            [("Best resize","exp12_best","resize"),("Final resize","exp12_final","resize"),("Final native","exp12_final","native")],
            "The overall selected checkpoint is from stage 1 (30.44% historical animal mIoU); the final checkpoint is 6.89%. Later sequential stages reduce performance on the combined validation set, consistent with forgetting and specialization. The final model can nevertheless score better on penguin-only development references: overall best refers to the historical multi-animal selection criterion. Displaying only the selected best would hide the final outcome.",extra=stage_table,
            historical=[("Overall best","exp12_best"),("Final","exp12_final")])
        self.current_experiments()
        self.reference_report()
        self.summary()

    def resources(self):
        data=read(ROOT/'artifacts/experiments/exp6_results/latest.json')
        folder=ROOT/'reports/exp06';folder.mkdir(exist_ok=True)
        write_csv(folder/'metrics.csv',data['results'])
        fig,axes=plt.subplots(1,2,figsize=(11,3.6))
        for layout in [False,True]:
            rows=[r for r in data['results'] if r['status']=='ok' and r['channels_last']==layout]
            label='channels-last' if layout else 'contiguous'
            axes[0].plot([r['batch_size'] for r in rows],[r['images_per_second'] for r in rows],'o-',label=label)
            axes[1].plot([r['batch_size'] for r in rows],[r['peak_allocated_GiB'] for r in rows],'o-',label=label)
        axes[0].set(xlabel='Batch size',ylabel='Images / second',title='Historical cached-input training benchmark')
        axes[1].set(xlabel='Batch size',ylabel='Peak allocated GiB',title='GPU memory')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/'comparison.jpg',dpi=140);plt.close(fig)
        rows=[[r['batch_size'],r['channels_last'],r['status'],f"{r.get('images_per_second',0):.2f}",f"{r.get('peak_allocated_GiB',0):.2f}"] for r in data['results']]
        text="# Experiment 6: batch size and throughput\n\nSelect a practical GPU batch size before accuracy experiments. The historical benchmark uses cached inputs, AMP, real backward/AdamW steps, and an 80% GPU-memory cap; it excludes image loading and decoding. No accuracy checkpoint or ten-epoch run is defined for this resource test.\n\n"
        text+=md_table(['Batch','Channels-last','Status','Images/s','Peak allocated GiB'],rows)
        text+="\n\nSelected batch: **6**, contiguous layout, **36.92 images/s**, **2.72 GiB** peak allocated. This was the smallest batch within 95% of measured maximum throughput; it is not a universal optimal batch size.\n\n![Resource benchmark](comparison.jpg)\n\n[All experiment reports](../README.md)\n"
        (folder/'README.md').write_text(text,encoding='utf-8')
        self.index.append(dict(experiment=6,title='Batch size and throughput',status='historical_resource_test',report='reports/exp06/README.md',reference_version='not_applicable'))

    def current_experiments(self):
        """Register separately evaluated experiments 13/14; never run a network."""
        from experiment13 import update_report
        update_report(ROOT)
        pointer=ROOT/'artifacts/experiments/exp13_results/latest_4class.json'
        completed=pointer.exists() and (Path(read(pointer)['folder'])/'training_result.json').exists()
        self.index.append(dict(experiment=13,title='New four-class ImageNet baseline',
            status='completed' if completed else 'prepared',report='reports/exp13/README.md',reference_version='benchmark_v2'))
        scale_pointer=ROOT/'artifacts/experiments/exp14_results/latest.json'
        if scale_pointer.exists():
            scale_result=read(scale_pointer)
            self.index.append(dict(experiment=14,title='Inference Scale Sensitivity',
                status='completed' if scale_result['status']=='complete' else scale_result['status'],
                report='reports/exp14/README.md',reference_version='benchmark_v2'))
        # Preserve later finalized reports; this historical builder never retrains them.
        for number,title,status,reference in [(15,'Five-class source baseline and proposed target adaptation','stage_a_complete_stage_b_not_run','frozen_target_split'),
                (16,'Six-group sampling with Antarctic background supervision','completed_group_b','SAM3_private_validation')]:
            if (ROOT/f'reports/exp{number}/README.md').is_file():
                self.index.append(dict(experiment=number,title=title,status=status,report=f'reports/exp{number}/README.md',reference_version=reference))

    def reference_report(self):
        folder=ROOT/'reports/references'
        rows=[]
        for entry in self.references['entries']:
            rows.append(dict(image=entry['image'],origin=entry['annotation_origin'],source_annotation=entry['source_annotation'],
                width=entry['width'],height=entry['height'],penguin_pixels=entry['pixel_counts'].get('1',0),mask_sha256=entry['mask_sha256']))
        write_csv(folder/'benchmark.csv',rows)
        text="# Fixed development reference set\n\n"+COMMON+f"\n\nCurrent reference version: **{BENCHMARK_VERSION}**. The user explicitly authorized all six filenames containing landscape as full-background references. The people photo is excluded. The colony uses edit_002 from data/examples; originals and both edit revisions are preserved. Far uses resize above 50%, native from 50-62.5%, and resize below 62.5%. Near uses the experiment-11 best checkpoint's resize mask by explicit user selection. This foreground-focused reference omits back-row penguins, and that source prediction matches it exactly. Class-ID masks are stored under ignored data/reference_masks/{BENCHMARK_VERSION}/. The prior benchmark_v1 masks and results are preserved; the other eight references are unchanged.\n\n"
        text+=md_table(['Image','Reference origin','Positive penguin pixels'],[(r['image'],r['origin'],f"{r['penguin_pixels']:,}") for r in rows])
        text+=f"\n\nMetrics are full-resolution pixel metrics with label 255 ignored. Background FP = predicted animal pixels on true background / true background pixels. Penguin false discovery = false penguin pixels / predicted penguin pixels. Wrong animal class counts true penguin pixels labeled bird, seal, or whale. Recall for absent positive classes is undefined. Per-photo IoU is supplied so the largest foreground object does not dominate every conclusion.\n\nAn auxiliary distant-point check uses 16 visible penguin body centers selected from RGB before the earlier fresh predictions. It measures point accuracy, not instance recall. These points can reveal omissions in the model-assisted full reference itself. Coordinates and provenance are frozen in artifacts/reference_benchmark/{BENCHMARK_VERSION}/point_check.json.\n\nThe images were used to design the inference methods and reference masks. Treat the set as development evaluation. The reference provenance and file hashes remain frozen; future edits should create a new benchmark version rather than silently changing this version.\n\n[Annotation editor instructions](README.md) | [All reports](../README.md)\n"
        (folder/'benchmark.md').write_text(text,encoding='utf-8')

    def summary(self):
        report_root=ROOT/'reports'
        self.index.sort(key=lambda r:r['experiment'])
        hist = lambda variant: percent((self.model(variant).get('historical_validation') or {}).get('foreground_miou'))
        photo = lambda variant,method: percent(self.lookup[(variant,method,'penguin_photos')].get('mean_penguin_photo_iou'))
        headlines={
            1:f"Tiles / stretch resize mean photo IoU: {photo('binary','tiles')} / {photo('binary','stretch_resize')}",
            2:f"Alpha .6 mean photo IoU: {photo('binary','fusion_060')}",
            3:f"Dual union mean photo IoU: {photo('binary','dual_union')}; landscape FP: {percent(self.lookup[('binary','dual_union','landscapes')]['background_fp'])}",
            4:f"Native mean photo IoU: {photo('binary','native')}",
            5:f"Original / tuned historical animal mIoU: {hist('exp5_initial')} / {hist('exp5_tuned')}",
            6:'Selected batch 6; 36.92 images/s; 2.72 GiB peak allocated (historical)',
            7:f"Normal / frozen historical animal mIoU: {hist('exp7_normal')} / {hist('exp7_frozen_stats')}",
            8:f"3x3 / 5x5 / 7x7 historical animal mIoU: {hist('exp7_normal')} / {hist('exp8_kernel5')} / {hist('exp8_kernel7')}",
            9:'Foreground union retains false positives from either branch; see matched method table',
            10:f"Baseline / background historical animal mIoU: {hist('exp8_kernel5')} / {hist('exp10_best')}",
            11:f"Historical animal mIoU: {hist('exp11_best')}; native/resize/union photo errors measured separately",
            12:f"Best / final historical animal mIoU: {hist('exp12_best')} / {hist('exp12_final')}",
            13:'Separate four-class baseline; see its own metrics, not a whale-only ablation',
            14:'Fixed experiment-13 weights; 75 float32 scale cases; see per-image diagnostics',
            15:'Source stage A completed; proposed foreground-label target fine-tuning not run',
            16:'Group B only; 10 epochs, selected epoch 6; private SAM3-reference mIoU 43.73%; fixed 960 inference'}
        for row in self.index:row['key_result']=headlines[row['experiment']]
        write_csv(report_root/'index.csv',self.index)
        write_csv(report_root/'metrics.csv',self.rows)
        shutil.copy2(self.folder/'class_metrics.csv',report_root/'class_metrics.csv')
        protocols=[('Binary baseline','binary','resize'),('Exp5 original','exp5_initial','resize'),('Exp5 tuned','exp5_tuned','resize'),
            ('Exp7 normal','exp7_normal','resize'),('Exp7 frozen','exp7_frozen_stats','resize'),('Exp8 5x5','exp8_kernel5','resize'),
            ('Exp8 7x7','exp8_kernel7','resize'),('Exp10','exp10_best','resize'),('Exp11','exp11_best','resize'),
            ('Exp12 best','exp12_best','resize'),('Exp12 final','exp12_final','resize')]
        labels=[title for title,_,_ in protocols]
        fig,axes=plt.subplots(1,3,figsize=(15.5,6.3),sharey=True)
        ious=[float(self.lookup[(v,m,'penguin_photos')]['mean_penguin_photo_iou'])*100 for _,v,m in protocols]
        fps=[float(self.lookup[(v,m,'landscapes')]['background_fp'])*100 for _,v,m in protocols]
        point_hits=[next(r['distant_penguin_points'] for r in self.rows if r['scope']=='image' and r['variant']==v and r['method']==m and r['image']=='penguins_far.jpg') for _,v,m in protocols]
        point_rates=[100*n/16 for n in point_hits]
        y=np.arange(len(labels))
        axes[0].barh(y,ious,color='#426bba');axes[1].barh(y,fps,color='#d78537')
        axes[2].barh(y,point_rates,color='#478a69')
        axes[0].set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Mean penguin-photo IoU (%)',title='Three fixed penguin references')
        axes[1].set(xlabel='False animal pixels (%)',title='Six user-declared negative landscapes')
        axes[2].set(xlabel='Correct penguin centers (%)',title='16 distant penguin centers',xlim=(0,115))
        axes[0].invert_yaxis()
        for ax,values in zip(axes,[ious,fps]):
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            for pos,value in enumerate(values):ax.text(value+.3,pos,f'{value:.2f}',va='center',fontsize=8)
        axes[1].set_xlim(0,max(2,max(fps)*1.16))
        axes[2].grid(axis='x',alpha=.2);axes[2].set_axisbelow(True)
        for pos,(rate,hits) in enumerate(zip(point_rates,point_hits)):axes[2].text(rate+.5,pos,f'{hits}/16',va='center',fontsize=8)
        fig.suptitle('Matched resize 640 | development references; not a blind test')
        fig.tight_layout();fig.savefig(report_root/'overview.jpg',dpi=150);plt.close(fig)
        # Keep the main cross-model table on a matched resize protocol.
        matched=[(title,v,'resize') for title,v,_ in protocols]
        text=f"# Experiment reports\n\n{len(self.index)} experiment entries have concise English reports. Experiments 1-4 are retrospective inference replays using the surviving binary checkpoint. Experiments 5 and 7-12 use preserved trained weights. Experiment 6 is a historical resource test; experiment 13 is a separate four-class baseline. Experiment 14, when available, reports fixed-weight float32 scale sensitivity separately from the historical AMP comparison. No training is performed by this report builder.\n\n"
        text+=md_table(['Experiment','Purpose','Result','Status','Report'],[(r['experiment'],r['title'],r['key_result'],r['status'],f"[Read]({Path(r['report']).relative_to('reports').as_posix()})") for r in self.index])
        text+='\n\nHistorical binary/five-class comparison only: 640 letterbox resize, restored logits, original-resolution scoring. Resize uses AMP as recorded below. Experiments 13 and 14 have separate four-class reports; their results are not pooled into this table.\n\n'+self.score_table(matched)
        text+='\n\n![Development-reference overview](overview.jpg)\n\nThe chart and table use the same resize protocol for every model. Direct/resize/union comparisons are in the individual reports.\n\n'+COMMON
        text+='\n\n[Reference definitions](references/benchmark.md) | [All image and pooled metrics](metrics.csv) | [Per-class pixel counts](class_metrics.csv)\n'
        text+='\n[Project overview](../README.md) | [Dataset policy](../DATASETS.md)\n\nThe register contains experiments 1-16. E15 foreground-label fine-tuning was not run; E16 group B is the final five-class model. The declined unnumbered experiment-10 preview remains excluded.\n'
        (report_root/'README.md').write_text(text,encoding='utf-8')


def main():
    Reports().build()
    print('Reports written to',ROOT/'reports')


if __name__=='__main__':
    main()
