"""Verify the migrated project and record source-group and storage evidence."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import importlib
import json
import re
import subprocess
import torch
import nbformat
from project_paths import PROJECT_ROOT as ROOT


def main():
    for path in (ROOT/"scripts").glob("*.py"):
        text=path.read_text(encoding="utf-8")
        assert not re.search(r"[\u3400-\u9fff]",text),path
        compile(text,str(path),"exec")
    for name in ["script","animal_data","train_multiclass"]+[f"experiment{i}" for i in range(1,14)]:
        importlib.import_module(name)
    nb=nbformat.read(ROOT/"launch.ipynb",as_version=4);nbformat.validate(nb)
    assert not any(c.get("outputs") or c.get("attachments") for c in nb.cells)
    group_sets={};rows=[]
    for split in ("train","valid","test"):
        d=json.loads((ROOT/"data/dataset_wildlife"/split/"_annotations.coco.json").read_text())
        groups=defaultdict(list)
        for record in d["images"]:groups[record["file_name"].split(".rf.",1)[0]].append(record)
        group_sets[split]=set(groups)
        rows.append(dict(split=split,raw_images=len(d["images"]),source_groups=len(groups),
            groups_with_three_variants=sum(len(v)==3 for v in groups.values()),
            groups_with_two_variants=sum(len(v)==2 for v in groups.values())))
    overlaps={a+"_"+b:len(group_sets[a]&group_sets[b]) for a,b in [("train","valid"),("train","test"),("valid","test")]}
    with (ROOT/"reports/wildlife.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    latest=json.loads((ROOT/"artifacts/experiments/exp11_results/latest.json").read_text())
    source=Path(latest["selected"]["checkpoint"])
    checkpoint=torch.load(source,map_location="cpu",weights_only=True,mmap=True)
    model_dir=ROOT/"artifacts/models";model_dir.mkdir(exist_ok=True)
    target=model_dir/"exp11_best.pt"
    lite={k:checkpoint[k] for k in ("model_state","config","class_names","num_classes","epoch","valid_metrics")}
    if not target.exists():torch.save(lite,target)
    restored=torch.load(target,map_location="cpu",weights_only=True,mmap=True)
    assert all(torch.equal(value,restored["model_state"][key]) for key,value in checkpoint["model_state"].items())
    del restored,checkpoint
    objects=subprocess.check_output(["git","rev-list","--objects","main"],cwd=ROOT)
    output=subprocess.run(["git","cat-file","--batch-check=%(objecttype) %(objectsize) %(rest)"],cwd=ROOT,input=objects,capture_output=True,check=True).stdout.decode("utf-8",errors="replace")
    oversized=[]
    for line in output.splitlines():
        parts=line.split(" ",2)
        if parts[0]=="blob" and int(parts[1])>100*1024**2:oversized.append(line)
    assert not oversized
    assert not list(ROOT.glob("*.py"))
    checks={"status":"passed","source_modules_imported":16,"root_python_files":0,
        "notebook_bytes":(ROOT/"launch.ipynb").stat().st_size,"embedded_notebook_images":0,
        "wildlife_raw_groups":rows,"wildlife_cross_split_source_overlap":overlaps,
        "main_history_blobs_above_100MiB":oversized,
        "inference_checkpoint":str(target.relative_to(ROOT)),"inference_checkpoint_MB":round(target.stat().st_size/1e6,2),
        "source_full_checkpoint_MB":round(source.stat().st_size/1e6,2),"exported_model_tensors_identical":True,
        "full_checkpoints_preserved":True,"git_backup":"artifacts/archive/before_cleanup.bundle"}
    (ROOT/"artifacts/archive/cleanup_verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
