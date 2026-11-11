"""Fresh inference, labeled evaluation, ROI audits, and explicit pseudo-references.

All photo-level model-reference scores measure agreement, not ground-truth
accuracy. ROI labels are provisional visual-review interiors, not full masks.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import json
import gc
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import psutil
import torch
from torch.utils.data import DataLoader,ConcatDataset
from project_paths import PROJECT_ROOT as ROOT
from animal_data import CLASS_NAMES,build_datasets
from script import ResNetUNet,evaluate_segmentation,evaluate_background,metrics_from_confusion
from experiment9 import infer_branch,foreground_union

COLORS=np.array([[166,133,245],[255,65,77],[255,167,38],[0,190,220],[61,190,90]],np.uint8)
# Deliberately conservative interior rectangles; boundaries are not evaluated.
REGIONS={
 "penguins_near.jpg":[(1,[.42,.68,.55,.88]),(1,[.64,.66,.69,.76]),(1,[.44,.32,.52,.36]),(0,[.85,.40,.98,.60]),(0,[.04,.65,.16,.90])],
 "penguins_far.jpg":[(1,[.142,.815,.160,.872]),(1,[.354,.852,.375,.914]),(0,[.1,.1,.9,.3]),(0,[.79,.35,.87,.44]),(0,[.4,.82,.55,.95])],
 "antarctic_landscape_mather_lake.jpg":[(0,[.42,.4,.8,.75])],
 "antarctic_landscape_untitled_2.jpg":[(0,[.1,.1,.9,.75])],
}
FAR_POINTS=[(96,2194),(290,2210),(620,2200),(780,2210),(873,2220),(985,2220),
 (1055,2225),(1160,2235),(1207,2225),(1290,2235),(1390,2255),(1557,2250),
 (1645,2250),(1728,2255),(1830,2260),(1982,2260)]


def save_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,indent=2,allow_nan=True)+"\n",encoding="utf-8")


def sha(path):
 with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def save_csv(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with path.open("w",encoding="utf-8-sig",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def confusion(pred,target):
 valid=target!=255
 return np.bincount((target[valid].astype(np.int64)*5+pred[valid]).ravel(),minlength=25).reshape(5,5)


def scores(matrix):
 m=metrics_from_confusion(matrix);total=matrix.sum();bg=matrix[0].sum();fg=matrix[1:].sum()
 m["background_false_positive_fraction"]=float(matrix[0,1:].sum()/bg) if bg else None
 m["animal_class_confusion_fraction"]=float((matrix[1:,1:].sum()-np.trace(matrix[1:,1:]))/fg) if fg else None
 m["animal_miss_fraction"]=float(matrix[1:,0].sum()/fg) if fg else None
 m["agreement"]=float(np.trace(matrix)/total) if total else None
 return m


def class_rows(matrix,scope,method,image=""):
 result=[]
 for c,name in enumerate(CLASS_NAMES):
  tp=int(matrix[c,c]);fp=int(matrix[:,c].sum()-tp);fn=int(matrix[c].sum()-tp)
  ratio=lambda a,b:a/b if b else None
  result.append(dict(scope=scope,image=image,method=method,animal_class=name,tp=tp,fp=fp,fn=fn,
   precision=ratio(tp,tp+fp),false_discovery_fraction=ratio(fp,tp+fp),recall=ratio(tp,tp+fn),
   iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn),true_pixels=int(matrix[c].sum())))
 return result


def overlay(image,mask):
 labels=Image.fromarray(mask);labels.putpalette(COLORS.ravel().tolist()+[0]*(768-15))
 alpha=Image.fromarray(np.where(mask==0,90,150).astype(np.uint8))
 return Image.composite(labels.convert("RGB"),image,alpha)


def panel(image,masks,name,bands=False):
 width,height=420,280
 canvas=Image.new("RGB",(5*width,360),"white");d=ImageDraw.Draw(canvas)
 font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18)
 d.text((10,4),name+" | purple=background; red=penguin; orange=bird; cyan=seal; green=whale",font=font,fill="black")
 names=["Original","Native","Resize 640","Foreground union","Model-derived reference"]
 for i,(im,title) in enumerate(zip([image]+[overlay(image,m) for m in masks],names)):
  d.text((i*width+8,33),title,font=font,fill="black")
  im=im.copy();im.thumbnail((width-8,height));x=i*width+(width-im.width)//2;y=63+(height-im.height)//2
  canvas.paste(im,(x,y))
  if i==4 and bands:
   for f in (.5,.75):d.line((x,y+int(im.height*f),x+im.width,y+int(im.height*f)),fill="white",width=2)
 d.text((10,340),"Reference agreement is NOT ground-truth accuracy. ROI results use reviewed interiors only.",font=font,fill="#555555")
 return canvas


@torch.inference_mode()
def run_evaluation():
 torch.set_num_threads(4)
 if not torch.cuda.is_available():raise RuntimeError("CUDA is required for the finite validation rerun")
 torch.cuda.set_per_process_memory_fraction(.8,0)
 start=time.perf_counter()
 run=ROOT/"artifacts/evaluation"/datetime.now().strftime("%Y%m%d_%H%M%S");run.mkdir(parents=True)
 report=run/"report";report.mkdir(parents=True,exist_ok=True)
 result=json.loads((ROOT/"artifacts/experiments/exp11_results/latest.json").read_text())
 checkpoint=Path(result["selected"]["checkpoint"])
 ckpt=torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
 model=ResNetUNet(num_classes=5,pretrained=False,kernel_size=ckpt["config"]["kernel_size"])
 model.load_state_dict(ckpt["model_state"],strict=True);del ckpt
 model.cuda().eval()
 data,metadata=build_datasets(ROOT,include_background=True)
 animals=ConcatDataset([d for d in data["valid"].datasets if d.root.name!="dataset_nature"])
 background=next(d for d in data["valid"].datasets if d.root.name=="dataset_nature")
 batch=4 if torch.cuda.mem_get_info()[0]>3*2**30 else 1
 loader=DataLoader(animals,batch_size=batch,num_workers=0)
 print("Fresh labeled validation:",len(animals),"images; batch",batch,flush=True)
 val=evaluate_segmentation(model,loader,"cuda:0",criterion=torch.nn.CrossEntropyLoss(ignore_index=255),amp=True,progress_every=100)
 bg=evaluate_background(model,DataLoader(background,batch_size=batch,num_workers=0),"cuda:0",amp=True)
 save_json(run/"labeled_validation.json",dict(animal_validation=val,nature_validation=bg,dataset_metadata=metadata))
 labeled_rows=class_rows(np.array(val["confusion_matrix"]),"existing_labels_640","direct")
 photo_rows=[];roi_rows=[];records=[];selected_panels=[]
 suffixes={".jpg",".jpeg",".png",".webp"}
 paths=[p for folder in (ROOT/"data/examples",ROOT/"data/nju_testset") for p in sorted(folder.iterdir())
  if p.is_file() and p.suffix.lower() in suffixes
  and not any(marker in p.stem.lower() for marker in (".edit_", ".seed", ".mask"))]
 for number,path in enumerate(paths,1):
  with Image.open(path) as raw:image=ImageOps.exif_transpose(raw).convert("RGB")
  print(f"Fresh photo inference {number}/{len(paths)}: {path.name} {image.size}",flush=True)
  native=None;native_info=None
  try:native,native_info=infer_branch(image,model,"cuda:0")
  except torch.OutOfMemoryError:pass
  gc.collect();torch.cuda.empty_cache()
  if native is None:
   model.cpu();gc.collect();torch.cuda.empty_cache()
   required=image.width*image.height*350+2*2**30
   if required>psutil.virtual_memory().available*.9:raise MemoryError(f"Insufficient CPU RAM for honest native inference: {path.name}")
   native,native_info=infer_branch(image,model,"cpu")
  scaled,resize_info=infer_branch(image,model,"cuda:0",resize=True)
  a,b,u,fusion=foreground_union(native,scaled)
  confidence={"native":native.max(0),"resize":scaled.max(0)}
  ref=b.copy()
  if path.name=="penguins_far.jpg":
   y1,y2=int(image.height*.5),int(image.height*.75);ref[y1:y2]=a[y1:y2]
  masks={"native":a,"resize":b,"union":u,"reference":ref}
  is_private=path.parent.name=="nju_testset"
  output=(ROOT/"data/nju_testset/predictions"/run.name if is_private else run/"masks")/path.stem
  output.mkdir(parents=True,exist_ok=True)
  target=np.full((image.height,image.width),255,np.uint8)
  for label,box in REGIONS.get(path.name,[]):
   x0,y0,x1,y1=[int(v*s) for v,s in zip(box,[image.width,image.height,image.width,image.height])]
   target[y0:y1,x0:x1]=label
  for method,mask in masks.items():
   Image.fromarray(mask).save(output/(method+".png"))
   histogram=np.bincount(mask.ravel(),minlength=5)/mask.size
   agreement=scores(confusion(mask,ref))
   row=dict(image=path.name,method=method,scope="same_model_reference_agreement",reference_kind="resize_native_resize_bands" if path.name=="penguins_far.jpg" else "resize_baseline",
    pixel_agreement=agreement["agreement"],foreground_iou_vs_reference=agreement["foreground_miou"],
    **{name+"_area_fraction":float(histogram[c]) for c,name in enumerate(CLASS_NAMES)})
   if method in confidence:
    animal=mask>0; conf=confidence[method]
    row["mean_confidence_on_predicted_animals"]=float(conf[animal].mean()) if animal.any() else None
    row["high_confidence_animal_area"]=float((animal&(conf>=.9)).mean())
   if (target!=255).any():
    matrix=confusion(mask,target);s=scores(matrix)
    row.update(reviewed_roi_background_fp=s["background_false_positive_fraction"],reviewed_roi_animal_confusion=s["animal_class_confusion_fraction"],reviewed_roi_animal_miss=s["animal_miss_fraction"])
    roi_rows.extend(class_rows(matrix,"provisional_rgb_reviewed_roi",method,path.name))
   if path.name=="penguins_far.jpg":
    values=np.array([mask[y,x] for x,y in FAR_POINTS])
    row["distant_point_penguin_accuracy"]=float((values==1).mean())
    row["distant_point_animal_detection"]=float((values>0).mean())
    row["distant_point_count"]=len(values)
   photo_rows.append(row)
  preview=panel(image,[a,b,u,ref],path.name,bands=path.name=="penguins_far.jpg")
  preview.save(output/"preview.jpg",quality=85)
  if path.name in ("penguins_far.jpg","penguins_near.jpg","antarctic_landscape_untitled_2.jpg"):selected_panels.append((path.name,preview))
  records.append(dict(image=path.name,private=is_private,source_sha256=sha(path),output=str(output),native=native_info,resize=resize_info,fusion=fusion))
  save_json(run/"progress.json",dict(records=records,photos=photo_rows))
  del native,scaled,confidence,a,b,u,ref,masks,image,target;gc.collect();torch.cuda.empty_cache()
 save_csv(report/"metrics.csv",labeled_rows+roi_rows)
 save_csv(report/"photo_metrics.csv",photo_rows)
 save_csv(run/"photo_metrics.csv",photo_rows)
 with (report/"confusion.csv").open("w",encoding="utf-8-sig",newline="") as stream:
  writer=csv.writer(stream);writer.writerow(["true / predicted"]+list(CLASS_NAMES))
  for name,row in zip(CLASS_NAMES,val["confusion_matrix"]):writer.writerow([name]+row)
 selected_panels.sort(key=lambda item:{"penguins_far.jpg":0,"penguins_near.jpg":1}.get(item[0],2))
 overview=Image.new("RGB",(2100,360*len(selected_panels)),"white")
 for i,(_,im) in enumerate(selected_panels):overview.paste(im,(0,i*360))
 overview.save(report/"preview.jpg",quality=88)
 config=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),best_epoch=result["selected"]["best_epoch"],
  run=str(run),fresh_inference=True,training_performed=False,reference_bands={"penguins_far.jpg":[[0,.5,"resize"],[.5,.75,"native"],[.75,1,"resize"]]},
  other_photo_reference="resize baseline; not a claimed ideal segmentation",roi_labels=REGIONS,
  roi_status="Assistant-reviewed RGB interiors, provisional, not independently human-confirmed full-image ground truth",
  far_point_labels={"original_size":[6000,4000],"class_id":1,"points":FAR_POINTS,"scope":"Visible distant body centers in left third, selected from RGB before fresh predictions"},
  metrics_scopes=["existing_labels_640","provisional_rgb_reviewed_roi","same_model_reference_agreement"],
  note="Never interpret model-reference agreement as independent accuracy. Historical model was trained before source-group deduplication.")
 save_json(report/"config.json",config)
 summary=dict(status="complete",config=config,validation=val,background=bg,records=records,seconds=time.perf_counter()-start)
 save_json(run/"result.json",summary);save_json(ROOT/"artifacts/evaluation/latest.json",summary)
 lines=["# Experiment 11: fresh error audit","",
  "Fresh inference with the best epoch-3 checkpoint. No training. The CSV scopes distinguish existing-label accuracy, provisional RGB-reviewed interiors, and same-model reference agreement.","",
  "The far-photo reference uses resize above 50%, native from 50-75%, and resize below 75%. Other photo references use resize as a baseline, not an ideal mask. The reference is derived from the same model and cannot establish real accuracy.","",
  "| Class | Precision | Recall | IoU | False discovery |","|---|---:|---:|---:|---:|"]
 for row in labeled_rows[1:]:lines.append(f"| {row['animal_class']} | {row['precision']:.3%} | {row['recall']:.3%} | {row['iou']:.3%} | {row['false_discovery_fraction']:.3%} |")
 lines += ["",f"Labeled animal validation: {len(animals)} images. Foreground mIoU: {val['foreground_miou']:.5f}.",
  f"Nature validation: {len(background)} images; false animal pixels on labeled background: {bg['false_positive_fraction']:.5%}.","",
  "![Photo comparisons](preview.jpg)","",
  "| Photo / method | Reviewed BG FP | Reviewed animal confusion | Distant penguin points | Reference agreement |","|---|---:|---:|---:|---:|"]
 for row in photo_rows:
  if row["image"] not in REGIONS:continue
  fmt=lambda v:"-" if v is None else f"{v:.2%}"
  lines.append(f"| {row['image']} / {row['method']} | {fmt(row.get('reviewed_roi_background_fp'))} | {fmt(row.get('reviewed_roi_animal_confusion'))} | {fmt(row.get('distant_point_penguin_accuracy'))} | {fmt(row['pixel_agreement'])} |")
 lines += ["","Reviewed BG FP = reviewed background pixels predicted as any animal / all reviewed background pixels. Reviewed animal confusion = reviewed animal pixels predicted as another animal class / all reviewed animal pixels. Distant-point accuracy covers the 16 selected body centers; it is not instance recall.","",
  "ROI labels and distant points are a provisional visual audit, not full-image boundary annotations. Existing penguin labels can miss distant birds, and recovered SealID masks are approximate. Model-reference metrics are biased toward their contributing branches. Course images and derived masks remain in ignored data storage.","",
  "metrics.csv contains TP/FP/FN, precision, false discovery, recall, IoU, and Dice. photo_metrics.csv contains prediction areas, confidence, ROI errors, distant-point accuracy, and explicitly labeled reference agreement. config.json preserves the exact review regions and compositing rule."]
 (report/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print("COMPLETE",run,"seconds",summary["seconds"],flush=True)
 return summary


if __name__=="__main__":
 run_evaluation()
