"""Shared experiment setup/training/reporting. No work starts on import.

Models, Trainer, metrics, preparation: script.py. Data reading: animal_data.py.
LLM-assisted consolidation, 2026-11-09.
"""

from project_paths import PROJECT_ROOT
import argparse
from collections import Counter
from datetime import datetime
from functools import lru_cache
import gc
import json
from pathlib import Path
import random
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from animal_data import CLASS_NAMES, build_datasets
from script import Trainer, ResNetUNet, load_initial_animal_model


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("These experiments require CUDA")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    torch.backends.cudnn.benchmark = False
    cleanup_cuda()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=2)
def experiment_data(root, image_size=640, include_background=False):
    return build_datasets(Path(root), image_size, include_background=include_background)


def make_loader(data, batch_size, *, sampler=None, workers=2):
    return DataLoader(data, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def close_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def experiment_folder(root, number):
    folder = Path(root) / "artifacts/experiments" / f"exp{number}_results"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def note(folder, message):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with (folder / (folder.name.removesuffix("_results")+".md")).open("a", encoding="utf-8") as stream:
        stream.write(f"\n\n### {datetime.now():%Y-%m-%d %H:%M:%S}\n\n{message}\n")


def new_run(folder):
    path = Path(folder) / ("run_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_multiclass_trainer(root=None, *, device=None, image_size=640, batch_size=1,
        learning_rate=1e-4, seed=42, kernel_size=3, bn_policy="normal", amp=False,
        channels_last=False, workers=0, checkpoint_dir=None, checkpoint_name="best_antarctic_multiclass.pt",
        include_background=False, samples_per_epoch=None, log_every=100):
    root = Path(root) if root is not None else PROJECT_ROOT
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seed_everything(seed)
    datasets, summary = experiment_data(str(root.resolve()), image_size, include_background)
    classes = [r["sample_class"] for d in datasets["train"].datasets for r in d.records]
    frequency = Counter(classes)
    if set(range(1,len(CLASS_NAMES)))-set(frequency):
        raise ValueError("Missing foreground training classes")
    if include_background and set(frequency) != set(range(len(CLASS_NAMES))):
        raise ValueError("Expected background plus four foreground sampling groups")
    sampler = WeightedRandomSampler([1./frequency[c] for c in classes], samples_per_epoch or len(classes),
        replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = make_loader(datasets["train"], batch_size, sampler=sampler, workers=workers)
    valid = datasets["valid"]
    background_loader = None
    if include_background:
        from torch.utils.data import ConcatDataset
        valid = ConcatDataset([d for d in valid.datasets if d.root.name != "dataset_nature"])
        background = next(d for d in datasets["valid"].datasets if d.root.name == "dataset_nature")
        background_loader = make_loader(background, batch_size, workers=workers)
    valid_loader = make_loader(valid, batch_size, workers=workers)
    initial = root / "artifacts/runs/penguin_baseline/best.pt"
    model = load_initial_animal_model(initial,len(CLASS_NAMES),kernel_size,seed).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    metadata = {**summary,"seed":seed,"sampling":"equal foreground-class mass with replacement",
        "initial_checkpoint":str(initial.relative_to(root)),"new_head":True,
        "batch_size":batch_size,"learning_rate":learning_rate,"kernel_size":kernel_size}
    if include_background:
        metadata.update(sampling="equal mass over background/penguin/other_bird/seal/whale with replacement",
            sampling_group_probability={CLASS_NAMES[c]:1/len(frequency) for c in sorted(frequency)},
            samples_per_epoch=len(sampler),
            validation_selection="Original animal validation only; nature validation reported separately")
    trainer = Trainer(model,train_loader,valid_loader,torch.nn.CrossEntropyLoss(ignore_index=255),
        optimizer,device,checkpoint_dir or root/"artifacts/runs/antarctic_multiclass",class_names=CLASS_NAMES,
        checkpoint_name=checkpoint_name,dataset_metadata=metadata,amp=amp,bn_policy=bn_policy,
        channels_last=channels_last,gradient_clip=1. if amp else None,log_every=log_every,
        background_loader=background_loader)
    return trainer,datasets["train"],valid


def benchmark(root, *, kernel_size=3, layouts=(False,True), batches=(1,2,4,6,8,10)):
    """Real AMP backward/AdamW steps; cached images exclude disk/decode time."""
    from IPython.display import display
    import pandas as pd
    data,_ = experiment_data(str(Path(root).resolve()))
    classes = [r["sample_class"] for d in data["train"].datasets for r in d.records]
    groups = [[i for i,c in enumerate(classes) if c==target][:4] for target in range(1,5)]
    indices = [group[j] for j in range(4) for group in groups]
    samples = [data["train"][i] for i in indices]
    def one(batch,layout):
        torch.cuda.reset_peak_memory_stats()
        model = load_initial_animal_model(Path(root)/"artifacts/runs/penguin_baseline/best.pt",kernel_size=kernel_size).cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        dummy = make_loader(Subset(data["train"],indices),batch,workers=0)
        trainer = Trainer(model,dummy,dummy,torch.nn.CrossEntropyLoss(ignore_index=255),optimizer,
            "cuda:0",Path(root)/"artifacts/runs",class_names=CLASS_NAMES,amp=True,channels_last=layout,
            gradient_clip=1.,log_every=0)
        trainer.set_training_mode()
        images = torch.stack([samples[i%len(samples)][0] for i in range(batch)]).cuda()
        masks = torch.stack([samples[i%len(samples)][1] for i in range(batch)]).cuda()
        for _ in range(3):trainer.train_step(images,masks)
        torch.cuda.synchronize();start=perf_counter()
        for _ in range(6):trainer.train_step(images,masks)
        torch.cuda.synchronize()
        return dict(kernel_size=kernel_size,batch_size=batch,channels_last=layout,status="ok",
            images_per_second=6*batch/(perf_counter()-start),
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            parameters=sum(p.numel() for p in model.parameters()))
    rows=[]
    for layout in layouts:
        for batch in batches:
            cleanup_cuda()
            try:row=one(batch,layout)
            except torch.OutOfMemoryError:
                row=dict(kernel_size=kernel_size,batch_size=batch,channels_last=layout,status="OOM under 80% memory cap")
            cleanup_cuda();rows.append(row);print("Benchmark:",row,flush=True)
            if row["status"]!="ok":break
    good=[r for r in rows if r["status"]=="ok"]
    if not good:raise RuntimeError(f"No feasible batch for kernel={kernel_size}")
    fastest=max(r["images_per_second"] for r in good)
    selected=min([r for r in good if r["images_per_second"]>=.95*fastest],
                 key=lambda r:(r["batch_size"],-r["images_per_second"]))
    display(pd.DataFrame(rows).round(3))
    return dict(results=rows,selected=selected)


def save_training_visuals(trainer,valid_data,folder):
    """Full-validation curves plus identical seeded examples for every run."""
    from IPython.display import display
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import pandas as pd
    folder=Path(folder)
    checkpoint=torch.load(folder/trainer.checkpoint_name,map_location="cpu",weights_only=True,mmap=True)
    trainer.model.load_state_dict(checkpoint["model_state"]);trainer.model.eval()
    metrics,best_epoch=checkpoint["valid_metrics"],checkpoint["epoch"]
    display(pd.DataFrame({"class":CLASS_NAMES,**{n:metrics[n+"_per_class"]
        for n in ("iou","dice","precision","recall")}}).round(4))
    history=trainer.history;epochs=[r["epoch"] for r in history]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(epochs,[r["train_loss"] for r in history],"o-",label="train")
    axes[0].plot(epochs,[r["valid_loss"] for r in history],"o-",label="valid")
    axes[0].legend();axes[0].set(title="Loss",xlabel="Epoch")
    axes[1].plot(epochs,[r["foreground_miou"] for r in history],"o-")
    axes[1].axvline(best_epoch,linestyle="--",color="gray")
    axes[1].set(title="Validation foreground mIoU",xlabel="Epoch",ylim=(0,1))
    axes[2].bar(CLASS_NAMES,metrics["iou_per_class"])
    axes[2].set(title=f"Best epoch {best_epoch}: class IoU",ylim=(0,1))
    axes[2].tick_params(axis="x",rotation=35)
    fig.tight_layout();fig.savefig(folder/"curves.png",dpi=140,bbox_inches="tight")
    plt.show();plt.close(fig)
    classes=np.array([r["sample_class"] for d in valid_data.datasets for r in d.records])
    rng=np.random.default_rng(142)
    indices=[int(i) for c in range(1,5) for i in rng.choice(np.flatnonzero(classes==c),2,replace=False)]
    write_json(folder/"visual_indices.json",indices)
    palette=["black","red","orange","deepskyblue","limegreen"]
    for index in indices:
        image,target=valid_data[index]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=trainer.amp):
            prediction=trainer.model(image.unsqueeze(0).to(trainer.device)).argmax(1)[0].cpu().numpy()
        rgb=(image*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]).clamp(0,1).permute(1,2,0)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax in axes:ax.imshow(rgb);ax.axis("off")
        for ax,mask in zip(axes[1:],[target.numpy(),prediction]):
            visible=(mask>0)&(mask!=255)
            ax.imshow(np.where(visible,mask,0),cmap=ListedColormap(palette),vmin=0,vmax=4,alpha=visible.astype(float)*.5)
        for ax,title in zip(axes,[f"Valid index {index}","Annotation (255 ignored)","Prediction"]):ax.set_title(title)
        fig.legend(handles=[Patch(color=palette[c],label=CLASS_NAMES[c]) for c in range(1,5)],loc="lower center",ncol=4)
        fig.tight_layout(rect=(0,.07,1,1));fig.savefig(folder/f"valid_{index}.png",dpi=110,bbox_inches="tight")
        plt.show();plt.close(fig)
    return metrics,best_epoch


def train_run(root,folder,*,epochs=10,kernel_size=3,bn_policy="frozen_stats",batch_size=4,channels_last=False,seed=42,
              include_background=False,samples_per_epoch=None,log_every=100):
    folder=Path(folder)
    trainer,_,valid=build_multiclass_trainer(root,batch_size=batch_size,kernel_size=kernel_size,
        bn_policy=bn_policy,amp=True,channels_last=channels_last,workers=2,
        checkpoint_dir=folder,checkpoint_name="best.pt",seed=seed,include_background=include_background,
        samples_per_epoch=samples_per_epoch,log_every=log_every)
    trainer.config.update(epochs=epochs,seed=seed,learning_rate=1e-4,image_size=640,
                         initial_checkpoint="artifacts/runs/penguin_baseline/best.pt")
    if include_background:
        trainer.config.update(include_background=True,samples_per_epoch=len(trainer.train_loader.sampler))
    write_json(folder/"config.json",trainer.config);write_json(folder/"data.json",trainer.dataset_metadata)
    if (folder/"last.pt").exists():
        trainer.resume(folder/"last.pt");print("Resuming after epoch",trainer.epoch,flush=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.fit(max(0,epochs-trainer.epoch))
        peak=torch.cuda.max_memory_allocated()/2**30
        metrics,best_epoch=save_training_visuals(trainer,valid,folder)
        result=dict(status="complete",checkpoint=str((folder/"best.pt").resolve()),folder=str(folder.resolve()),
            best_epoch=best_epoch,metrics=metrics,config=trainer.config,completed_epochs=trainer.epoch,
            peak_allocated_GiB=peak,seconds=sum(r["epoch_seconds"] for r in trainer.history))
        write_json(folder/"result.json",result)
        return result
    finally:
        close_loader(trainer.train_loader);close_loader(trainer.valid_loader)
        if trainer.background_loader is not None:
            close_loader(trainer.background_loader)
        del trainer
        cleanup_cuda()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",action="store_true");parser.add_argument("--epochs",type=int,default=10)
    args=parser.parse_args();trainer,train,valid=build_multiclass_trainer()
    print("Train / valid:",len(train),len(valid))
    if args.train:trainer.fit(args.epochs)
    else:print("Setup only. No training performed.")


if __name__=="__main__":main()
