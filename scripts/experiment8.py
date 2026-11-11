"""Experiment 8: 3/5/7-kernel resource/pilot comparison, selected large kernel 10 epochs.

All former 3x3 encoder+decoder convolutions change; 7x7 stem and 1x1 layers stay.
Initial center-copy/zero-ring equivalence is tested separately. LLM-assisted 2026-11-09.
"""

from project_paths import PROJECT_ROOT
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
import torch
from torch.utils.data import Subset,WeightedRandomSampler
from train_multiclass import (experiment_folder,new_run,note,write_json,train_run,
    prepare_runtime,benchmark,experiment_data,build_multiclass_trainer,make_loader,
    close_loader,cleanup_cuda)


def kernel_pilot(root,kernel,batch,policy,layout,train_indices,valid_indices):
    trainer,train,valid=build_multiclass_trainer(root,kernel_size=kernel,batch_size=batch,
        bn_policy=policy,amp=True,channels_last=layout,workers=2)
    trainer.train_loader=make_loader(Subset(train,train_indices),batch)
    trainer.valid_loader=make_loader(Subset(valid,valid_indices),batch)
    trainer.epoch=1
    try:
        loss=trainer.train_one_epoch();_,_=trainer.validate()
        scores=trainer.validation_metrics
        return dict(kernel_size=kernel,train_loss=loss,valid_loss=scores['loss'],
                    foreground_miou=scores['foreground_miou'],iou_per_class=scores['iou_per_class'])
    finally:
        close_loader(trainer.train_loader);close_loader(trainer.valid_loader)
        del trainer;cleanup_cuda()


def run_experiment8(bn_result,*,root=None,epochs=10,run_dir=None):
    root=Path(root or PROJECT_ROOT).resolve();prepare_runtime()
    folder=experiment_folder(root,8);run=Path(run_dir) if run_dir else new_run(folder)
    policy=bn_result['selected']['config']['bn_policy']
    layout=bn_result['selected']['config']['channels_last']
    base_batch=bn_result['selected']['config']['batch_size']
    note(folder, "Run status updated. See structured run metadata.")
    resources={}
    for kernel in (3,5,7):
        print(f'\nEXPERIMENT 8 | resource test kernel={kernel}',flush=True)
        resources[kernel]=benchmark(root,kernel_size=kernel,layouts=(layout,),batches=(1,2,4,6))
        write_json(run/'resources.json',resources)
        table='| batch | status | images/s | peak allocated GiB | peak reserved GiB |\n|---:|---|---:|---:|---:|\n'
        for resource in resources[kernel]['results']:
            table+=f"| {resource['batch_size']} | {resource['status']} | {resource.get('images_per_second',float('nan')):.2f} | {resource.get('peak_allocated_GiB',float('nan')):.2f} | {resource.get('peak_reserved_GiB',float('nan')):.2f} |\n"
        note(folder, "Run status updated. See structured run metadata.")
    safe_batches={k:{r['batch_size'] for r in resource['results'] if r['status']=='ok' and r['batch_size']<=base_batch}
                  for k,resource in resources.items()}
    pilot_batch=max(set.intersection(*safe_batches.values()))
    planned_batches={k:max(values) for k,values in safe_batches.items()}
    data,_=experiment_data(str(root))
    classes=[r['sample_class'] for d in data['train'].datasets for r in d.records]
    freq=Counter(classes)
    sampler=WeightedRandomSampler([1./freq[c] for c in classes],1024,replacement=True,
                                  generator=torch.Generator().manual_seed(43))
    train_indices=list(sampler)
    valid_classes=np.array([r['sample_class'] for d in data['valid'].datasets for r in d.records])
    rng=np.random.default_rng(42);valid_indices=[]
    for c in range(1,5):
        indices=np.flatnonzero(valid_classes==c)
        valid_indices.extend(rng.choice(indices,min(16,len(indices)),replace=False).tolist())
    write_json(run/'pilot_indices.json',dict(train=train_indices,valid=valid_indices,batch_size=pilot_batch))
    pilots=[]
    for kernel in (3,5,7):
        print(f'\nEXPERIMENT 8 | matched pilot kernel={kernel}',flush=True)
        row=kernel_pilot(root,kernel,pilot_batch,policy,layout,train_indices,valid_indices)
        row['images_per_second']=next(r['images_per_second'] for r in resources[kernel]['results']
                                       if r['batch_size']==planned_batches[kernel] and r['status']=='ok')
        pilots.append(row);write_json(run/'pilots.json',pilots)
        note(folder, "Run status updated. See structured run metadata.")
    display(pd.DataFrame(pilots).round(4))
    larger=[r for r in pilots if r['kernel_size']>3]
    best_score=max(r['foreground_miou'] for r in larger)
    chosen=max([r for r in larger if r['foreground_miou']>=best_score-.01],key=lambda r:r['images_per_second'])
    kernel=chosen['kernel_size'];batch=planned_batches[kernel]
    note(folder, "Run status updated. See structured run metadata.")
    if batch!=base_batch:
        note(folder, "Run status updated. See structured run metadata.")
    result=train_run(root,run/f'kernel_{kernel}',epochs=epochs,kernel_size=kernel,
                     bn_policy=policy,batch_size=batch,channels_last=layout)
    summary=dict(experiment=8,folder=str(run),resources=resources,pilots=pilots,
                 selected=result,baseline=bn_result['selected'])
    write_json(run/'result.json',summary);write_json(folder/'latest.json',summary)
    delta=result['metrics']['foreground_miou']-bn_result['selected']['metrics']['foreground_miou']
    note(folder, "Run status updated. See structured run metadata.")
    return summary


def run_experiment8_single(bn_result,*,root=None,epochs=10,kernel_size=5,
                           preflight_path=None,run_dir=None):
    """Train exactly the requested larger kernel; do not start other candidates.

    Each requested kernel is isolated. Follow-up inference is called separately.
    LLM-assisted, updated 2026-11-10 for the requested 7x7 continuation.
    """
    if kernel_size not in (5,7):
        raise ValueError('Single larger-kernel experiment requires 5 or 7')
    root=Path(root or PROJECT_ROOT).resolve();prepare_runtime()
    folder=experiment_folder(root,8);run=Path(run_dir) if run_dir else new_run(folder)
    config=bn_result['selected']['config']
    policy,layout,base_batch=config['bn_policy'],config['channels_last'],config['batch_size']
    if preflight_path:
        preflight_path=Path(preflight_path)
        resources=json.loads(preflight_path.read_text(encoding='utf-8'))[str(kernel_size)]
    else:
        resources=benchmark(root,kernel_size=kernel_size,layouts=(layout,),batches=(1,2,4,6))
    safe=[r['batch_size'] for r in resources['results'] if r['status']=='ok' and r['batch_size']<=base_batch]
    batch=max(safe)
    write_json(run/'resource_test.json',resources)
    write_json(run/'request.json',dict(mode='single_requested_kernel',kernel_size=kernel_size,
        epochs=epochs,preflight_source=str(preflight_path) if preflight_path else None,
        scope='only the explicitly requested kernel; follow-up inference is a separate call'))
    note(folder, "Run status updated. See structured run metadata.")
    try:
        result=train_run(root,run/f'kernel_{kernel_size}',epochs=epochs,kernel_size=kernel_size,
            bn_policy=policy,batch_size=batch,channels_last=layout)
    except Exception as error:
        note(folder, "Run status updated. See structured run metadata.")
        raise
    summary=dict(experiment=8,mode='single_requested_kernel',folder=str(run),
                 selected=result,baseline=bn_result['selected'],resources={kernel_size:resources})
    write_json(run/'result.json',summary);write_json(folder/'latest.json',summary)
    write_json(folder/f'latest_kernel{kernel_size}.json',summary)
    delta=result['metrics']['foreground_miou']-bn_result['selected']['metrics']['foreground_miou']
    note(folder, "Run status updated. See structured run metadata.")
    return summary
