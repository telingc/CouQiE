"""Experiment 13: independent four-class, ImageNet-encoder baseline.

No work on import. Default calls print a plan only. Historical five-class
datasets/models and benchmark_v2 are never modified. LLM-assisted 2026-11-11.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
import csv
import gc
import hashlib
import json
import math
import random
import shutil

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import psutil
import torch
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler

from project_paths import PROJECT_ROOT
from animal_data import build_datasets, fit_image_and_mask
from script import ResNetUNet, Trainer, evaluate_segmentation, metrics_from_confusion
from experiment9 import infer_branch
from gallery import write_gallery_html

CLASSES = ('background', 'penguin', 'other_bird', 'seal')
BENCHMARK = 'benchmark_v2'
PROTOCOL = 'exp13_four_class_imagenet_v1'
BASE_CONFIG = dict(protocol=PROTOCOL, class_names=list(CLASSES), kernel_size=3,
    num_classes=4, epochs=10, image_size=640, requested_batch_size=6, amp=True,
    bn_policy='normal', optimizer='AdamW', learning_rate=1e-4, weight_decay=.01,
    seed=42, ignore_index=255, gradient_clip=None, learning_rate_schedule='constant',
    initialization='ImageNet ResNet34 encoder; random decoder/head; no historical weights',
    sampling='per-image 1/sqrt(group_size), replacement=True, draws=len(train)',
    group_rule='largest converted foreground area; background if no target foreground',
    workers=2, label_mapping={'4':0, '255':255}, benchmark_version=BENCHMARK)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def save_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows(clean(rows))


def convert_labels(values):
    """Work on a copy: whale=background; unknown/void/padding stay ignored."""
    if not np.isin(values,[0,1,2,3,4,255]).all():
        raise ValueError('Unexpected source label; do not silently call it background')
    result=np.array(values,dtype=np.uint8,copy=True)
    result[result==4]=0
    return result


def sampling_group(counts):
    return int(np.argmax(counts[1:4])+1) if sum(counts[1:4]) else 0


class FourClassView(Dataset):
    def __init__(self, base, records):
        self.base,self.records=base,records
        self.groups=np.asarray([r['group'] for r in records],dtype=np.int64)
    def __len__(self): return len(self.base)
    def __getitem__(self,index):
        image,target=self.base[index]
        target=target.clone();target[target==4]=0
        return image,target


def audited_data(root, run):
    """Audit all current deduplicated splits; cache grouping, not source labels."""
    datasets,metadata=build_datasets(root,640,include_background=True)
    cache=run/'data_audit.json';record_path=run/'data_records.json'
    if cache.exists():
        audit=json.loads(cache.read_text(encoding='utf-8'))
        if audit['source_metadata']!=metadata:
            raise ValueError('Dataset view changed: use a new run directory')
        if sha(record_path)!=audit['records_sha256']:
            raise ValueError('Cached data audit changed')
        records=json.loads(record_path.read_text(encoding='utf-8'))
        for split in records:
            actual=[child.image_path(i).relative_to(root).as_posix()
                    for child in datasets[split].datasets for i in range(len(child))]
            if actual!=[r['image'] for r in records[split]]:
                raise ValueError('Audited dataset order changed; use a new run')
            for row in records[split]:
                if sha(root/row['image'])!=row['image_sha256']:
                    raise ValueError('An audited image changed: '+row['image'])
                if row.get('prepared_mask') and sha(root/row['prepared_mask'])!=row['mask_sha256']:
                    raise ValueError('An audited mask changed')
    else:
        records={};statistics=[];transition=Counter()
        for split,base in datasets.items():
            records[split]=[]
            for child in base.datasets:
                pixels=np.zeros(256,dtype=np.int64);before=np.zeros(256,dtype=np.int64)
                groups=Counter()
                for local,record in enumerate(child.records):
                    image,mask=child.get_raw(local)
                    canvas,target=fit_image_and_mask(image,mask,640)
                    source=np.asarray(target)
                    converted=convert_labels(source)
                    counts=np.bincount(converted.ravel(),minlength=256)
                    original=np.bincount(source.ravel(),minlength=256)
                    if counts[:4].sum()==0: raise ValueError('Fully ignored image in '+str(child.image_path(local)))
                    assert counts[255]==original[255] and counts[0]==original[0]+original[4]
                    assert np.array_equal(counts[1:4],original[1:4])
                    group=sampling_group(counts);groups[group]+=1;pixels+=counts;before+=original
                    transition[(split,record['sample_class'],group)]+=1
                    path=child.image_path(local)
                    row=dict(source=child.root.name,image=path.relative_to(root).as_posix(),
                        image_sha256=sha(path),old_sample_class=record['sample_class'],group=group,
                        pixel_counts={str(c):int(counts[c]) for c in (0,1,2,3,255)})
                    if child.prepared:
                        p=child.split_dir/record['mask']
                        row.update(prepared_mask=p.relative_to(root).as_posix(),mask_sha256=sha(p))
                    records[split].append(row)
                    image.close();mask.close();canvas.close();target.close()
                statistics.append(dict(split=split,source=child.root.name,images=len(child),
                    **{name+'_images':groups[c] for c,name in enumerate(CLASSES)},
                    **{name+'_pixels':int(pixels[c]) for c,name in enumerate(CLASSES)},
                    ignored_pixels=int(pixels[255]),whale_pixels_mapped_to_background=int(before[4])))
                print('Data audit:',split,child.root.name,len(child),flush=True)
            assert len(records[split])==len(base)
        frequencies=Counter(r['group'] for r in records['train'])
        if set(frequencies)!=set(range(4)): raise ValueError('All four sampling groups are required')
        total=sum(math.sqrt(n) for n in frequencies.values())
        save_json(record_path,records)
        audit=dict(protocol=PROTOCOL,class_names=list(CLASSES),allowed_labels=[0,1,2,3,255],source_metadata=metadata,
            statistics=statistics,records_sha256=sha(record_path),
            sampling=[dict(group=CLASSES[c],images=frequencies[c],image_weight=1/math.sqrt(frequencies[c]),
                expected_group_probability=math.sqrt(frequencies[c])/total) for c in range(4)],
            regrouping=[dict(split=s,old_group=o,new_group=n,images=count) for (s,o,n),count in transition.items()],
            notes='Groups use foreground dominance after 4->0 at working resolution. Original files/splits and five-class globals are unchanged.')
        save_json(cache,audit);save_csv(run/'data_statistics.csv',statistics)
    views={split:FourClassView(datasets[split],records[split]) for split in datasets}
    animal_indices=[i for i,r in enumerate(records['valid']) if r['source']!='dataset_nature']
    nature_indices=[i for i,r in enumerate(records['valid']) if r['source']=='dataset_nature']
    return views['train'],Subset(views['valid'],animal_indices),Subset(views['valid'],nature_indices),audit


class RecordedSampler(WeightedRandomSampler):
    def __init__(self, groups):
        self.groups=np.asarray(groups);counts=np.bincount(self.groups,minlength=4)
        super().__init__([1/math.sqrt(counts[g]) for g in self.groups],len(groups),
                         replacement=True,generator=torch.Generator().manual_seed(42))
        self.last_counts=None
    def __iter__(self):
        indices=list(super().__iter__())
        self.last_counts=np.bincount(self.groups[indices],minlength=4).tolist()
        return iter(indices)


def background_errors(matrix):
    matrix=np.asarray(matrix,dtype=np.int64);denominator=int(matrix[0].sum())
    return dict(background_pixels=denominator,background_false_positive_pixels=int(matrix[0,1:].sum()),
        background_fp_fraction=float(matrix[0,1:].sum()/denominator) if denominator else None,
        **{name+'_on_background':float(matrix[0,c]/denominator) if denominator else None
           for c,name in enumerate(CLASSES) if c})


def flatten_metrics(metrics):
    row=dict(loss=metrics['loss'],foreground_miou=metrics['foreground_miou'],
             **background_errors(metrics['confusion_matrix']))
    for c,name in enumerate(CLASSES[1:],1):
        for key in ('iou','precision','recall','dice'):
            row[name+'_'+key]=metrics[key+'_per_class'][c]
    return row


class BaselineTrainer(Trainer):
    def validate(self):
        metrics=evaluate_segmentation(self.model,self.valid_loader,self.device,criterion=self.criterion,
                                      num_classes=4,amp=self.amp)
        nature=evaluate_segmentation(self.model,self.nature_loader,self.device,criterion=self.criterion,
                                     num_classes=4,amp=self.amp)
        metrics.update(background_errors(metrics['confusion_matrix']))
        nature.update(background_errors(nature['confusion_matrix']))
        metrics['nature_validation']=nature
        self.validation_metrics=clean(metrics)
        return metrics['loss'],metrics['foreground_miou']
    def fit(self,num_epochs):
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        for _ in range(num_epochs):
            self.epoch+=1;torch.cuda.reset_peak_memory_stats();started=perf_counter()
            train_loss=self.train_one_epoch();train_seconds=perf_counter()-started
            valid_loss,score=self.validate();torch.cuda.synchronize()
            counts=self.train_loader.sampler.last_counts
            assert sum(counts)==len(self.train_loader.dataset)
            row=dict(epoch=self.epoch,train_loss=train_loss,valid_loss=valid_loss,foreground_miou=score,
                valid_metrics=self.validation_metrics,iou_per_class=self.validation_metrics['iou_per_class'],
                train_seconds=train_seconds,epoch_seconds=perf_counter()-started,batch_size=self.train_loader.batch_size,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                actual_sampling_counts=dict(zip(CLASSES,counts)))
            self.history.append(row)
            save_json(self.checkpoint_dir/'history.json',self.history)
            save_csv(self.checkpoint_dir/'epochs.csv',epoch_rows(self.history))
            print(f"EPOCH {self.epoch:02d} | train={train_loss:.4f} | valid={valid_loss:.4f} | "
                  f"foreground_mIoU={score:.4f} | background_FP={self.validation_metrics['background_fp_fraction']:.3%} | "
                  f"nature_FP={self.validation_metrics['nature_validation']['background_fp_fraction']:.3%}",flush=True)
            self.save_checkpoint(valid_loss,score)
            self._write_checkpoint(self.checkpoint_dir/'last.pt')
        return self.history


def epoch_rows(history):
    return [dict(epoch=r['epoch'],train_loss=r['train_loss'],batch_size=r['batch_size'],
        epoch_seconds=r['epoch_seconds'],peak_allocated_GiB=r['peak_allocated_GiB'],
        peak_reserved_GiB=r['peak_reserved_GiB'],**flatten_metrics(r['valid_metrics']),
        **{'nature_'+k:v for k,v in flatten_metrics(r['valid_metrics']['nature_validation']).items()},
        **{'sampled_'+k:v for k,v in r['actual_sampling_counts'].items()}) for r in history]


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def cached_imagenet():
    path=Path(torch.hub.get_dir())/'checkpoints/resnet34-b627a593.pth'
    if not path.exists(): raise FileNotFoundError('ImageNet weights are not cached; no historical fallback is allowed: '+str(path))
    digest=sha(path)
    if not digest.startswith('b627a593'): raise ValueError('ImageNet checkpoint hash mismatch')
    return path,digest


def encoder_key(name):
    if name.startswith('stem.0.'): return name.replace('stem.0.','conv1.',1)
    if name.startswith('stem.1.'): return name.replace('stem.1.','bn1.',1)
    if name.startswith(('enc1.','enc2.','enc3.','enc4.')): return 'layer'+name[3:]
    return None


def imagenet_value(weights,key,template):
    # Some official torchvision checkpoints predate BN's batch counter buffer.
    if key.endswith('num_batches_tracked') and key not in weights:
        return torch.zeros_like(template)
    return weights[key]


def new_model():
    # Explicit cached ImageNet copy: no historical loader and no network fallback.
    seed();path,_=cached_imagenet()
    model=ResNetUNet(num_classes=4,pretrained=False,kernel_size=3)
    weights=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    state=model.state_dict()
    for name in state:
        key=encoder_key(name)
        if key is not None: state[name]=imagenet_value(weights,key,state[name])
    model.load_state_dict(state,strict=True)
    return model


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def release():
    gc.collect();torch.cuda.empty_cache()


def preflight(train,run):
    """Two real AMP forward/backward/AdamW steps, discarded before training."""
    weights_path,weights_sha=cached_imagenet();attempts=[]
    indices=[int(np.flatnonzero(train.groups==c)[0]) for c in range(4)]
    indices += [int(np.flatnonzero(train.groups==c)[-1]) for c in (1,2)]
    samples=[train[i] for i in indices]
    for batch in (6,4):
        model=optimizer=scaler=images=targets=logits=loss=None
        try:
            model=new_model();initial_hash=state_hash(model)
            expected=torch.load(weights_path,map_location='cpu',weights_only=True,mmap=True)
            for name,value in model.state_dict().items():
                key=encoder_key(name)
                if key is not None and not torch.equal(value,imagenet_value(expected,key,value)):
                    raise ValueError('Encoder initialization mismatch: '+name)
            del expected
            model=model.cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=1024)
            images=torch.stack([x[0] for x in samples[:batch]]).cuda()
            targets=torch.stack([x[1] for x in samples[:batch]]).cuda()
            torch.cuda.reset_peak_memory_stats();losses=[]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(images);loss=torch.nn.functional.cross_entropy(logits,targets,ignore_index=255)
                if logits.shape!=(batch,4,640,640) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise RuntimeError('Invalid preflight logits/loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Non-finite or missing gradients')
                scaler.step(optimizer);scaler.update();losses.append(float(loss))
            torch.cuda.synchronize()
            attempts.append(dict(batch_size=batch,status='passed',losses=losses,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            result=dict(batch_size=batch,attempts=attempts,initial_state_sha256=initial_hash,
                imagenet_checkpoint=str(weights_path),imagenet_sha256=weights_sha,
                encoder_matches_imagenet=True,preflight_updates_reused=False)
            save_json(run/'preflight.json',result)
            return result
        except torch.OutOfMemoryError:
            attempts.append(dict(batch_size=batch,status='OOM'))
            save_json(run/'preflight_attempts.json',attempts)
            if batch==4: raise
        finally:
            del model,optimizer,scaler,images,targets,logits,loss
            release()


def _loader(data,batch,*,sampler=None):
    return DataLoader(data,batch_size=batch,sampler=sampler,shuffle=False,num_workers=2,
                      pin_memory=True,persistent_workers=True)


def _close(loader):
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None: iterator._shutdown_workers();loader._iterator=None


def train_baseline(root,run):
    if (run/'training_result.json').exists():
        return json.loads((run/'training_result.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this AMP baseline')
    torch.cuda.set_per_process_memory_fraction(.8,0);torch.backends.cudnn.benchmark=False
    train,animals,nature,audit=audited_data(root,run)
    probe=json.loads((run/'preflight.json').read_text()) if (run/'preflight.json').exists() else preflight(train,run)
    batch=4 if (run/'training_oom.json').exists() else probe['batch_size']
    while True:
        model=optimizer=trainer=None;loaders=[]
        try:
            model=new_model()
            if state_hash(model)!=probe['initial_state_sha256']: raise ValueError('Formal model differs from preflight initializer')
            model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
            sampler=RecordedSampler(train.groups)
            train_loader=_loader(train,batch,sampler=sampler);loaders.append(train_loader)
            valid_loader=_loader(animals,batch);loaders.append(valid_loader)
            nature_loader=_loader(nature,batch);loaders.append(nature_loader)
            trainer=BaselineTrainer(model,train_loader,valid_loader,torch.nn.CrossEntropyLoss(ignore_index=255),
                optimizer,'cuda:0',run,class_names=CLASSES,amp=True,bn_policy='normal',
                gradient_clip=None,log_every=0,dataset_metadata=audit)
            trainer.nature_loader=nature_loader
            trainer.config.update(BASE_CONFIG,batch_size=batch,imagenet_sha256=probe['imagenet_sha256'])
            if (run/'last.pt').exists():
                checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=True,mmap=True)
                previous_batch=checkpoint['config']['batch_size'];del checkpoint
                if previous_batch!=batch and not(previous_batch==6 and batch==4 and (run/'training_oom.json').exists()):
                    raise ValueError('Unexpected batch-size change')
                trainer.config['batch_size']=previous_batch
                trainer.resume(run/'last.pt');trainer.config['batch_size']=batch
            save_json(run/'config.json',trainer.config)
            trainer.fit(max(0,10-trainer.epoch))
            best=torch.load(run/'best.pt',map_location='cpu',weights_only=True,mmap=True)
            result=dict(protocol=PROTOCOL,status='complete',folder=str(run),checkpoint=str(run/'best.pt'),
                last_checkpoint=str(run/'last.pt'),best_epoch=best['epoch'],completed_epochs=trainer.epoch,
                best_metrics=best['valid_metrics'],last_metrics=trainer.history[-1]['valid_metrics'],
                last_minus_best_foreground_miou=trainer.history[-1]['foreground_miou']-best['valid_score'],
                config=trainer.config,initialization=probe)
            del best
            save_json(run/'training_result.json',result)
            save_csv(run/'best_last_metrics.csv',[dict(checkpoint=name,epoch=e,**flatten_metrics(m))
                for name,e,m in [('best',result['best_epoch'],result['best_metrics']),('last',10,result['last_metrics'])]])
            return result
        except torch.OutOfMemoryError:
            if batch==4: raise
            save_json(run/'training_oom.json',dict(reason='CUDA OOM at batch 6; discard incomplete epoch and resume last completed epoch at batch 4',time=datetime.now().isoformat()))
            batch=4
        finally:
            for loader in loaders:_close(loader)
            del trainer,model,optimizer
            release()


def confusion4(pred,target):
    matrix=np.zeros((4,4),dtype=np.int64)
    if pred.shape!=target.shape or not np.isin(pred,[0,1,2,3]).all(): raise ValueError('Prediction labels/shape invalid')
    for y in range(0,len(target),128):
        t=target[y:y+128];p=pred[y:y+128];valid=t!=255
        codes=t[valid].astype(np.int64)*4+p[valid]
        matrix+=np.bincount(codes,minlength=16).reshape(4,4)
    return matrix


def photo_metrics(matrix):
    metrics=clean(metrics_from_confusion(matrix));row=background_errors(matrix)
    for c,name in enumerate(CLASSES):
        row[name+'_target_pixels']=int(matrix[c].sum())
        for metric in ('iou','precision','recall','dice'):row[name+'_'+metric]=metrics[metric+'_per_class'][c]
    row['note']='Reference agreement; bird/seal recall is undefined without positive labels.'
    return row


def thumbnail(image,mask=None):
    from PIL import ImageFont
    rgb=image.copy();rgb.thumbnail((480,320))
    if mask is not None:
        small=np.asarray(Image.fromarray(mask).resize(rgb.size,Image.Resampling.NEAREST))
        palette=np.array([[0,0,0],[255,65,77],[255,167,38],[0,190,220]],np.uint8)
        arr=np.asarray(rgb).copy();visible=(small>0)&(small<4)
        arr[visible]=(.5*arr[visible]+.5*palette[small[visible]]).astype(np.uint8)
        rgb=Image.fromarray(arr)
    return rgb


def panel(image,masks,title):
    canvas=Image.new('RGB',(1440,365),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((8,5),title,font=font,fill='black')
    for i,(label,mask) in enumerate(zip(('Original','Native','Resize 640'),(None,*masks))):
        draw.text((i*480+8,31),label,font=font,fill='black');small=thumbnail(image,mask)
        canvas.paste(small,(i*480+(480-small.width)//2,57+(300-small.height)//2))
    return canvas


def load_references(root):
    folder=root/'data/reference_masks'/BENCHMARK
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['version']!=BENCHMARK or len(manifest['entries'])!=9:
        raise ValueError('Expected the existing frozen benchmark_v2 with nine images')
    for entry in manifest['entries']:
        if sha(root/entry['image_path'])!=entry['image_sha256'] or sha(folder/entry['mask'])!=entry['mask_sha256']:
            raise ValueError('Frozen development references changed')
    return folder,manifest


def rescore_history(root,run,folder,manifest):
    """CPU-only remapping of verified cached predictions; no historical networks."""
    cache=root/'artifacts/reference_benchmark'/BENCHMARK/'result.json'
    if not cache.exists():return
    result=json.loads(cache.read_text(encoding='utf-8'));rows=[]
    entries={e['image']:e for e in manifest['entries']}
    for model in result['models']:
        name=model['spec']['id']
        if name not in ('exp8_kernel5','exp11_best'):continue
        pooled={}
        for record in model['records']:
            entry=entries[record['image']]
            if record['image_sha256']!=entry['image_sha256']:raise ValueError('Historical source mismatch')
            with Image.open(folder/entry['mask']) as raw:target=convert_labels(np.asarray(raw))
            for method in ('native','resize'):
                artifact=record['masks'][method];path=root/artifact['path']
                if sha(path)!=artifact['sha256']:raise ValueError('Historical prediction cache changed')
                with Image.open(path) as raw:pred=convert_labels(np.asarray(raw))
                matrix=confusion4(pred,target)
                rows.append(dict(variant=name,image=entry['image'],method=method,group=entry['group'],
                    scope='image',whale_remapped_to_background=True,**photo_metrics(matrix)))
                key=(method,entry['group']);pooled[key]=pooled.get(key,np.zeros((4,4),np.int64))+matrix
        for (method,group),matrix in pooled.items():
            rows.append(dict(variant=name,image='',method=method,group=group,scope='pooled',
                whale_remapped_to_background=True,**photo_metrics(matrix)))
    save_csv(run/'historical_reference_metrics.csv',rows)


@torch.inference_mode()
def evaluate_best(root,run):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for resize inference')
    torch.cuda.set_per_process_memory_fraction(.8,0)
    weights=run/'best.pt';checkpoint=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
    if tuple(checkpoint['class_names'])!=CLASSES or checkpoint['config'].get('protocol')!=PROTOCOL:
        raise ValueError('Expected this experiment four-class checkpoint')
    model=ResNetUNet(num_classes=4,pretrained=False,kernel_size=3)
    model.load_state_dict(checkpoint['model_state']);del checkpoint
    ref_folder,manifest=load_references(root)
    save_json(run/'reference_manifest_snapshot.json',manifest)
    from evaluate_project import FAR_POINTS
    if len(FAR_POINTS)!=16:raise ValueError('Expected the existing 16-point far-penguin protocol')
    save_json(run/'far_point_protocol.json',dict(coordinates=FAR_POINTS,hit_rule='exact pixel; no radius',
        source_code_sha256=sha(root/'scripts/evaluate_project.py')))
    entries={e['image']:e for e in manifest['entries']}
    courses=sorted(p for p in (root/'data/nju_testset').iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    if len(courses)!=6:raise ValueError('Expected six NJU source images')
    paths=[root/e['image_path'] for e in manifest['entries']]+courses
    records=[];rows=[];points=[];pooled={};gallery=[];panels={}
    for path in paths:
        private=path.parent.name=='nju_testset';entry=None if private else entries[path.name]
        with Image.open(path) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
        output=run/'predictions'/('nju' if private else 'references')/path.stem
        output.mkdir(parents=True,exist_ok=True)
        masks=[];details={}
        for method in ('native','resize'):
            target_path=output/(method+'.png');info_path=output/(method+'.json')
            cached=json.loads(info_path.read_text()) if info_path.exists() else None
            if cached and cached['checkpoint_sha256']==sha(weights) and cached['image_sha256']==sha(path) and target_path.exists() and sha(target_path)==cached['mask_sha256']:
                with Image.open(target_path) as raw:prediction=np.array(raw,dtype=np.uint8)
                info=cached['inference']
            else:
                print('Inference:',path.name,method,image.size,flush=True)
                oom=False
                try:probability,info=infer_branch(image,model,'cuda:0',resize=method=='resize')
                except torch.OutOfMemoryError:
                    if method!='native':raise
                    oom=True
                if oom:
                    model.cpu();release()
                    if image.width*image.height*350+2*2**30>psutil.virtual_memory().available*.9:
                        raise MemoryError('Insufficient CPU RAM for native inference; no hidden resize is permitted')
                    probability,info=infer_branch(image,model,'cpu')
                    info['fallback_reason']='native GPU OOM; original pixels preserved'
                prediction=probability.argmax(0).astype(np.uint8);del probability
                Image.fromarray(prediction).save(target_path)
                save_json(info_path,dict(checkpoint_sha256=sha(weights),image_sha256=sha(path),
                    mask_sha256=sha(target_path),inference=info))
            if prediction.shape!=(image.height,image.width):raise ValueError('Prediction shape mismatch')
            masks.append(prediction);details[method]=info
            if entry is not None:
                with Image.open(ref_folder/entry['mask']) as raw:target=convert_labels(np.asarray(raw))
                matrix=confusion4(prediction,target)
                rows.append(dict(variant='exp13_best',image=path.name,method=method,group=entry['group'],
                    scope='image',benchmark_version=BENCHMARK,**photo_metrics(matrix)))
                key=(method,entry['group']);pooled[key]=pooled.get(key,np.zeros((4,4),np.int64))+matrix
                if path.name=='penguins_far.jpg':
                    for index,(x,y) in enumerate(FAR_POINTS,1):
                        value=int(prediction[y,x]);points.append(dict(method=method,point=index,x=x,y=y,
                            predicted_class=CLASSES[value],penguin_hit=value==1,any_animal_hit=value>0))
                del target
        preview=panel(image,masks,path.name+' | red=penguin, orange=bird, cyan=seal')
        preview.save(output/'preview.jpg',quality=85)
        gallery.append((path.name+(' | qualitative only; no IoU' if private else ' | benchmark_v2 development reference'),output/'preview.jpg'))
        if not private:panels[path.name]=preview.copy()
        records.append(dict(image=path.name,private=private,source_sha256=sha(path),output=str(output),inference=details))
        save_json(run/'prediction_progress.json',records)
        image.close();preview.close();del masks;release()
    for (method,group),matrix in pooled.items():
        rows.append(dict(variant='exp13_best',image='',method=method,group=group,scope='pooled',
            benchmark_version=BENCHMARK,**photo_metrics(matrix)))
    save_csv(run/'reference_metrics.csv',rows)
    save_csv(run/'landscape_metrics.csv',[r for r in rows if r['group']=='landscapes'])
    save_csv(run/'far_points.csv',points)
    point_summary=[dict(method=m,points=16,penguin_hits=sum(r['penguin_hit'] for r in points if r['method']==m),
        any_animal_hits=sum(r['any_animal_hit'] for r in points if r['method']==m),
        note='Strict pixel hits at the existing 16 body centers; not instance recall') for m in ('native','resize')]
    save_csv(run/'far_point_summary.csv',point_summary)
    rescore_history(root,run,ref_folder,manifest)
    gallery_path=write_gallery_html(run/'gallery.html','Experiment 13: four-class native / resize',gallery,
        'Frozen development-reference agreement only; NJU is qualitative. No union or sliding windows.')
    landscape=max((r for r in rows if r['scope']=='image' and r['group']=='landscapes' and r['method']=='native'),key=lambda r:r['background_fp_fraction'])['image']
    selected=[panels[name] for name in ('penguins_far.jpg','penguins_near.jpg',landscape)]
    canvas=Image.new('RGB',(1440,365*3),'white')
    for i,im in enumerate(selected):canvas.paste(im,(0,i*365))
    canvas.save(run/'comparison.jpg',quality=88)
    result=dict(protocol=PROTOCOL,status='complete',folder=str(run),benchmark_version=BENCHMARK,
        reference_manifest_sha256=sha(ref_folder/'manifest.json'),checkpoint_sha256=sha(weights),
        reference_images=9,landscapes=6,nju_images=6,records=records,far_points=point_summary,gallery=str(gallery_path),
        limitations=manifest['limitations'],inference_methods=['native','resize'],training_performed=False)
    save_json(run/'evaluation.json',result)
    del model;release()
    return result


def plot_training_history(run,report):
    """Render saved epoch logs only; no model loading, training or inference."""
    path=Path(run)/'history.json'
    if not path.exists():return None
    history=json.loads(path.read_text(encoding='utf-8'))
    if not history:return None
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(15,4.5));FigureCanvasAgg(fig);axes=fig.subplots(1,3)
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],'o-',label='Training')
    axes[0].plot(epochs,[r['valid_loss'] for r in history],'o-',label='Animal validation')
    axes[0].set(title='Cross-entropy loss',ylabel='Mean loss')
    for index,(name,color) in enumerate([('Penguin','#d34747'),('Other bird','#e59c23'),('Seal','#1c9eab')],1):
        axes[1].plot(epochs,[100*r['valid_metrics']['iou_per_class'][index] for r in history],'o-',color=color,label=name)
    axes[1].plot(epochs,[100*r['foreground_miou'] for r in history],'k--',linewidth=2,label='Foreground mean')
    axes[1].set(title='Animal-validation IoU',ylabel='IoU (%)',ylim=(0,100))
    for name,key in [('Animal validation',None),('Natural-background validation','nature_validation')]:
        metrics=[r['valid_metrics'] if key is None else r['valid_metrics'][key] for r in history]
        axes[2].plot(epochs,[100*m['background_fp_fraction'] for m in metrics],'o-',label=name)
    axes[2].set(title='Background predicted as any animal',ylabel='Reference background pixels (%)',ylim=(0,None))
    best=max(history,key=lambda r:r['foreground_miou'])['epoch']
    for ax in axes:
        ax.axvline(best,color='gray',linestyle=':',alpha=.6)
        ax.set(xlabel='Epoch',xticks=epochs);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Experiment 13 | four-class baseline | best checkpoint: epoch {best}',fontsize=14)
    fig.tight_layout();target=Path(report)/'training_curves.png';fig.savefig(target,dpi=150)
    return target


def update_report(root=None,run=None):
    root=Path(root or PROJECT_ROOT).resolve();report=root/'reports/exp13';report.mkdir(parents=True,exist_ok=True)
    pointer=root/'artifacts/experiments/exp13_results/latest_4class.json'
    if run is None and pointer.exists():run=Path(json.loads(pointer.read_text())['folder'])
    run=Path(run) if run else None
    training=json.loads((run/'training_result.json').read_text()) if run and (run/'training_result.json').exists() else None
    evaluation=json.loads((run/'evaluation.json').read_text()) if run and (run/'evaluation.json').exists() else None
    save_json(report/'config.json',training['config'] if training else BASE_CONFIG)
    text='# Experiment 13: new four-class baseline\n\n'
    text+='Status: '+('training complete' if training else 'prepared; not trained')+('. Evaluation complete.\n\n' if evaluation else '. Photo evaluation not run.\n\n')
    text+='Hand-written ResNet34 U-Net, four outputs, 3x3 kernels with the original 7x7 stem. ImageNet encoder; random decoder/head. No penguin-baseline or experiment-8/11 initialization. Ten joint-training epochs; 640 letterbox input, nearest masks; AMP, normal BN; AdamW lr=1e-4, weight decay=0.01; CrossEntropy(ignore=255); seed 42; constant LR. Batch 6 changes to 4 only after an OOM. Real preflight updates are discarded.\n\n'
    text+='Classes: background=0, penguin=1, other_bird=2, seal=3, ignore=255. Whale pixels map 4 to 0 on a separate dataset view; void/unknown/padding remain 255. Original data, splits, five-class globals and historical readers are preserved. Groups are recomputed from converted masks: largest target-foreground area, otherwise background. Per-image sampling weight is 1/sqrt(group size), with replacement and len(train) draws per epoch. This is an unvalidated mild resampling candidate, not pixel balancing. Actual draw counts are recorded per completed epoch.\n\n'
    text+='Model selection uses the three foreground-class mean IoU on the current deduplicated animal validation set. Natural-background validation is separate. Bird/seal recall comes from this animal validation, since the development references have no positive labels for these classes.\n\n'
    if training:
        text+=f"Best epoch: {training['best_epoch']}; animal foreground mIoU: {training['best_metrics']['foreground_miou']:.4f}; last epoch mIoU: {training['last_metrics']['foreground_miou']:.4f}. Actual final batch size: {training['config']['batch_size']}.\n\n"
        for source,target in [('epochs.csv','metrics.csv'),('data_statistics.csv','data_statistics.csv'),('best_last_metrics.csv','best_last_metrics.csv')]:
            shutil.copy2(run/source,report/target)
        if plot_training_history(run,report):
            text+='![Training curves](training_curves.png)\n\nSaved epoch logs: [CSV](metrics.csv). The plot separates training/animal-validation losses, class IoUs and background false positives. Natural-background validation is distinct from the six development landscape photographs.\n\n'
    if evaluation:
        for name in ('reference_metrics.csv','landscape_metrics.csv','far_points.csv','far_point_summary.csv','historical_reference_metrics.csv','comparison.jpg'):
            if (run/name).exists():shutil.copy2(run/name,report/name)
        text+='![Public development comparison](comparison.jpg)\n\n'
        text+=f"Full masks, inference devices and local gallery: `{run}`. Nine frozen benchmark_v2 references are scored with native and aspect-preserving resize inference only. The 16 far-penguin points use exact existing pixel coordinates, reporting penguin hits and any-animal hits, not instance recall. Six user-declared landscape negatives report pooled and per-animal false-positive pixel fractions. NJU photos are visualization only; no IoU is reported.\n\n"
    else:text+='After local training/evaluation, this folder receives metric CSVs and one small public comparison image; complete predictions and checkpoints remain under artifacts/.\n\n'
    text+='benchmark_v2 is preserved unchanged. Its masks are development references, partly model-assisted: near uses experiment-11 resize, and far has model-assisted bands. Scores are reference agreement, not an independent blind test. Historical reference predictions, when available, are rescored after mapping their whale labels to background; historical five-class validation mIoU is never compared directly with this four-class mIoU.\n\n'
    text+='This is a new baseline, not an isolated whale-removal ablation: initialization, kernel, current deduplicated data and sampling also differ from historical runs. This replaces the superseded sequential experiment-13 proposal. No test data is used for training or checkpoint selection.\n'
    (report/'README.md').write_text(text,encoding='utf-8')
    index=root/'reports/index.csv'
    if index.exists():
        with index.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames;entries=list(reader)
        if not any(str(row.get('experiment'))=='13' for row in entries):
            entries.append({key:'' for key in fields});entries[-1]['experiment']='13'
        for row in entries:
            if str(row.get('experiment'))=='13':
                updates=dict(title='New four-class ImageNet baseline',status='completed' if training else 'prepared',
                    report='reports/exp13/README.md',reference_version=BENCHMARK,
                    first_protocol='Native / resize' if evaluation else '',
                    key_result='Four-class baseline; see separate metrics; not a whale-only ablation')
                row.update({k:v for k,v in updates.items() if k in fields})
        temp=index.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(entries)
        temp.replace(index)


def run_experiment13(*,root=None,run_dir=None,train=False,evaluate=False):
    root=Path(root or PROJECT_ROOT).resolve()
    if not train and not evaluate:
        print('PLAN ONLY. Set RUN_TRAINING_13=True locally to audit, preflight and train; RUN_EVALUATION_13=True to infer the trained best model.')
        return dict(status='prepared',config=BASE_CONFIG,run_dir=str(run_dir) if run_dir else None)
    if run_dir is None:
        raise ValueError('Pass an explicit run_<timestamp> directory so repeated notebook calls can resume safely')
    run=Path(run_dir).resolve()
    if evaluate and not train and not (run/'best.pt').exists():
        raise FileNotFoundError('Evaluation needs best.pt from this trained run; pass its run_dir')
    run.mkdir(parents=True,exist_ok=True)
    path=run/'request.json'
    request=json.loads(path.read_text()) if path.exists() else {'config':BASE_CONFIG,'imagenet_sha256':cached_imagenet()[1]}
    if request['config']!=BASE_CONFIG:raise ValueError('Existing run config differs; choose a new directory')
    if train and not (run/'training_result.json').exists() and request['imagenet_sha256']!=cached_imagenet()[1]:
        raise ValueError('ImageNet initializer changed')
    save_json(path,request)
    source_hashes={p.name:sha(p) for p in (Path(__file__),root/'scripts/script.py',root/'scripts/animal_data.py',root/'scripts/experiment9.py')}
    if not (run/'source_hashes.json').exists():save_json(run/'source_hashes.json',source_hashes)
    if evaluate:save_json(run/'evaluation_source_hashes.json',source_hashes)
    result={'protocol':PROTOCOL,'folder':str(run),'config':BASE_CONFIG}
    try:
        if train:result['training']=train_baseline(root,run)
        if evaluate:result['evaluation']=evaluate_best(root,run)
    finally:
        save_json(root/'artifacts/experiments/exp13_results/latest_4class.json',dict(folder=str(run),protocol=PROTOCOL))
        update_report(root,run)
    save_json(run/'result.json',result)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true');parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    run_experiment13(run_dir=args.run_dir,train=args.train,evaluate=args.evaluate)
