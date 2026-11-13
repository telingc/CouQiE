"""Experiment 16, group B only: source-aware background sampling.

LLM-assisted: Codex, 2026-11-12. Nothing executes on import. The default call
does not train, infer or create a run. Actual results and reporting are deferred
until the user explicitly runs training. No historical segmentation initializer.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
from time import perf_counter
import csv
import gc
import json

import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import ConcatDataset,DataLoader,WeightedRandomSampler

from project_paths import PROJECT_ROOT
from animal_data import AnimalDataset,CLASS_NAMES,build_datasets
from script import Trainer,evaluate_segmentation
from experiment13 import sha,save_json,save_csv,cached_imagenet,state_hash,_close
from experiment15 import new_model,load_checkpoint,metrics5,flatten,evaluate960,infer960,confusion5
from experiment14 import strict_float32,NJU_NAMES
from gallery import write_gallery_html,save_visualization

GROUPS=('penguin','other_bird','seal','whale','nature_background','antarctic_background')
SIZES=(423,5326,871,385,1759,142)
PROBABILITIES=(.111,.395,.160,.106,.057,.171)
SOURCE_NAMES=('dataset_penguins','dataset_birds','dataset_seals','dataset_whales','dataset_nature')
PROTOCOL='exp16_group_b_fixed_source_sampling_v1'
CONFIG=dict(experiment=16,group='B',protocol=PROTOCOL,num_classes=5,class_names=list(CLASS_NAMES),
    kernel_size=3,stem_kernel=7,initialization='ImageNet encoder; random decoder/head; no historical segmentation weights',
    epochs=10,input_size=640,private_inference_size=960,source_validation_size=640,
    requested_batch_size=6,oom_fallback_batch_size=4,amp=True,bn_policy='normal',seed=42,
    optimizer='AdamW',learning_rate=1e-4,weight_decay=.01,learning_rate_schedule='constant',
    loss='CrossEntropyLoss(ignore_index=255); no class weights',draws_per_epoch=8764,
    groups={name:dict(images=n,probability=p,per_image_weight=p/n) for name,n,p in zip(GROUPS,SIZES,PROBABILITIES)},
    sampling='group_probability/group_size; uniform within groups; replacement',
    selection='private SAM3-reference validation foreground mIoU; not an independent blind test',
    private_validation_precision='float32; autocast and TF32 off; native-coordinate metrics after restoring continuous logits',
    augmentation='none',fusion='none',sliding_window=False)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class SixGroupSampler(WeightedRandomSampler):
    def __init__(self,sizes=SIZES):
        if tuple(sizes)!=SIZES:raise ValueError('Unexpected training-group sizes')
        self.groups=np.repeat(np.arange(6),sizes)
        weights=np.concatenate([np.full(n,p/n) for n,p in zip(sizes,PROBABILITIES)])
        if not np.isclose(weights.sum(),1):raise ValueError('Sampling probabilities do not sum to one')
        super().__init__(torch.as_tensor(weights,dtype=torch.double),8764,replacement=True,
            generator=torch.Generator().manual_seed(42))
        self.last_counts=None
    def __iter__(self):
        indices=list(super().__iter__())
        self.last_counts=np.bincount(self.groups[indices],minlength=6).tolist()
        return iter(indices)


def metrics_row(metrics):
    row=flatten(metrics)
    row.update({key:metrics[key] for key in ('seal_on_background','penguin_on_background',
        'pixel_accuracy','miou','dice','precision','recall')})
    return row


def load_data(root,run=None):
    """Use only the active merged view; private foreground masks never train B."""
    root=Path(root);layout=read(root/'data/dataset_layout.json')
    if layout['version']!='merged_sources_v1':raise ValueError('The current merged source layout is required')
    source,metadata=build_datasets(root,640,include_background=True)
    children=source['train'].datasets
    if tuple(d.root.name for d in children)!=SOURCE_NAMES:raise ValueError('Unexpected source datasets; no raw Wildlife input is permitted')
    if any(d.root.resolve()!=root/'data'/name for d,name in zip(children,SOURCE_NAMES)):raise ValueError('Archived source data is forbidden in experiment 16')
    antarctic=AnimalDataset(root/'data/dataset_antarctic_background','train',640)
    datasets=[*children,antarctic]
    if tuple(len(d) for d in datasets)!=SIZES:raise ValueError('Training counts differ from 423/5326/871/385/1759/142')
    for dataset,semantic in zip(datasets,(1,2,3,4,0,0)):
        if any(r['sample_class']!=semantic for r in dataset.records):raise ValueError('Sampling group and semantic labels disagree')
    whale_origins=Counter(r.get('origin_dataset') for r in children[3].records)
    if whale_origins!={'dataset_whales':344,'dataset_wildlife':41}:raise ValueError('Whales must include 344 Happywhale plus 41 legacy samples')
    private=AnimalDataset(root/'data/dataset_private_prepared','valid',640)
    private_train=read(root/'data/dataset_private_prepared/train/manifest.json')['samples']
    if len(private)!=25:raise ValueError('Expected 25 frozen private validation images')
    if {r['id'] for r in antarctic.records}!={r['id'] for r in private_train}:raise ValueError('Antarctic background must use exactly private training photos')
    if {r['scene_group'] for r in antarctic.records}&{r['scene_group'] for r in private.records}:raise ValueError('Target scene leakage')
    bg_root=antarctic.root;request=read(bg_root/'request.json');verified=read(bg_root/'verification.json')
    if verified['status']!='passed' or request['margin_reference_pixels']!=16 or request['margin_reference_canvas']!=960:
        raise ValueError('The verified full-image safety-margin dataset is required')
    if request['source_train_manifest_sha256']!=sha(root/'data/dataset_private_prepared/train/manifest.json'):
        raise ValueError('Antarctic background parent annotations changed')
    if request['source_valid_manifest_sha256']!=sha(root/'data/dataset_private_prepared/valid/manifest.json'):
        raise ValueError('Frozen target validation changed')
    animals=ConcatDataset(source['valid'].datasets[:4]);nature=source['valid'].datasets[4]
    manifests=[root/'data/dataset_layout.json',bg_root/'request.json',bg_root/'verification.json',
        root/'data/dataset_private_prepared/train/manifest.json',root/'data/dataset_private_prepared/valid/manifest.json']
    manifests += [d.split_dir/'manifest.json' for split in source.values() for d in split.datasets]
    manifests += [antarctic.split_dir/'manifest.json']
    identity={str(p.relative_to(root).as_posix()):sha(p) for p in manifests}
    audit=dict(manifest_sha256=identity,source_layout=layout['version'],training_images=8906,
        train_group_counts=dict(zip(GROUPS,SIZES)),whale_origins=dict(whale_origins),
        validation_images=dict(source_animals=len(animals),nature=len(nature),private=25),
        private_annotation_origin='User-confirmed high-quality SAM3 reference; not independent human ground truth',
        training_private_use='Only the background-only 142-image derivative; animal RGB retained, excluded from loss',
        background_labels=[0,255],source_metadata=metadata)
    if run is not None:
        audit_files(root,datasets,source,private,private_train)
        path=Path(run)/'data_audit.json'
        if path.exists() and read(path)!=audit:raise ValueError('Dataset identity changed; choose a new run directory')
        save_json(path,audit)
    return ConcatDataset(datasets),animals,nature,private,audit


def audit_files(root,datasets,source,private,private_train):
    """Check actual file hashes, split isolation and excluded animal pixels once."""
    forbidden=set()
    for split in ('valid','test'):
        for child in source[split].datasets:
            for row in child.records:forbidden.update(filter(None,(row.get('image_sha256'),row.get('prepared_image_sha256'),row.get('raw_sha256'))))
    forbidden.update(r['image_sha256'] for r in private.records)
    forbidden.update(sha(root/'data/nju_testset'/name) for name in NJU_NAMES)
    seen=set()
    for child in [*datasets,private]:
        for i,row in enumerate(child.records):
            digest=sha(child.image_path(i));expected=row.get('image_sha256',row.get('prepared_image_sha256'))
            if digest!=expected:raise ValueError('Dataset image hash mismatch')
            if sha(child.split_dir/row['mask'])!=row.get('mask_sha256',row.get('prepared_mask_sha256')):raise ValueError('Mask hash mismatch')
            if child is not private:
                if digest in forbidden or row.get('raw_sha256') in forbidden or digest in seen:raise ValueError('Training duplicate or validation/test overlap')
                seen.add(digest)
    parents={r['id']:r for r in private_train}
    for row in datasets[-1].records:
        parent=parents[row['id']]
        if row['parent_mask_sha256']!=parent['mask_sha256'] or row['image_sha256']!=parent['image_sha256']:
            raise ValueError('Background parent identity mismatch')
        if sha(root/row['parent_mask'])!=row['parent_mask_sha256']:
            raise ValueError('Background parent mask bytes changed')
        with Image.open(datasets[-1].split_dir/row['mask']) as im:
            values=np.array(im);hist=im.histogram()
            if im.size!=(row['width'],row['height']) or not hist[0] or sum(hist[1:255]):raise ValueError('Expected original-size background 0 / ignore 255')
        with Image.open(root/row['parent_mask']) as im:
            target=np.asarray(im)
            if target.shape!=values.shape or not np.all(values[target!=0]==255):raise ValueError('Animal/unknown pixels are not fully excluded')
        if row['margin_native_pixels']!=int(np.ceil(16*max(row['width'],row['height'])/960)):
            raise ValueError('Unexpected safety margin')


def make_loader(dataset,batch,sampler=None,workers=2):
    return DataLoader(dataset,batch_size=batch,sampler=sampler,shuffle=False,num_workers=workers,
        pin_memory=True,persistent_workers=workers>0)


def preflight(train,run):
    """Two real AMP optimizer steps on a disposable model; only OOM changes batch."""
    first=[0,*train.cumulative_sizes[:-1]];samples=[train[i] for i in first];attempts=[]
    for batch in (6,4):
        model=optimizer=scaler=images=targets=logits=loss=None
        try:
            model=new_model();initial=state_hash(model);model.cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=1024)
            images=torch.stack([x for x,y in samples[:batch]]).cuda();targets=torch.stack([y for x,y in samples[:batch]]).cuda()
            torch.cuda.reset_peak_memory_stats();losses=[]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(images);loss=torch.nn.functional.cross_entropy(logits,targets,ignore_index=255)
                if logits.shape!=(batch,5,640,640) or not torch.isfinite(loss):raise ValueError('Invalid five-class preflight')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Non-finite gradients')
                scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            torch.cuda.synchronize();attempts.append(dict(batch=batch,status='passed',losses=losses,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
            result=dict(batch_size=batch,initial_state_sha256=initial,attempts=attempts,preflight_updates_discarded=True)
            save_json(run/'preflight.json',result);return result
        except torch.OutOfMemoryError:
            attempts.append(dict(batch=batch,status='OOM'));save_json(run/'preflight_attempts.json',attempts)
            if batch==4:raise
        finally:
            del model,optimizer,scaler,images,targets,logits,loss
            gc.collect();torch.cuda.empty_cache()


class Experiment16Trainer(Trainer):
    def validate(self):
        monitors={}
        for name,loader in [('source_animals',self.valid_loader),('nature',self.background_loader)]:
            raw=evaluate_segmentation(self.model,loader,self.device,criterion=self.criterion,num_classes=5,amp=True)
            monitors[name]=dict(metrics5(raw['confusion_matrix']),loss=raw['loss'],num_images=raw['num_images'],evaluation_size=640,precision='AMP')
        self.source_validation_peak=torch.cuda.max_memory_allocated()/2**30
        folder=self.checkpoint_dir/'validation/private'/f'epoch_{self.epoch:02d}'
        private=evaluate960(self.model,self.private_valid,output=folder)
        private.update(annotation_origin='User-confirmed-high-quality SAM3 reference',role='checkpoint validation; not blind test')
        save_json(folder/'metrics.json',private)
        with (folder/'image_metrics.csv').open(encoding='utf-8-sig') as stream:resources=list(csv.DictReader(stream))
        self.private_validation_peak=max([float(r['peak_allocated_GiB']) for r in resources if r.get('peak_allocated_GiB')]+[0.])
        self.model.to(self.device)
        self.validation_metrics=dict(private,source_validation=monitors)
        if private['foreground_miou'] is None:raise ValueError('Private validation has no foreground positive labels')
        return private['loss'],private['foreground_miou']
    def fit(self,epochs):
        for _ in range(epochs):
            self.epoch+=1;started=perf_counter();torch.cuda.reset_peak_memory_stats()
            loss=self.train_one_epoch();torch.cuda.synchronize();train_seconds=perf_counter()-started
            train_peak=torch.cuda.max_memory_allocated()/2**30
            valid_loss,score=self.validate();torch.cuda.synchronize()
            counts=self.train_loader.sampler.last_counts
            if sum(counts)!=8764:raise ValueError('An epoch must contain exactly 8764 draws')
            row=dict(epoch=self.epoch,train_loss=loss,valid_loss=valid_loss,foreground_miou=score,
                valid_metrics=self.validation_metrics,actual_sampling_counts=dict(zip(GROUPS,counts)),
                draws=8764,batch_size=self.train_loader.batch_size,train_seconds=train_seconds,
                epoch_seconds=perf_counter()-started,peak_allocated_GiB=max(train_peak,self.source_validation_peak,self.private_validation_peak))
            self.history.append(row);save_json(self.checkpoint_dir/'history.json',self.history)
            rows=[]
            for r in self.history:
                flat={k:r[k] for k in ('epoch','train_loss','batch_size','train_seconds','epoch_seconds','peak_allocated_GiB')}
                flat.update({'sampled_'+k:v for k,v in r['actual_sampling_counts'].items()})
                for name,m in [('private',r['valid_metrics']),*r['valid_metrics']['source_validation'].items()]:
                    flat.update({name+'_'+k:v for k,v in metrics_row(m).items()})
                rows.append(flat)
            save_csv(self.checkpoint_dir/'epochs.csv',rows)
            self.save_checkpoint(valid_loss,score);self._write_checkpoint(self.checkpoint_dir/'last.pt')
            save_json(self.checkpoint_dir/'status.json',dict(status='training',group='B',completed_epochs=self.epoch,best_private_miou=self.best_iou))
            print(f'EPOCH {self.epoch:02d}/10 | train={loss:.4f} | private mIoU={score:.4f} | private background FP={self.validation_metrics["background_fp_fraction"]:.3%} | seconds={row["epoch_seconds"]:.1f}',flush=True)
            print('ACTUAL DRAWS | '+' | '.join(f'{n}={v}' for n,v in row['actual_sampling_counts'].items()),flush=True)
        return self.history


def train_group_b(root,run,workers):
    train,animals,nature,private,audit=load_data(root,run)
    if (run/'training_result.json').exists():
        result=read(run/'training_result.json')
        if sha(run/'best.pt')!=result['best_sha256'] or sha(run/'last.pt')!=result['last_sha256']:
            raise ValueError('Completed checkpoint identity changed')
        return result
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this AMP training protocol')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    probe=read(run/'preflight.json') if (run/'preflight.json').exists() else preflight(train,run)
    model=new_model()
    if state_hash(model)!=probe['initial_state_sha256']:raise ValueError('Formal model did not reset after preflight')
    model.cuda();batch=probe['batch_size'];optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    loaders=[]
    try:
        train_loader=make_loader(train,batch,SixGroupSampler(),workers);loaders.append(train_loader)
        animal_loader=make_loader(animals,batch,workers=workers);loaders.append(animal_loader)
        nature_loader=make_loader(nature,batch,workers=workers);loaders.append(nature_loader)
        trainer=Experiment16Trainer(model,train_loader,animal_loader,torch.nn.CrossEntropyLoss(ignore_index=255),optimizer,
            'cuda:0',run,class_names=CLASS_NAMES,amp=True,bn_policy='normal',gradient_clip=None,
            log_every=0,background_loader=nature_loader,dataset_metadata=audit)
        trainer.private_valid=private
        trainer.config.update(CONFIG,batch_size=batch,imagenet_sha256=cached_imagenet()[1],preflight_updates_discarded=True)
        if (run/'last.pt').exists():trainer.resume(run/'last.pt')
        save_json(run/'config.json',trainer.config);trainer.fit(max(0,10-trainer.epoch))
        # Strictly reload both five-class states on CPU; no further optimization.
        best_model,best=load_checkpoint(run/'best.pt');del best_model
        last_model,last=load_checkpoint(run/'last.pt');del last_model
        save_json(run/'best_last_comparison.json',dict(best_epoch=best['epoch'],last_epoch=last['epoch'],
            best_metrics=best['valid_metrics'],last_metrics=last['valid_metrics'],
            strict_five_class_loading='passed',selection='private validation foreground mIoU'))
        result=dict(status='complete',group='B',epochs=trainer.epoch,batch_size=batch,best_private_miou=trainer.best_iou,
            best_epoch=best['epoch'],strict_five_class_loading='passed',
            best=str(run/'best.pt'),last=str(run/'last.pt'),best_sha256=sha(run/'best.pt'),last_sha256=sha(run/'last.pt'),report_status='waiting_for_user_instruction')
        save_json(run/'training_result.json',result);save_json(run/'status.json',result)
        return result
    finally:
        for item in loaders:_close(item)
        del model,optimizer
        gc.collect();torch.cuda.empty_cache()


def visualize_best(root,run):
    """960-only best-model predictions; no NJU IoU and no public course figures."""
    *_,audit=load_data(root)
    if read(run/'data_audit.json')['manifest_sha256']!=audit['manifest_sha256']:
        raise ValueError('Dataset manifests changed since training')
    model,saved=load_checkpoint(run/'best.pt')
    if saved['config'].get('protocol')!=PROTOCOL:raise ValueError('Expected experiment-16 group-B weights')
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    out=run/'predictions'/('best_'+sha(run/'best.pt')[:12]);out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for name in ('penguins_far.jpg','penguins_near.jpg'):
        cases.append(dict(name=name,source='data/examples/'+name,group='examples'))
    for name in ('antarctic_landscape_mather_lake.jpg','antarctic_landscape_untitled_1.jpeg',*[f'antarctic_landscape_untitled_{i}.jpg' for i in range(2,6)]):
        cases.append(dict(name=name,source='data/examples/'+name,group='landscapes'))
    private=AnimalDataset(root/'data/dataset_private_prepared','valid')
    for row in private.records:
        cases.append(dict(name=row['original_name'],source=str((private.split_dir/row['image']).relative_to(root).as_posix()),
            reference_mask=str((private.split_dir/row['mask']).relative_to(root).as_posix()),group='private_valid_sam3'))
    whale=AnimalDataset(root/'data/dataset_whales','valid')
    row=next(r for r in whale.records if r['origin_dataset']=='dataset_whales')
    cases.append(dict(name=Path(row['image']).name,source=str((whale.split_dir/row['image']).relative_to(root).as_posix()),group='source_whale'))
    cases.extend(dict(name=name,source='data/nju_testset/'+name,group='nju') for name in NJU_NAMES)
    items=[];records=[]
    with strict_float32():
        for case in cases:
            with Image.open(root/case['source']) as im:image=ImageOps.exif_transpose(im).convert('RGB')
            try:prediction,stats,_=infer960(model,image,device)
            except torch.OutOfMemoryError:
                model.cpu();torch.cuda.empty_cache();prediction,stats,_=infer960(model,image,'cpu')
            folder=out/case['group']/Path(case['name']).stem;folder.mkdir(parents=True,exist_ok=True)
            mask=folder/'class_ids.png';Image.fromarray(prediction).save(mask)
            visual=save_visualization(root,folder,case,prediction,mask)
            record=dict(**case,**stats,checkpoint_sha256=sha(run/'best.pt'),image_sha256=sha(root/case['source']),
                mask=str(mask),mask_sha256=sha(mask),**visual)
            if 'reference_mask' in case:
                with Image.open(root/case['reference_mask']) as gt:record['metrics']=metrics5(confusion5(prediction,np.asarray(gt)))
                record['reference_mask_sha256']=sha(root/case['reference_mask'])
                record['reference_origin']='User-confirmed-high-quality SAM3 reference'
            save_json(folder/'record.json',record);records.append(record)
            items.append((case['group']+' / '+case['name'],folder/visual['overlay']));image.close()
    gallery=write_gallery_html(out/'gallery.html','Experiment 16 group B: best checkpoint, 960 float32',items,
        'Background purple, penguin red, other bird orange, seal cyan, whale green. Private references are SAM3-assisted validation, not a blind test. NJU is qualitative only. No sliding windows or fusion.')
    save_json(out/'result.json',dict(checkpoint=str(run/'best.pt'),gallery=str(gallery),records=records))
    save_csv(out/'image_metrics.csv',[dict(image=r['name'],group=r['group'],**metrics_row(r['metrics'])) for r in records if 'metrics' in r])
    model.cpu();del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(gallery=str(gallery),folder=str(out),images=len(records))


def run_experiment16(*,root=None,run_dir=None,run_training=False,run_visualization=False,workers=2):
    root=Path(root or PROJECT_ROOT).resolve();base=root/'artifacts/experiments/exp16_results'
    if not run_training and not run_visualization:return dict(status='prepared_only',config=CONFIG,training_executed=False)
    latest=base/'latest.json'
    if run_dir is None:
        if latest.exists():run=Path(read(latest)['run_dir'])
        elif run_training:run=base/('run_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
        else:raise FileNotFoundError('Run group-B training first or provide its run_dir')
    else:run=Path(run_dir).resolve()
    if not run.is_relative_to(base):raise ValueError('Use an experiment-16 run directory')
    run.mkdir(parents=True,exist_ok=True)
    request=dict(config=CONFIG,workers=workers,implementation_sha256=sha(Path(__file__)))
    if run_training:
        if not (run/'protocol.json').exists() and any(run.iterdir()):raise ValueError('Nonempty directory has no experiment-16 protocol')
        if (run/'protocol.json').exists() and read(run/'protocol.json')!=request:raise ValueError('Training protocol/code changed; use a new run_dir or the frozen source snapshot')
        save_json(run/'protocol.json',request)
    elif not (run/'protocol.json').exists() or read(run/'protocol.json')['config']!=CONFIG:
        raise ValueError('Visualization requires a completed matching experiment-16 run')
    save_json(latest,dict(run_dir=str(run),group='B'))
    result=dict(run_dir=str(run))
    if run_training:result['training']=train_group_b(root,run,workers)
    if run_visualization:result['visualization']=visualize_best(root,run)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--train',action='store_true');parser.add_argument('--visualize',action='store_true');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();print(json.dumps(run_experiment16(run_dir=args.run_dir,run_training=args.train,
        run_visualization=args.visualize,workers=args.workers),indent=2))
