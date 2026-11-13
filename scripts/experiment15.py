"""Experiment 15: completed five-class source baseline and fixed 960 inference.

No work on import. The unexecuted annotation/fine-tuning workflow is archived.
Historical experiments/reports and submission bundles are never updated.
"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path,WindowsPath
from time import perf_counter
import json
import math

import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler

from project_paths import PROJECT_ROOT
from script import ResNetUNet,Trainer,evaluate_segmentation,metrics_from_confusion
from experiment13 import sha,save_json,save_csv,clean,seed,cached_imagenet,encoder_key,imagenet_value,state_hash,_close
from experiment15_data import CLASSES,ALLOWED,source_data,target_inventory

PROTOCOL='exp15_five_class_human_corrected_target_v1'
CONFIG=dict(protocol=PROTOCOL,class_names=list(CLASSES),num_classes=5,kernel_size=3,stem_kernel=7,
    initialization='ImageNet encoder; random decoder/head; no historical weights',epochs=10,
    image_size=640,inference_size=960,requested_batch_size=6,amp=True,bn_policy='normal',
    optimizer='AdamW',learning_rate=1e-4,weight_decay=.01,seed=42,ignore_index=255,
    sampling='per-image 1/sqrt(group_size); replacement; len(train) draws',workers=2,
    learning_rate_schedule='constant',label_mapping='identity: whale remains 4',
    stage_b=dict(epochs=5,learning_rate=1e-5,target_source_ratio='1:1',optimizer_reset=True,
        requires_explicit_human_completion=True,training_size=640,evaluation_size=960))


def new_model():
    seed();path,_=cached_imagenet();model=ResNetUNet(num_classes=5,pretrained=False,kernel_size=3)
    weights=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=model.state_dict()
    for name in state:
        key=encoder_key(name)
        if key is not None:state[name]=imagenet_value(weights,key,state[name])
    model.load_state_dict(state,strict=True)
    return model


def load_checkpoint(path):
    with torch.serialization.safe_globals([WindowsPath]):saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if saved['num_classes']!=5 or tuple(saved['class_names'])!=CLASSES:raise ValueError('Expected five-class weights')
    model=ResNetUNet(num_classes=5,pretrained=False,kernel_size=3)
    model.load_state_dict(saved['model_state'],strict=True);model.float().eval()
    return model,saved


def metrics5(matrix):
    """Undefined class metrics are None; foreground mean uses GT-present classes."""
    matrix=np.asarray(matrix,dtype=np.int64)
    if matrix.shape!=(5,5):raise ValueError('Expected 5x5 confusion matrix')
    result=clean(metrics_from_confusion(matrix));present=matrix.sum(1)>0
    for name in ('iou','precision','recall','dice'):
        for i in range(5):
            if not present[i]:result[name+'_per_class'][i]=None
    foreground=[result['iou_per_class'][i] for i in range(1,5) if present[i]]
    result['foreground_miou']=float(np.mean(foreground)) if foreground else None
    result['miou']=float(np.mean([result['iou_per_class'][i] for i in range(5) if present[i]])) if present.any() else None
    for metric in ('precision','recall','dice'):
        values=[result[metric+'_per_class'][i] for i in range(1,5) if result[metric+'_per_class'][i] is not None]
        result[metric]=float(np.mean(values)) if values else None
    result['foreground_classes_with_positive_labels']=int(present[1:].sum())
    denominator=int(matrix[0].sum())
    result.update(background_pixels=denominator,background_false_positive_pixels=int(matrix[0,1:].sum()),
        background_fp_fraction=float(matrix[0,1:].sum()/denominator) if denominator else None,
        label_pixels={name:int(matrix[c].sum()) for c,name in enumerate(CLASSES)},
        **{name+'_on_background':float(matrix[0,c]/denominator) if denominator else None for c,name in enumerate(CLASSES) if c})
    return result


def flatten(metrics):
    row={k:metrics.get(k) for k in ('loss','foreground_miou','pixel_accuracy','background_fp_fraction','other_bird_on_background','whale_on_background')}
    for i,name in enumerate(CLASSES):
        for metric in ('iou','precision','recall','dice'):row[name+'_'+metric]=metrics[metric+'_per_class'][i]
        row[name+'_label_pixels']=metrics['label_pixels'][name]
    return row


class RecordedSampler(WeightedRandomSampler):
    def __init__(self,groups):
        self.groups=np.asarray(groups);counts=np.bincount(self.groups,minlength=5)
        super().__init__([1/math.sqrt(counts[g]) for g in self.groups],len(groups),replacement=True,
            generator=torch.Generator().manual_seed(42));self.last_counts=None
    def __iter__(self):
        indices=list(super().__iter__());self.last_counts=np.bincount(self.groups[indices],minlength=5).tolist()
        return iter(indices)


class StageTrainer(Trainer):
    def validate(self):
        if getattr(self,'target_evaluator',None):
            metrics=self.target_evaluator(self.model)
            metrics['source_monitor']=self.source_evaluator(self.model)
        else:
            raw=evaluate_segmentation(self.model,self.valid_loader,self.device,criterion=self.criterion,num_classes=5,amp=self.amp)
            metrics=metrics5(raw['confusion_matrix']);metrics.update(loss=raw['loss'],num_images=raw['num_images'])
            raw_nature=evaluate_segmentation(self.model,self.nature_loader,self.device,criterion=self.criterion,num_classes=5,amp=self.amp)
            nature=metrics5(raw_nature['confusion_matrix']);nature.update(loss=raw_nature['loss'],num_images=raw_nature['num_images'])
            metrics['nature_validation']=nature
        self.validation_metrics=clean(metrics)
        self.model.to(self.device)
        if metrics['foreground_miou'] is None:raise ValueError('Validation has no foreground positive labels')
        return metrics['loss'],metrics['foreground_miou']
    def fit(self,epochs):
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        for _ in range(epochs):
            if getattr(self,'annotation_manifest_path',None) and sha(self.annotation_manifest_path)!=self.annotation_snapshot_sha:
                raise ValueError('Human annotations changed after the fine-tuning snapshot; stop and review')
            self.epoch+=1;torch.cuda.reset_peak_memory_stats();started=perf_counter()
            print(f'EPOCH {self.epoch:02d} START',flush=True)
            train_loss=self.train_one_epoch()
            if getattr(self,'annotation_manifest_path',None) and sha(self.annotation_manifest_path)!=self.annotation_snapshot_sha:
                raise ValueError('Human annotations changed during the epoch; no checkpoint is selected')
            valid_loss,score=self.validate();torch.cuda.synchronize()
            counts=self.train_loader.sampler.last_counts
            if sum(counts)!=len(self.train_loader.sampler):raise ValueError('Sampling accounting mismatch')
            row=dict(epoch=self.epoch,train_loss=train_loss,valid_loss=valid_loss,foreground_miou=score,
                valid_metrics=self.validation_metrics,epoch_seconds=perf_counter()-started,
                batch_size=self.train_loader.batch_size,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,actual_sampling_counts=dict(zip(CLASSES,counts)))
            if hasattr(self.train_loader.sampler,'last_target_count'):
                row.update(target_draws=self.train_loader.sampler.last_target_count,source_draws=self.train_loader.sampler.last_source_count,
                    source_group_draws=dict(zip(CLASSES,self.train_loader.sampler.last_source_group_counts)))
            self.history.append(row);save_json(self.checkpoint_dir/'history.json',self.history)
            save_csv(self.checkpoint_dir/'epochs.csv',[dict(epoch=r['epoch'],train_loss=r['train_loss'],
                epoch_seconds=r['epoch_seconds'],peak_allocated_GiB=r['peak_allocated_GiB'],
                **flatten(r['valid_metrics']),**{'sampled_'+n:v for n,v in r['actual_sampling_counts'].items()}) for r in self.history])
            if 'nature_validation' in self.validation_metrics:
                save_csv(self.checkpoint_dir/'background_epochs.csv',[dict(epoch=r['epoch'],**flatten(r['valid_metrics']['nature_validation'])) for r in self.history])
            self.save_checkpoint(valid_loss,score);self._write_checkpoint(self.checkpoint_dir/'last.pt')
            display=' | '.join(f"{name}={self.validation_metrics['iou_per_class'][i]:.4f}" if self.validation_metrics['iou_per_class'][i] is not None else name+'=N/A' for i,name in enumerate(CLASSES))
            print(f'EPOCH {self.epoch:02d} | train={train_loss:.4f} | valid={valid_loss:.4f} | foreground_mIoU={score:.4f} | seconds={row["epoch_seconds"]:.1f}',flush=True)
            print('CLASS IOU | '+display,flush=True)
            save_json(self.checkpoint_dir.parent/'status.json',dict(stage='A' if self.config.get('stage')=='A' else 'B',epoch=self.epoch,status='training',stage_b='waiting_for_human_annotations'))
        return self.history


def loader(data,batch,sampler=None):
    return DataLoader(data,batch_size=batch,sampler=sampler,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)


def preflight(train,folder):
    samples=[train[int(np.flatnonzero(train.groups==c)[0])] for c in (1,2,3,4,0)]
    samples.append(train[int(np.flatnonzero(train.groups==1)[-1])]);attempts=[]
    for batch in (6,4):
        model=optimizer=scaler=images=targets=logits=loss=None
        try:
            model=new_model();initial=state_hash(model);weights_path,digest=cached_imagenet()
            weights=torch.load(weights_path,map_location='cpu',weights_only=True,mmap=True)
            for name,value in model.state_dict().items():
                key=encoder_key(name)
                if key is not None and not torch.equal(value,imagenet_value(weights,key,value)):raise ValueError('Encoder initializer mismatch')
            del weights
            model.cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=1024)
            images=torch.stack([x[0] for x in samples[:batch]]).cuda();targets=torch.stack([x[1] for x in samples[:batch]]).cuda()
            losses=[];torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):logits=model(images);loss=torch.nn.functional.cross_entropy(logits,targets,ignore_index=255)
                if logits.shape!=(batch,5,640,640) or not torch.isfinite(loss):raise ValueError('Invalid five-class preflight')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Invalid gradients')
                scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            torch.cuda.synchronize();attempts.append(dict(batch_size=batch,status='passed',losses=losses,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            result=dict(batch_size=batch,initial_state_sha256=initial,imagenet_sha256=digest,attempts=attempts,
                encoder_verified=True,preflight_updates_reused=False,formal_model_reinitialized=True)
            save_json(folder/'preflight.json',result);return result
        except torch.OutOfMemoryError:
            attempts.append(dict(batch_size=batch,status='OOM'));save_json(folder/'preflight_attempts.json',attempts)
            if batch==4:raise
        finally:
            del model,optimizer,scaler,images,targets,logits,loss
            import gc
            gc.collect();torch.cuda.empty_cache()


def train_stage_a(root,run):
    folder=run/'stage_a';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_result.json').exists():return json.loads((folder/'training_result.json').read_text())
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for AMP source training')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.backends.cudnn.benchmark=False
    train,animals,nature,audit=source_data(root,run)
    probe=json.loads((folder/'preflight.json').read_text()) if (folder/'preflight.json').exists() else preflight(train,folder)
    model=new_model()
    if state_hash(model)!=probe['initial_state_sha256']:raise ValueError('Formal initialization differs from discarded preflight initializer')
    model.cuda();batch=probe['batch_size'];optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    loaders=[]
    try:
        train_loader=loader(train,batch,RecordedSampler(train.groups));loaders.append(train_loader)
        valid_loader=loader(animals,batch);loaders.append(valid_loader)
        nature_loader=loader(nature,batch);loaders.append(nature_loader)
        trainer=StageTrainer(model,train_loader,valid_loader,torch.nn.CrossEntropyLoss(ignore_index=255),optimizer,
            'cuda:0',folder,class_names=CLASSES,amp=True,bn_policy='normal',gradient_clip=None,log_every=0,dataset_metadata=audit)
        trainer.nature_loader=nature_loader;trainer.config.update(CONFIG,stage='A',batch_size=batch,imagenet_sha256=probe['imagenet_sha256'])
        if (folder/'last.pt').exists():trainer.resume(folder/'last.pt')
        save_json(folder/'config.json',trainer.config);trainer.fit(max(0,10-trainer.epoch))
        best_model,best=load_checkpoint(folder/'best.pt');last_model,last=load_checkpoint(folder/'last.pt')
        result=dict(status='complete',completed_epochs=trainer.epoch,best_epoch=best['epoch'],
            best_checkpoint=str(folder/'best.pt'),last_checkpoint=str(folder/'last.pt'),
            checkpoint_sha256=sha(folder/'best.pt'),best_metrics=best['valid_metrics'],last_metrics=last['valid_metrics'],
            last_minus_best_foreground_miou=last['valid_score']-best['valid_score'],config=trainer.config,
            strict_five_class_load=True,initialization=probe)
        save_json(folder/'training_result.json',result)
        save_csv(folder/'best_last_metrics.csv',[dict(checkpoint=name,epoch=item['epoch'],**flatten(item['valid_metrics'])) for name,item in [('best',best),('last',last)]])
        save_json(run/'status.json',dict(stage_a='complete',stage_b='waiting_for_human_annotations',predictions='pending_frozen_target_split'))
        return result
    finally:
        for item in loaders:_close(item)
        del model,optimizer
        import gc
        gc.collect();torch.cuda.empty_cache()


def setup_run(root,run):
    run.mkdir(parents=True,exist_ok=True);path=run/'protocol.json'
    protocol=dict(config=CONFIG,stage_b_status='waiting_for_human_annotations',formal_reports_updated=False,
        target_validation_role='human-labeled validation for checkpoint selection; not an independent final test',
        method='Pseudo-label-assisted human annotation followed by target-domain fine-tuning; not a named self-training algorithm')
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Experiment protocol changed; use a new run')
    save_json(path,protocol)
    return protocol


def confusion5(prediction,target):
    if prediction.shape!=target.shape or not np.isin(prediction,range(5)).all() or not np.isin(target,ALLOWED).all():
        raise ValueError('Invalid five-class prediction/annotation coordinates or labels')
    matrix=np.zeros((5,5),dtype=np.int64)
    for y in range(0,len(target),128):
        truth=target[y:y+128];pred=prediction[y:y+128];valid=truth!=255
        matrix+=np.bincount(truth[valid].astype(np.int64)*5+pred[valid],minlength=25).reshape(5,5)
    return matrix


@torch.inference_mode()
def infer960(model,image,device='cuda:0',target=None):
    """Five float32 logits: actual 960 canvas, crop padding, restore, argmax."""
    from torchvision.transforms import functional as TF
    from experiment14 import prepare_input,restore_logits,forward_compact_head,MEAN,STD
    device=torch.device(device);started=perf_counter();seen=[]
    canvas,geometry=prepare_input(image,960)
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    model.to(device=device,dtype=torch.float32).eval()
    hook=model.stem.register_forward_pre_hook(lambda module,args:seen.append(list(args[0].shape)))
    try:
        inputs=TF.normalize(TF.to_tensor(canvas).float(),MEAN,STD).unsqueeze(0).to(device)
        with torch.autocast(device.type,enabled=False):logits=forward_compact_head(model,inputs)
        if seen!=[[1,3,960,960]] or logits.dtype!=torch.float32 or logits.shape[1]!=5:raise ValueError('Actual float32 960 input/logits mismatch')
        logits=restore_logits(logits,geometry,image.size)
        if not torch.isfinite(logits).all():raise ValueError('Non-finite logits')
        loss=None
        if target is not None and (target!=255).any():
            truth=torch.from_numpy(np.array(target,dtype=np.int64,copy=True)).unsqueeze(0).to(device)
            loss=float(torch.nn.functional.cross_entropy(logits,truth,ignore_index=255))
        prediction=logits.argmax(1)[0].to(dtype=torch.uint8,device='cpu').numpy().copy()
        if device.type=='cuda':torch.cuda.synchronize()
        stats=dict(actual_input_shape=seen[0],device=str(device),precision='float32',amp=False,tf32=False,
            geometry=geometry,inference_seconds=perf_counter()-started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else None)
        return prediction,stats,loss
    finally:hook.remove();canvas.close()


def display_prediction(image,prediction,folder):
    colors=np.array([[0,0,0],[255,65,77],[255,167,38],[0,190,220],[61,190,90]],np.uint8)
    color=colors[prediction];Image.fromarray(color).save(folder/'color.png')
    values=np.asarray(image).copy();visible=prediction>0
    values[visible]=(.5*values[visible]+.5*color[visible]).astype(np.uint8)
    overlay=Image.fromarray(values);overlay.save(folder/'overlay.jpg',quality=90)
    small=overlay.copy();small.thumbnail((1100,800));small.save(folder/'preview.jpg',quality=90)
    overlay.close();small.close()


def predict_stage_a(root,run):
    import hashlib,inspect,gc
    from PIL import __version__ as pillow_version
    from experiment14 import strict_float32,prepare_input,restore_logits,forward_compact_head
    from gallery import write_gallery_html
    split_path=run/'split_manifest.json'
    if not split_path.exists():raise PermissionError('Freeze the RGB-reviewed target split before predictions')
    split=json.loads(split_path.read_text())
    if not split['frozen']:raise PermissionError('Target split is not frozen')
    weights=run/'stage_a/best.pt';model,saved=load_checkpoint(weights);del saved
    checkpoint_sha=sha(weights);state_before=state_hash(model)
    code_sha=hashlib.sha256(''.join(inspect.getsource(f) for f in (infer960,prepare_input,restore_logits,forward_compact_head)).encode()).hexdigest()
    records=[];gallery=[]
    with strict_float32() as backend:
        for index,entry in enumerate(split['entries'],1):
            source=root/entry['source']
            if sha(source)!=entry['image_sha256']:raise ValueError('Frozen target image changed')
            folder=run/'predictions/stage_a'/entry['split']/entry['id'];folder.mkdir(parents=True,exist_ok=True)
            identity=dict(checkpoint_sha256=checkpoint_sha,image_sha256=entry['image_sha256'],split_sha256=sha(split_path),
                classes=list(CLASSES),target_size=960,precision='float32',amp=False,tf32=False,
                mean=[.485,.456,.406],std=[.229,.224,.225],padding=[124,116,104],backend=backend,
                torch_version=str(torch.__version__),pillow_version=pillow_version,cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),inference_code_sha256=code_sha)
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            record_path=folder/'record.json';mask_path=folder/'class_ids.png'
            if record_path.exists():
                cached=json.loads(record_path.read_text())
                if cached.get('identity_key')==key and mask_path.exists() and sha(mask_path)==cached['mask_sha256']:
                    records.append({**cached,'cache_reused':True})
                    if entry['split']=='train':gallery.append((entry['image'],folder/'preview.jpg'))
                    continue
            with Image.open(source) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
            print(f"PREDICT A {index}/167 | {entry['split']} | {entry['image']} | 960",flush=True)
            attempts=[]
            try:prediction,stats,_=infer960(model,image)
            except torch.OutOfMemoryError as error:
                attempts.append(dict(device='cuda:0',status='OOM',reason=str(error)))
                model.cpu();gc.collect();torch.cuda.empty_cache();prediction,stats,_=infer960(model,image,'cpu')
            attempts.append(dict(status='complete',**stats))
            if prediction.shape!=(entry['height'],entry['width']):raise ValueError('Original-coordinate mask mismatch')
            Image.fromarray(prediction).save(mask_path);display_prediction(image,prediction,folder);image.close()
            record=dict(id=entry['id'],image=entry['image'],split=entry['split'],identity=identity,identity_key=key,
                mask=str(mask_path),mask_sha256=sha(mask_path),color=str(folder/'color.png'),overlay=str(folder/'overlay.jpg'),
                preview=str(folder/'preview.jpg'),cache_reused=False,attempts=attempts,**stats)
            save_json(record_path,record);records.append(record)
            save_json(run/'predictions/stage_a/progress.json',records)
            if entry['split']=='train':gallery.append((entry['image'],folder/'preview.jpg'))
            gc.collect();torch.cuda.empty_cache()
    if state_hash(model)!=state_before or sha(weights)!=checkpoint_sha:raise ValueError('Fixed model weights changed during predictions')
    if sha(split_path)!=records[0]['identity']['split_sha256']:raise ValueError('Frozen split changed')
    html=write_gallery_html(run/'predictions/stage_a/training_gallery.html','Experiment 15 A: training-image annotation drafts',gallery,
        'Only target training predictions are shown. Background unchanged; penguin red, bird orange, seal cyan, whale green. These are unapproved drafts. Validation predictions are kept separate and are not annotation seeds.')
    result=dict(status='complete',images=len(records),train=sum(r['split']=='train' for r in records),valid=sum(r['split']=='valid' for r in records),
        checkpoint_sha256=checkpoint_sha,split_sha256=sha(split_path),inference_size=960,precision='float32',amp=False,tf32=False,
        records=records,training_gallery=str(html),validation_predictions='stored separately; never served as validation annotation seeds')
    save_json(run/'predictions/stage_a/result.json',result)
    save_csv(run/'predictions/stage_a/device_log.csv',[dict(image=r['image'],split=r['split'],device=r['device'],
        precision=r['precision'],actual_input_shape=r['actual_input_shape'],seconds=r['inference_seconds'],peak_allocated_GiB=r['peak_allocated_GiB']) for r in records])
    del model;gc.collect();torch.cuda.empty_cache()
    save_json(run/'status.json',dict(stage_a='complete',predictions='complete',annotation_workflow='archived',stage_b='not_run'))
    return result








def evaluate960(model,dataset,*,saved_records=None,output=None):
    from experiment14 import strict_float32
    matrix=np.zeros((5,5),dtype=np.int64);coverage=np.zeros(5,dtype=np.int64);rows=[]
    total_loss=0.;loss_pixels=0
    with strict_float32():
        for i in range(len(dataset)):
            image,mask=dataset.get_raw(i);target=np.asarray(mask);record=None
            if saved_records is not None:
                entry=dataset.records[i];record=saved_records[entry['id']]
                if sha(Path(record['mask']))!=record['mask_sha256'] or sha(dataset.root/entry['source'])!=record['identity']['image_sha256']:
                    raise ValueError('Saved prediction identity changed')
                with Image.open(record['mask']) as im:prediction=np.array(im)
                stats={};loss=None
            else:
                try:prediction,stats,loss=infer960(model,image,target=target)
                except torch.OutOfMemoryError:
                    model.cpu();torch.cuda.empty_cache();prediction,stats,loss=infer960(model,image,'cpu',target=target)
            cm=confusion5(prediction,target);matrix+=cm;coverage+=cm.sum(1)>0
            valid_pixels=int(cm.sum())
            if loss is not None:total_loss+=loss*valid_pixels;loss_pixels+=valid_pixels
            if output is not None:
                output=Path(output);output.mkdir(parents=True,exist_ok=True)
                name=dataset.records[i]['id'] if hasattr(dataset,'records') else f'img_{i:05d}'
                Image.fromarray(prediction).save(output/(name+'.png'))
            rows.append(dict(index=i,**flatten(dict(metrics5(cm),loss=loss)),**stats));image.close();mask.close()
    result=metrics5(matrix);result.update(loss=total_loss/loss_pixels if loss_pixels else None,num_images=len(dataset),
        image_coverage=dict(zip(CLASSES,coverage.tolist())),evaluation_size=960,precision='float32',amp=False,tf32=False)
    if output is not None:save_json(output/'metrics.json',result);save_csv(output/'image_metrics.csv',rows)
    return result












def run_experiment15(*,root=None,run_dir=None,train_a=False,predict=False):
    """Reproduce the completed source baseline; foreground adaptation was not run."""
    if not train_a and not predict:return dict(status='prepared',config=CONFIG,stage_b='not_run')
    if run_dir is None:raise ValueError('An explicit run directory is required')
    root=Path(root or PROJECT_ROOT).resolve();run=Path(run_dir).resolve()
    setup_run(root,run);result={}
    if train_a:result['stage_a']=train_stage_a(root,run)
    if predict:result['predictions']=predict_stage_a(root,run)
    result['report']=str(root/'reports/exp15/README.md')
    save_json(root/'artifacts/experiments/exp15_results/latest.json',dict(folder=str(run),status='source_baseline',stage_b_executed=False))
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--train-a',action='store_true');parser.add_argument('--predict',action='store_true')
    args=parser.parse_args();run_experiment15(run_dir=args.run_dir,train_a=args.train_a,predict=args.predict)
