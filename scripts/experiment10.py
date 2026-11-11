"""Experiment 10: 5x5, ten epochs, five equally weighted sampling groups.

Same binary initialization/new five-class head and optimizer as experiment 8.
Original animal validation selects checkpoints; background validation is separate.
No training on import. LLM-assisted, 2026-11-10.
"""

from project_paths import PROJECT_ROOT
from gallery import write_gallery_html
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from animal_data import CLASS_NAMES
from script import ResNetUNet, evaluate_background
from train_multiclass import (experiment_data, experiment_folder, new_run, note,
    write_json, train_run, prepare_runtime, make_loader, close_loader, cleanup_cuda)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare_experiment10(*,root=None,run_dir=None,epochs=10):
    root=Path(root or PROJECT_ROOT).resolve()
    folder=experiment_folder(root,10)
    run=Path(run_dir).resolve() if run_dir else new_run(folder)
    run.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((root/'artifacts/experiments/exp8_results/latest_kernel5.json').read_text(encoding='utf-8'))['selected']
    config=baseline['config']
    if baseline['completed_epochs']!=10 or config['kernel_size']!=5:
        raise ValueError('Expected the completed ten-epoch 5x5 baseline')
    data,metadata=experiment_data(str(root),640,True)
    old=json.loads((Path(baseline['folder'])/'data.json').read_text(encoding='utf-8'))
    for split in ('train','valid','test'):
        for source,count in old['sources'][split].items():
            if metadata['sources'][split][source]!=count:
                raise ValueError('Original animal sample count changed')
        for group,count in old['class_samples'][split].items():
            if metadata['class_samples'][split][group]!=count:
                raise ValueError('Original animal sampling groups changed')
    for key,value in old['annotation_manifest_sha256'].items():
        if metadata['annotation_manifest_sha256'][key]!=value:
            raise ValueError(f'Original animal annotation changed: {key}')
    classes=[r['sample_class'] for d in data['train'].datasets for r in d.records]
    frequencies=Counter(classes)
    if set(frequencies)!=set(range(5)):
        raise ValueError('All five sampling groups are required')
    weights=torch.tensor([1/frequencies[c] for c in classes],dtype=torch.float64)
    probability=weights/weights.sum()
    masses={CLASS_NAMES[c]:float(probability[torch.tensor(classes)==c].sum()) for c in range(5)}
    if any(abs(value-.2)>1e-12 for value in masses.values()):
        raise ValueError('Sampling probabilities are not 20% per group')
    samples=sum(old['sources']['train'].values())
    request=dict(experiment=10,epochs=epochs,kernel_size=5,seed=42,
        batch_size=config['batch_size'],bn_policy=config['bn_policy'],amp=True,
        channels_last=config['channels_last'],learning_rate=1e-4,image_size=640,
        samples_per_epoch=samples,sampling_group_probability=masses,
        sampling='WeightedRandomSampler, replacement=True; expected proportions, not exact batch quotas',
        initial_checkpoint='artifacts/runs/penguin_baseline/best.pt',fresh_five_class_head=True,
        initial_checkpoint_sha256=file_hash(root/'artifacts/runs/penguin_baseline/best.pt'),
        baseline_checkpoint=baseline['checkpoint'],baseline_checkpoint_sha256=file_hash(baseline['checkpoint']),
        checkpoint_selection='foreground_miou on unchanged animal validation only',
        annotation_manifest_sha256=metadata['annotation_manifest_sha256'])
    request_path=run/'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text(encoding='utf-8'))!=request:
            raise ValueError('Existing run request differs; choose a new run_dir')
    else:
        write_json(request_path,request)
    write_json(run/'data_check.json',dict(status='passed',metadata=metadata,
        sampling_group_counts={CLASS_NAMES[c]:frequencies[c] for c in range(5)},
        sampler_probability_mass=masses,original_animal_annotations_and_counts_unchanged=True))
    if not (run/'source_hashes.json').exists():
        write_json(run/'source_hashes.json',{name:file_hash(root/'scripts'/name) for name in
            ('script.py','animal_data.py','train_multiclass.py','experiment10.py')})
    return run,baseline,request


def run_experiment10(*,root=None,run_dir=None,epochs=10):
    root=Path(root or PROJECT_ROOT).resolve()
    run,baseline,request=prepare_experiment10(root=root,run_dir=run_dir,epochs=epochs)
    if (run/'result.json').exists():
        existing=json.loads((run/'result.json').read_text(encoding='utf-8'))
        if existing.get('status')=='complete':
            print('Experiment 10 already complete; reusing:',run,flush=True)
            return existing
    prepare_runtime()
    folder=experiment_folder(root,10)
    note(folder, "Run status updated. See structured run metadata.")
    print('EXPERIMENT 10 | 5x5 | five groups x 20% | epochs:',epochs,
          '| samples/epoch:',request['samples_per_epoch'],flush=True)
    print('Background train/valid/test:',experiment_data(str(root),640,True)[1]['sources'],flush=True)
    try:
        selected=train_run(root,run/'kernel_5',epochs=epochs,kernel_size=5,
            bn_policy=request['bn_policy'],batch_size=request['batch_size'],
            channels_last=request['channels_last'],seed=request['seed'],
            include_background=True,samples_per_epoch=request['samples_per_epoch'],log_every=0)
    except Exception as error:
        note(folder, "Run status updated. See structured run metadata.")
        raise
    result=dict(experiment=10,status='complete',folder=str(run),selected=selected,
                baseline=baseline,request=request)
    write_json(run/'result.json',result);write_json(folder/'latest.json',result)
    note(folder, "Run status updated. See structured run metadata.")
    return result


def evaluate_experiment10(result=None,*,root=None):
    """Compare old/new checkpoints on the same background holdout; no training."""
    root=Path(root or PROJECT_ROOT).resolve();prepare_runtime()
    if result is None:
        result=json.loads((root/'artifacts/experiments/exp10_results/latest.json').read_text(encoding='utf-8'))
    run=Path(result['folder']);selected=result['selected'];baseline=result['baseline']
    data,_=experiment_data(str(root),640,True)
    background=next(d for d in data['valid'].datasets if d.root.name=='dataset_nature')
    loader=make_loader(background,6,workers=2)
    rows=[];models={};group_rows=[]
    try:
        for name,source in [('5x5_before',baseline),('5x5_background20',selected)]:
            checkpoint=torch.load(source['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
            model=ResNetUNet(num_classes=5,pretrained=False,kernel_size=5)
            model.load_state_dict(checkpoint['model_state']);del checkpoint
            model=model.cuda().eval()
            bg=evaluate_background(model,loader,'cuda:0',amp=True)
            # Report limited holdout coverage explicitly (rock_ground has four
            # images); the pooled rate alone can hide different backgrounds.
            from torch.utils.data import Subset
            for group in sorted({r['group'] for r in background.records}):
                indices=[i for i,r in enumerate(background.records) if r['group']==group]
                grouped=make_loader(Subset(background,indices),6,workers=0)
                group_metrics=evaluate_background(model,grouped,'cuda:0',amp=True)
                group_rows.append(dict(model=name,group=group,**group_metrics))
            cm=np.asarray(source['metrics']['confusion_matrix'])
            rows.append(dict(model=name,best_epoch=source['best_epoch'],
                foreground_miou=source['metrics']['foreground_miou'],
                animal_validation_background_fp_fraction=float(cm[0,1:].sum()/cm[0].sum()),
                foreground_macro_precision=source['metrics']['precision'],
                foreground_macro_recall=source['metrics']['recall'],
                **{CLASS_NAMES[c]+'_iou':source['metrics']['iou_per_class'][c] for c in range(1,5)},
                **{'nature_'+key:value for key,value in bg.items()}))
            models[name]=model.cpu();cleanup_cuda()
        table=pd.DataFrame(rows);table.to_csv(run/'validation_comparison.csv',index=False)
        display(table[['model','best_epoch','foreground_miou','foreground_macro_precision',
            'foreground_macro_recall','animal_validation_background_fp_fraction',
            'nature_false_positive_fraction']].round(5))
        display(table[['model',*[CLASS_NAMES[c]+'_iou' for c in range(1,5)]]].round(5))
        group_table=pd.DataFrame(group_rows)
        group_table.to_csv(run/'background_group_comparison.csv',index=False)
        display(group_table[['model','group','num_images','false_positive_fraction']].round(5))
        fig,ax=plt.subplots(figsize=(9,4))
        groups=sorted(group_table['group'].unique());x=np.arange(len(groups))
        for offset,name in [(-.18,'5x5_before'),(.18,'5x5_background20')]:
            values=[100*next(r['false_positive_fraction'] for r in group_rows if r['model']==name and r['group']==g) for g in groups]
            ax.bar(x+offset,values,width=.34,label=name)
        ax.set(xticks=x,xticklabels=[f"{g}\n(n={sum(r['group']==g for r in background.records)})" for g in groups],
            ylabel='False foreground pixels (%)',title='Nature validation by dominant background group')
        ax.legend();fig.tight_layout();fig.savefig(run/'background_group_comparison.png',dpi=140);plt.show();plt.close(fig)
        _background_figures(root,run,selected,background,models)
        comparison=dict(status='complete',folder=str(run),rows=rows,
            animal_validation_unchanged=True,background_holdout_images=len(background),background_groups=group_rows)
        write_json(run/'evaluation.json',comparison)
        write_json(experiment_folder(root,10)/'latest_evaluation.json',comparison)
        note(experiment_folder(root,10), "Run status updated. See structured run metadata.")
        return comparison
    finally:
        close_loader(loader);models.clear();cleanup_cuda()


def _background_figures(root,run,selected,background,models):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    history=json.loads((Path(selected['folder'])/'history.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epochs=[row['epoch'] for row in history]
    axes[0].plot(epochs,[row['foreground_miou'] for row in history],'o-')
    axes[0].set(title='Unchanged animal validation: foreground mIoU',xlabel='Epoch',ylim=(0,1))
    axes[1].plot(epochs,[100*row['valid_metrics']['background_holdout']['false_positive_fraction'] for row in history],'o-')
    axes[1].set(title='Nature validation: false animal pixels',xlabel='Epoch',ylabel='Non-ignored background pixels (%)')
    fig.tight_layout();fig.savefig(run/'background_validation_curves.png',dpi=140);plt.show();plt.close(fig)
    choices={}
    for i,record in enumerate(background.records):choices.setdefault(record['group'],i)
    palette=['black','red','orange','deepskyblue','limegreen']
    for group,index in sorted(choices.items()):
        image,target=background[index];valid=target.numpy()==0
        rgb=(image*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]).clamp(0,1).permute(1,2,0)
        predictions=[]
        for model in models.values():
            model.cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                predictions.append(model(image.unsqueeze(0).cuda()).argmax(1)[0].cpu().numpy())
            model.cpu();cleanup_cuda()
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        for ax in axes:ax.imshow(rgb);ax.axis('off')
        axes[1].imshow(np.where(valid,0,1),cmap=ListedColormap(['limegreen','gray']),vmin=0,vmax=1,alpha=.45)
        for ax,prediction in zip(axes[2:],predictions):
            visible=(prediction!=0)&valid
            ax.imshow(np.where(visible,prediction,0),cmap=ListedColormap(palette),vmin=0,vmax=4,alpha=.5*visible)
        for ax,title in zip(axes,[group,'Labeled background=green; ignore=gray','Before: animal errors','After: animal errors']):ax.set_title(title)
        fig.legend(handles=[Patch(color=palette[c],label=CLASS_NAMES[c]) for c in range(1,5)],loc='lower center',ncol=4)
        fig.tight_layout(rect=(0,.07,1,1));fig.savefig(run/f'background_{group}.png',dpi=110);plt.show();plt.close(fig)


def visualize_experiment10(result=None,*,root=None):
    """Same nine native/resize/union inputs as experiment 9, old/new rows.

    Reuse completed inference on rerun. This never changes experiment-9 indices.
    Course images and the all-nine gallery stay under the ignored private tree.
    """
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    from IPython.display import Image as DisplayImage
    from experiment9 import run_experiment9
    root=Path(root or PROJECT_ROOT).resolve()
    folder=experiment_folder(root,10)
    if result is None:
        result=json.loads((folder/'latest.json').read_text(encoding='utf-8'))
    selected=result['selected']
    before=json.loads((root/'artifacts/experiments/exp9_results/latest_kernel5.json').read_text(encoding='utf-8'))
    if Path(before['checkpoint'])!=Path(result['baseline']['checkpoint']):
        raise ValueError('Before-photo checkpoint differs from the training baseline')
    manifests=json.loads((root/'artifacts/experiments/exp8_results/run_20261110_073349_055995/comparison_input_manifest.json').read_text(encoding='utf-8'))
    paths=[];aliases={};resolved=[]
    for item in manifests:
        original=Path(item['image']);path=original
        if not path.exists():
            # The user may rename an example while training. Match bytes,
            # preserving that rename and the exact original comparison input.
            matches=[p for p in sorted(original.parent.iterdir()) if p.is_file()
                and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} and file_hash(p)==item['sha256']]
            if not matches:
                raise FileNotFoundError(f'No byte-identical comparison photo found: {original}')
            path=matches[0]
        if file_hash(path)!=item['sha256']:
            raise ValueError(f"Photo changed since experiment 9: {path}")
        paths.append(path);aliases[path.name]=original.name
        resolved.append(dict(original_path=str(original),current_path=str(path),sha256=item['sha256']))
    write_json(Path(result['folder'])/'photo_input_manifest.json',resolved)
    index=folder/'latest_inference.json'
    after=json.loads(index.read_text(encoding='utf-8')) if index.exists() else None
    if not after or Path(after['checkpoint'])!=Path(selected['checkpoint']) or after['status']!='complete':
        after=run_experiment9(result,root=root,image_paths=paths,experiment_number=10,
            source_experiment=10,index_stem='latest_inference',display_private=False,display_examples=False)
        after['source_image_aliases']=aliases
        write_json(Path(after['folder'])/'result.json',after)
        write_json(index,after);write_json(folder/'latest_inference_kernel5.json',after)
    if after['status']!='complete' or len(after['records'])!=9:
        raise RuntimeError('Not all nine original-resolution photo predictions completed')
    run=Path(after['folder'])
    private=root/'data/nju_testset/predictions/exp10'/run.name/'before_after'
    private.mkdir(parents=True,exist_ok=True)
    old={(r['group'],r['image']):r for r in before['records']}
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),30)
    statistics=[];comparison_paths=[]
    for new in after['records']:
        original_name=after.get('source_image_aliases',aliases).get(new['image'],new['image'])
        previous=old[(new['group'],original_name)]
        if previous['native']['original_size']!=new['native']['original_size']:
            raise ValueError('Before/after photo dimensions differ')
        panels=[]
        for source in (previous,new):
            with Image.open(source['comparison_path']) as image:panels.append(image.convert('RGB'))
        canvas=Image.new('RGB',(max(p.width for p in panels),sum(p.height+50 for p in panels)),'white')
        draw=ImageDraw.Draw(canvas);top=0
        for title,panel in zip(('Before | 5x5 | four animal sampling groups','After | 5x5 | background + four animal groups'),panels):
            draw.text((12,top+6),title,font=font,fill='black');canvas.paste(panel,(0,top+50));top+=panel.height+50
        target=private if new['group']=='nju_testset' else run/'before_after_examples'
        target.mkdir(exist_ok=True)
        path=target/(Path(new['image']).stem+'_before_after.png');canvas.save(path);comparison_paths.append(str(path))
        if new['group']=='examples':display(DisplayImage(filename=str(path)))
        for label,source in (('before',previous),('after',new)):
            row=dict(group=new['group'],image=new['image'],model=label,native_device=source['native']['device'],native_seconds=source['native']['seconds'])
            for branch in ('native','resize','union'):
                with Image.open(Path(source['output_folder'])/(Path(source['image']).stem+'_'+branch+'.png')) as mask:
                    row[branch+'_foreground_fraction']=float((np.asarray(mask)>0).mean())
            statistics.append(row)
        for panel in panels:panel.close()
        canvas.close()
    gallery=write_gallery_html(private/'before_after.html','Before and after',[(Path(p).stem,p) for p in comparison_paths])
    pd.DataFrame(statistics).to_csv(run/'photo_foreground_areas.csv',index=False)
    comparison=dict(status='complete',type='before_after',folder=str(run),
        before_checkpoint=before['checkpoint'],after_checkpoint=after['checkpoint'],
        photo_count=9,private_gallery=str(gallery),comparison_paths=comparison_paths,
        inference_gallery=after['private_gallery'],foreground_area_is_not_accuracy=True)
    write_json(run/'photo_comparison.json',comparison);write_json(folder/'latest_comparison.json',comparison)
    note(folder, "Run status updated. See structured run metadata.")
    print('Experiment 10 all-nine before/after gallery:',gallery,flush=True)
    return comparison
