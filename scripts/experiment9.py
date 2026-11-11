"""Experiment 9: native/letterbox-resize inference and multiclass foreground union.

No training. Course-image outputs are routed to the Git-ignored nju_testset tree.
LLM-assisted, 2026-11-09.
The final affine 1x1 head is evaluated before final bilinear interpolation:
these linear operators commute, avoiding a full-resolution 64-channel tensor.
"""

from project_paths import PROJECT_ROOT
from gallery import write_gallery_html
import gc
import json
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image,ImageOps
import psutil
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from animal_data import CLASS_NAMES,fit_image_and_mask
from script import ResNetUNet
from train_multiclass import prepare_runtime,experiment_folder,new_run,note,write_json,cleanup_cuda

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def forward_compact_head(model,x):
    """Same learned model; commute only the final 1x1 head and interpolation."""
    size=x.shape[-2:]
    s0=model.stem(x)
    s1=model.enc1(model.pool(s0))
    s2=model.enc2(s1);s3=model.enc3(s2);deep=model.enc4(s3)
    d4=model.dec4(deep,s3);d3=model.dec3(d4,s2)
    d2=model.dec2(d3,s1);d1=model.dec1(d2,s0)
    return F.interpolate(model.head(d1),size=size,mode='bilinear',align_corners=False)


@torch.inference_mode()
def verify_compact_graph(model):
    """Record parity on the actual trained checkpoint, including AMP rounding."""
    generator=torch.Generator().manual_seed(42)
    x=torch.randn(1,3,127,159,generator=generator)
    model.cpu().eval()
    reference=model(x);compact=forward_compact_head(model,x)
    error=(reference-compact).abs().max().item()
    if not torch.allclose(reference,compact,atol=1e-3,rtol=1e-4):
        raise RuntimeError(f'Compact inference graph failed float32 parity: {error}')
    model.cuda();x=x.cuda()
    with torch.autocast('cuda',dtype=torch.float16):
        reference=model(x);compact=forward_compact_head(model,x)
    return dict(float32_max_absolute_error=error,
        amp_max_absolute_error=(reference-compact).abs().max().item(),
        amp_argmax_disagreement_fraction=(reference.argmax(1)!=compact.argmax(1)).float().mean().item(),
        verification_input=[1,3,127,159])


@torch.inference_mode()
def infer_branch(image,model,device,*,resize=False,channels_last=False):
    device=torch.device(device)
    model=model.to(device).eval()
    if channels_last:model.to(memory_format=torch.channels_last)
    if device.type=='cuda':
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=perf_counter()
    width,height=image.size
    if resize:
        working,_=fit_image_and_mask(image,Image.new('L',image.size),640)
    else:
        working=image
    inputs=TF.normalize(TF.to_tensor(working),MEAN,STD).unsqueeze(0).to(device)
    if channels_last:inputs=inputs.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        logits=forward_compact_head(model,inputs)
    if resize:
        scale=min(640/width,640/height)
        w,h=max(1,round(width*scale)),max(1,round(height*scale))
        left,top=(640-w)//2,(640-h)//2
        logits=F.interpolate(logits[:,:,top:top+h,left:left+w].float(),size=(height,width),mode='bilinear',align_corners=False)
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite inference logits')
    probabilities=logits.float().softmax(dim=1)[0].to(dtype=torch.float16,device='cpu').numpy()
    if device.type=='cuda':torch.cuda.synchronize()
    statistics=dict(device=str(device),seconds=perf_counter()-start,
        input_size=list(working.size),original_size=[width,height],
        precision='AMP float16' if device.type=='cuda' else 'float32',
        score_storage='float16 CPU; softmax calculated in float32',
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else None,
        inference_graph='linear 1x1 head before final interpolation')
    return probabilities,statistics


def foreground_union(native,scaled):
    """Retain foreground from either branch; resolve class conflicts by own score."""
    if native.shape!=scaled.shape or native.ndim!=3:
        raise ValueError('Both probability maps must have equal [C,H,W] shapes')
    a,b=native.argmax(0).astype(np.uint8),scaled.argmax(0).astype(np.uint8)
    score_a=np.take_along_axis(native,a[None],axis=0)[0]
    score_b=np.take_along_axis(scaled,b[None],axis=0)[0]
    choose_a=(a>0)&((b==0)|(score_a>=score_b))
    fused=np.where(choose_a,a,b).astype(np.uint8)
    union=(a>0)|(b>0)
    if not np.array_equal(fused>0,union):
        raise AssertionError('Fusion lost pixels from the foreground union')
    conflict=(a>0)&(b>0)&(a!=b)
    return a,b,fused,dict(foreground_union_pixels=int(union.sum()),
        conflict_pixels=int(conflict.sum()),conflict_fraction=float(conflict.mean()),
        foreground_fraction=float(union.mean()))


def show_comparison(image,masks,folder,stem,statistics,*,display=True):
    preview=image.copy();preview.thumbnail((1500,1500),Image.Resampling.LANCZOS)
    palette=['mediumpurple','red','orange','deepskyblue','limegreen']
    fig,axes=plt.subplots(1,4,figsize=(22,6))
    for ax in axes:ax.imshow(preview);ax.axis('off')
    axes[0].set_title('Original')
    titles=[f"Native | {statistics['native']['device']}",'Resize to 640 + restore','Foreground union (conflicts resolved)']
    for ax,mask,title in zip(axes[1:],masks,titles):
        values=np.asarray(Image.fromarray(mask).resize(preview.size,Image.Resampling.NEAREST))
        ax.imshow(values,cmap=ListedColormap(palette),vmin=0,vmax=4,alpha=np.where(values==0,.30,.5))
        ax.set_title(title)
    fig.legend(handles=[Patch(color=palette[c],label=CLASS_NAMES[c]) for c in range(5)],loc='lower center',ncol=4)
    kernel=statistics.get('kernel_size','?')
    fig.suptitle(f'{stem} | kernel {kernel}x{kernel} | {image.width} x {image.height} | no ground-truth mask')
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(folder/f'{stem}_comparison.png',dpi=140,bbox_inches='tight')
    if display:
        plt.show()
    plt.close(fig)


def save_overlay(image,mask,path):
    """Full-resolution colored overlay for zooming; class-ID masks stay separate."""
    colors=np.array([[0,0,0],[255,0,0],[255,165,0],[0,191,255],[50,205,50]],dtype=np.uint8)
    color_layer=Image.fromarray(colors[mask])
    alpha=Image.fromarray(np.where(mask>0,115,0).astype(np.uint8))
    Image.composite(color_layer,image,alpha).save(path,quality=94,subsampling=0)


def save_private_gallery(root,run,records,checkpoint_path,*,kernel_size,experiment_number=9):
    """Reference saved comparisons from a private local HTML gallery."""
    private=Path(root)/'data/nju_testset/predictions'/f'exp{experiment_number}'/run.name
    items=[(r['group']+' / '+r['image'],r['comparison_path']) for r in records if r['status']=='complete']
    return write_gallery_html(private/'predictions.html',f'Experiment {experiment_number}: {kernel_size}x{kernel_size}',items,
        f'Checkpoint: {checkpoint_path}. Background is purple. Photos have no independent evaluation masks.')


def run_experiment9(kernel_result,*,root=None,image_paths=None,display_private=False,
                    experiment_number=9,source_experiment=None,index_stem='latest',display_examples=True):
    root=Path(root or PROJECT_ROOT).resolve();prepare_runtime()
    folder=experiment_folder(root,experiment_number);run=new_run(folder)
    selected=kernel_result['selected'];checkpoint_path=Path(selected['checkpoint'])
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    kernel=checkpoint['config']['kernel_size']
    model=ResNetUNet(num_classes=5,pretrained=False,kernel_size=kernel)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    layout=checkpoint['config'].get('channels_last',False)
    del checkpoint
    graph_parity=verify_compact_graph(model)
    write_json(run/'graph_parity.json',graph_parity)
    print('Inference graph verification:',graph_parity,flush=True)
    paths=image_paths or [root/'data/examples/penguins_far.jpg',root/'data/examples/penguins_near.jpg',root/'data/examples/penguin_colony.jpg']
    source_experiment=source_experiment or (7 if kernel==3 else 8)
    note(folder, "Run status updated. See structured run metadata.")
    records=[]
    for path in map(Path,paths):
        path=path.resolve()
        is_private=path.is_relative_to(root/'data/nju_testset')
        group='nju_testset' if is_private else 'examples'
        output_folder=(root/'data/nju_testset'/'predictions'/f'exp{experiment_number}'/run.name) if is_private else run/'examples'
        output_folder.mkdir(parents=True,exist_ok=True)
        with Image.open(path) as source:image=ImageOps.exif_transpose(source).convert('RGB')
        print(f'EXPERIMENT {experiment_number} | {path.name} | native {image.size}',flush=True)
        native=None;native_statistics=None;gpu_oom=False
        try:
            native,native_statistics=infer_branch(image,model,'cuda:0',channels_last=layout)
        except torch.OutOfMemoryError:
            gpu_oom=True
        cleanup_cuda()
        if gpu_oom:
            model.cpu();cleanup_cuda()
            required=image.width*image.height*350+sum(p.numel()*p.element_size() for p in model.parameters())*3+2**30
            available=psutil.virtual_memory().available
            note(folder, "Run status updated. See structured run metadata.")
            if required>available*.85:
                record=dict(image=path.name,group=group,status='native_gpu_oom_cpu_memory_insufficient',original_size=list(image.size))
                records.append(record);write_json(run/'results.json',records)
                note(folder, "Run status updated. See structured run metadata.")
                continue
            print('  Native GPU OOM; explicit CPU float32 fallback, original pixels retained.',flush=True)
            native,native_statistics=infer_branch(image,model,'cpu',channels_last=False)
        print('  Native complete:',native_statistics,flush=True)
        scaled,scaled_statistics=infer_branch(image,model,'cuda:0',resize=True,channels_last=layout)
        a,b,fused,fusion_statistics=foreground_union(native,scaled)
        for name,mask in [('native',a),('resize',b),('union',fused)]:
            Image.fromarray(mask).save(output_folder/f'{path.stem}_{name}.png')
        statistics=dict(image=path.name,group=group,kernel_size=kernel,status='complete',native=native_statistics,
                        resize=scaled_statistics,fusion=fusion_statistics,native_gpu_oom=gpu_oom,
                        output_folder=str(output_folder),comparison_path=str(output_folder/f'{path.stem}_comparison.png'))
        records.append(statistics);write_json(run/'results.json',records)
        show_comparison(image,[a,b,fused],output_folder,path.stem,statistics,
                        display=display_private if is_private else display_examples)
        note(folder, "Run status updated. See structured run metadata.")
        del native,scaled,a,b,fused,image;cleanup_cuda()
    gallery=save_private_gallery(root,run,records,checkpoint_path,kernel_size=kernel,experiment_number=experiment_number)
    summary=dict(experiment=experiment_number,folder=str(run),checkpoint=str(checkpoint_path),records=records,
                 graph_parity=graph_parity,
                 kernel_size=kernel,source_experiment=source_experiment,
                 private_gallery=str(gallery),
                 status='complete' if all(r['status']=='complete' for r in records) else 'partial')
    write_json(run/'result.json',summary);write_json(folder/f'{index_stem}.json',summary)
    write_json(folder/f'{index_stem}_kernel{kernel}.json',summary)
    print('All-image local prediction gallery:',gallery,flush=True)
    note(folder, "Run status updated. See structured run metadata.")
    del model;cleanup_cuda()
    return summary


def _kernel_comparison_inputs(root):
    """Read explicit per-kernel indices so latest.json cannot select a wrong model."""
    root=Path(root)
    training={3:json.loads((root/'artifacts/experiments/exp7_results/latest.json').read_text(encoding='utf-8'))['selected']}
    for kernel in (5,7):
        training[kernel]=json.loads((root/f'artifacts/experiments/exp8_results/latest_kernel{kernel}.json').read_text(encoding='utf-8'))['selected']
    inference={kernel:json.loads((root/f'artifacts/experiments/exp9_results/latest_kernel{kernel}.json').read_text(encoding='utf-8'))
               for kernel in (3,5,7)}
    for kernel in (3,5,7):
        assert training[kernel]['config']['kernel_size']==kernel
        assert training[kernel]['completed_epochs']==10
        assert Path(inference[kernel]['checkpoint'])==Path(training[kernel]['checkpoint'])
        assert inference[kernel]['status']=='complete' and len(inference[kernel]['records'])==9
    return training,inference


def compare_kernel_runs(*,root=None):
    """Archive a matched three-kernel report/gallery without changing predictions.

    Course-derived images remain local-only. Foreground area is not accuracy.
    LLM-assisted analysis, 2026-11-10.
    """
    from datetime import datetime
    import csv
    import pandas as pd
    from PIL import ImageDraw,ImageFont
    from matplotlib import font_manager
    from IPython.display import display,Image as DisplayImage
    root=Path(root or PROJECT_ROOT).resolve()
    training,inference=_kernel_comparison_inputs(root)
    base=experiment_folder(root,9)
    run=base/('comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir()
    private=root/'data/nju_testset/predictions/exp9'/run.name
    private.mkdir(parents=True)
    reference=training[3]
    reference_data=json.loads((Path(reference['folder'])/'data.json').read_text(encoding='utf-8'))
    rows=[]
    for kernel,result in training.items():
        for key in ('batch_size','bn_policy','amp','seed','learning_rate','image_size'):
            assert result['config'][key]==reference['config'][key],key
        metadata=json.loads((Path(result['folder'])/'data.json').read_text(encoding='utf-8'))
        assert metadata['annotation_manifest_sha256']==reference_data['annotation_manifest_sha256']
        assert [sum(row) for row in result['metrics']['confusion_matrix']]==[sum(row) for row in reference['metrics']['confusion_matrix']]
        checkpoint=torch.load(result['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        parameters=sum(t.numel() for name,t in checkpoint['model_state'].items() if name.endswith(('.weight','.bias')))
        del checkpoint
        confusion=np.asarray(result['metrics']['confusion_matrix'],dtype=np.int64)
        fp=int(confusion[0,1:].sum());tn=int(confusion[0,0])
        fn=int(confusion[1:,0].sum());tp=int(confusion[1:,1:].sum())
        row=dict(kernel=kernel,best_epoch=result['best_epoch'],foreground_miou=result['metrics']['foreground_miou'],
                 parameters=parameters,training_validation_minutes=result['seconds']/60,
                 peak_training_allocated_GiB=result['peak_allocated_GiB'],
                 validation_background_fp_fraction=fp/(fp+tn),
                 validation_binary_foreground_precision=tp/(tp+fp),
                 validation_binary_foreground_recall=tp/(tp+fn),
                 validation_macro_class_precision=result['metrics']['precision'],
                 validation_macro_class_recall=result['metrics']['recall'],
                 **{name+'_iou':result['metrics']['iou_per_class'][i] for i,name in enumerate(CLASS_NAMES)})
        rows.append(row)
    table=pd.DataFrame(rows)
    display(table[['kernel','best_epoch','foreground_miou','validation_background_fp_fraction',
                   'validation_binary_foreground_precision','validation_binary_foreground_recall',
                   'training_validation_minutes','peak_training_allocated_GiB']].round(4))
    table.to_csv(run/'training_comparison.csv',index=False)
    fig,axes=plt.subplots(1,4,figsize=(20,4))
    for kernel,result in training.items():
        history=json.loads((Path(result['folder'])/'history.json').read_text(encoding='utf-8'))
        epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['foreground_miou'] for r in history],'o-',label=f'{kernel}x{kernel}')
        axes[1].plot(epochs,[r['valid_loss'] for r in history],'o-',label=f'{kernel}x{kernel}')
        x=np.arange(4)+(kernel-5)*.13
        axes[2].bar(x,result['metrics']['iou_per_class'][1:],width=.24,label=f'{kernel}x{kernel}')
    axes[0].set(title='Full validation foreground mIoU',xlabel='Epoch',ylim=(0,1))
    axes[1].set(title='Full validation loss',xlabel='Epoch')
    axes[2].set(title='Best checkpoint class IoU',ylim=(0,1),xticks=np.arange(4),xticklabels=CLASS_NAMES[1:])
    axes[2].tick_params(axis='x',rotation=25)
    axes[3].bar([f"{row['kernel']}x{row['kernel']}" for row in rows],
                [100*row['validation_background_fp_fraction'] for row in rows])
    axes[3].set(title='Labeled-validation background error',ylabel='Labeled-background pixels predicted as animal (%)')
    for ax in axes[:3]:ax.legend()
    fig.tight_layout();fig.savefig(run/'training_comparison.png',dpi=140,bbox_inches='tight')
    plt.show();plt.close(fig)
    records={k:{(r['group'],r['image']):r for r in value['records']} for k,value in inference.items()}
    keys=list(records[3])
    assert all(set(values)==set(keys) for values in records.values())
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),30)
    per_image=[];comparison_paths=[]
    for group,name in keys:
        panels=[]
        for kernel in (3,5,7):
            record=records[kernel][(group,name)]
            assert record['native']['original_size']==records[3][(group,name)]['native']['original_size']
            with Image.open(record['comparison_path']) as source:panels.append(source.convert('RGB'))
            folder=Path(record['output_folder']);stem=Path(name).stem
            areas=[]
            for branch in ('native','resize','union'):
                with Image.open(folder/f'{stem}_{branch}.png') as image:
                    assert list(image.size)==record['native']['original_size']
                    areas.append(float((np.asarray(image)>0).mean()))
            per_image.append(dict(group=group,image=name,kernel=kernel,native_device=record['native']['device'],
                native_precision=record['native']['precision'],native_seconds=record['native']['seconds'],
                resize_seconds=record['resize']['seconds'],native_foreground_fraction=areas[0],
                resize_foreground_fraction=areas[1],union_foreground_fraction=areas[2],
                class_conflict_fraction=record['fusion']['conflict_fraction']))
        width=max(p.width for p in panels);height=sum(p.height+50 for p in panels)
        canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);top=0
        for kernel,panel in zip((3,5,7),panels):
            draw.text((12,top+6),f'{kernel}x{kernel} | best epoch {training[kernel]["best_epoch"]}',font=font,fill='black')
            canvas.paste(panel,(0,top+50));top+=panel.height+50
        target=private if group=='nju_testset' else run/'examples'
        target.mkdir(parents=True,exist_ok=True)
        path=target/(Path(name).stem+'_kernels_comparison.png')
        canvas.save(path);comparison_paths.append(str(path))
        if group=='examples':display(DisplayImage(filename=str(path)))
        for panel in panels:panel.close()
        canvas.close()
    pd.DataFrame(per_image).to_csv(run/'image_comparison.csv',index=False)
    gallery=write_gallery_html(private/'kernel_comparison.html','Kernel comparison',[(Path(p).stem,p) for p in comparison_paths])
    result=dict(status='complete',type='kernel_comparison',folder=str(run),training=rows,
        source_inference_runs={str(k):d['folder'] for k,d in inference.items()},
        comparison_paths=comparison_paths,private_gallery=str(gallery),image_statistics=per_image,
        validation_annotation_hashes_match=True)
    write_json(run/'result.json',result);write_json(base/'latest_comparison.json',result)
    report='# Experiment 9: matched 3/5/7 comparison\n\n'
    report+='All structures trained for ten epochs with matched initialization/configuration and identical annotation hashes. '
    report+='Validation scores pool all labeled validation pixels. High-resolution photos have no masks; image areas below are not accuracy.\n\n'
    report+='Background FP fraction is FP/(FP+TN) after collapsing animal classes: numerator = pixels labeled background but predicted as any animal; denominator = all labeled background pixels. Binary foreground precision/recall ignore animal-to-animal class confusion, unlike per-class IoU. These are labeled-validation metrics, not high-resolution photo scores. Incomplete penguin annotations and approximate SealID silhouettes limit literal interpretation of the background labels.\n\n'
    report+='| kernel | best epoch | foreground mIoU | background FP % | foreground precision | foreground recall | parameters | train+valid minutes | peak allocated GiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:report+=f"| {row['kernel']} | {row['best_epoch']} | {row['foreground_miou']:.4f} | {100*row['validation_background_fp_fraction']:.3f} | {row['validation_binary_foreground_precision']:.4f} | {row['validation_binary_foreground_recall']:.4f} | {row['parameters']} | {row['training_validation_minutes']:.1f} | {row['peak_training_allocated_GiB']:.2f} |\n"
    report+=f'\nAll-nine-image local gallery: `{gallery}`. Native device/precision per image are in image_comparison.csv. No predictions were modified by the comparison.\n'
    (run/'summary.md').write_text(report,encoding='utf-8');note(base,report)
    print('Three-kernel local gallery:',gallery,flush=True)
    return result


@torch.inference_mode()
def run_background_probe(*,root=None,sizes=(640,1280,2560)):
    """Finite synthetic-negative diagnosis; no training or threshold selection.

    Every constructed input contains zero animals. Foreground fraction is thus
    a synthetic-background false-positive pixel fraction, not a real-photo rate.
    LLM-assisted, 2026-11-10.
    """
    from datetime import datetime
    import pandas as pd
    from IPython.display import display
    root=Path(root or PROJECT_ROOT).resolve();prepare_runtime()
    training,_=_kernel_comparison_inputs(root)
    base=experiment_folder(root,9)
    run=base/('background_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir()
    colors={'near_white':(245,245,245),'mid_gray':(128,128,128),
            'pale_sky':(190,205,220),'normalization_mean':(124,116,104)}
    rows=[];previews={}
    for kernel in (3,5,7):
        checkpoint=torch.load(training[kernel]['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        model=ResNetUNet(num_classes=5,pretrained=False,kernel_size=kernel)
        model.load_state_dict(checkpoint['model_state']);del checkpoint
        model=model.cuda().eval()
        for name,color in colors.items():
            for size in sizes:
                torch.cuda.reset_peak_memory_stats()
                image=Image.new('RGB',(size,size),color)
                inputs=TF.normalize(TF.to_tensor(image),MEAN,STD).unsqueeze(0).cuda()
                with torch.autocast('cuda',dtype=torch.float16):logits=forward_compact_head(model,inputs)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f'Non-finite synthetic-probe logits: kernel={kernel}, {name}, {size}')
                probability=logits.float().softmax(1)
                confidence,prediction=probability.max(1)
                foreground=prediction!=0
                fraction=foreground.float().mean().item()
                row=dict(kernel=kernel,color=name,rgb=list(color),size=size,
                    synthetic_false_positive_fraction=fraction,
                    false_positive_fraction_confidence_ge_09=(foreground&(confidence>=.9)).float().mean().item(),
                    false_positive_fraction_confidence_ge_099=(foreground&(confidence>=.99)).float().mean().item(),
                    mean_confidence_on_false_positives=confidence[foreground].mean().item() if foreground.any() else None,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                rows.append(row)
                if size==max(sizes):
                    values=prediction[0].to(dtype=torch.uint8,device='cpu').numpy()
                    preview_folder=run/'prediction_maps'
                    preview_folder.mkdir(exist_ok=True)
                    Image.fromarray(values).save(preview_folder/f'kernel{kernel}_{name}.png')
                    previews[(kernel,name)]=np.asarray(Image.fromarray(values).resize((320,320),Image.Resampling.NEAREST))
                print(f'Background probe | kernel={kernel} | {name} | {size} | false foreground={fraction:.3%}',flush=True)
                write_json(run/'partial_results.json',rows)
                del image,inputs,logits,probability,confidence,prediction,foreground
        del model;cleanup_cuda()
    table=pd.DataFrame(rows);table.to_csv(run/'background_probe.csv',index=False)
    display(table[['kernel','color','size','synthetic_false_positive_fraction','false_positive_fraction_confidence_ge_09']].round(4))
    fig,axes=plt.subplots(1,4,figsize=(17,4),sharey=True)
    for ax,name in zip(axes,colors):
        for kernel in (3,5,7):
            subset=table[(table.kernel==kernel)&(table.color==name)]
            ax.plot(subset['size'],100*subset['synthetic_false_positive_fraction'],'o-',label=f'{kernel}x{kernel}')
        ax.set(title=name,xlabel='Square input side',xticks=list(sizes),ylim=(0,100));ax.legend()
    axes[0].set_ylabel('Synthetic background pixels predicted as animal (%)')
    fig.tight_layout();fig.savefig(run/'background_probe.png',dpi=150,bbox_inches='tight')
    plt.show();plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(12,12))
    palette=ListedColormap(['black','red','orange','deepskyblue','limegreen'])
    for row,kernel in enumerate((3,5,7)):
        for col,name in enumerate(colors):
            axes[row,col].imshow(previews[(kernel,name)],cmap=palette,vmin=0,vmax=4)
            axes[row,col].set_title(f'{kernel}x{kernel} | {name}',fontsize=10,pad=9);axes[row,col].axis('off')
    fig.suptitle(f'All inputs contain no animals | side={max(sizes)} | correct mask is entirely black')
    fig.legend(handles=[Patch(color=color,label=name) for name,color in zip(CLASS_NAMES,
        ['black','red','orange','deepskyblue','limegreen'])],loc='lower center',ncol=5)
    fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.92,wspace=.08,hspace=.22)
    fig.savefig(run/'background_prediction_maps.png',dpi=140,bbox_inches='tight')
    plt.show();plt.close(fig)
    summary=dict(status='complete',type='synthetic_background_probe',folder=str(run),sizes=list(sizes),
        colors=colors,results=rows,checkpoints={str(k):r['checkpoint'] for k,r in training.items()},
        training_performed=False,scope='Synthetic known-empty canvases only; not real-photo false-positive rates')
    write_json(run/'result.json',summary);write_json(base/'latest_background_probe.json',summary)
    note(base, "Run status updated. See structured run metadata.")
    return summary
