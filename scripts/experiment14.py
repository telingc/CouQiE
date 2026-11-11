"""Fixed-weight float32 inference-scale diagnosis. Nothing runs on import.

Only target input size changes. No training, fusion, threshold tuning or edits
to benchmark_v2. English code/reporting; local-only NJU visualizations.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
import hashlib
import inspect
import json
import platform
import shutil
import textwrap

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, __version__ as PIL_VERSION
import psutil
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from project_paths import PROJECT_ROOT
from script import ResNetUNet
from experiment9 import forward_compact_head, MEAN, STD
from experiment13 import sha, save_json, save_csv, convert_labels, confusion4, photo_metrics
from gallery import write_gallery_html

PROTOCOL = 'exp14_fixed_weights_float32_v1'
CHECKPOINT_REL = 'artifacts/experiments/exp13_results/run_20261111_150108/best.pt'
POINTS_REL = 'artifacts/experiments/exp13_results/run_20261111_150108/far_point_protocol.json'
REFERENCE_REL = 'data/reference_masks/benchmark_v2/manifest.json'
SCALES = (640, 960, 1280, 1920, None)
LABELS = ('640', '960', '1280', '1920', 'Native')
CLASSES = ('background', 'penguin', 'other_bird', 'seal')
PAD_COLOR = (124, 116, 104)
REFERENCE_LEGEND = '640 x 640 input-size reference; not receptive field'
COLORS = np.array([[0,0,0],[255,65,77],[255,167,38],[0,190,220]], np.uint8)
NJU_NAMES = ('5N0A5174_1.jpg','5N0A5204_1.jpg','5N0A5965_1.jpg',
             '5N0A7706_1.jpg','5N0A7727_1.jpg','5N0A9908_1.jpg')
# Chosen from original RGB before viewing any experiment-14 predictions.
ROIS = {
    'penguins_near.jpg': {'head':[1950,900,3100,2050], 'chest':[2100,2100,3400,3300]},
    'penguins_far.jpg': {'distant_region':[0,2050,2300,2450]},
}
EXPECTED_ROI_IMAGES = {'penguins_near.jpg':[5184,3456], 'penguins_far.jpg':[6000,4000]}


def release():
    import gc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


@contextmanager
def strict_float32():
    """Disable autocast in forwards and TF32 in both CUDA GEMM and convolution."""
    old_threads=torch.get_num_threads();old_benchmark=torch.backends.cudnn.benchmark
    modern=hasattr(torch.backends.cuda.matmul,'fp32_precision') and hasattr(torch.backends.cudnn,'conv')
    if modern:
        old_m=torch.backends.cuda.matmul.fp32_precision
        old_c=torch.backends.cudnn.conv.fp32_precision
        torch.backends.cuda.matmul.fp32_precision='ieee'
        torch.backends.cudnn.conv.fp32_precision='ieee'
        settings=dict(api='fp32_precision',matmul='ieee',convolution='ieee')
    else:
        old_m=torch.backends.cuda.matmul.allow_tf32;old_c=torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        settings=dict(api='allow_tf32',matmul=False,convolution=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    try: yield settings
    finally:
        if modern:
            torch.backends.cuda.matmul.fp32_precision=old_m
            torch.backends.cudnn.conv.fp32_precision=old_c
        else:
            torch.backends.cuda.matmul.allow_tf32=old_m;torch.backends.cudnn.allow_tf32=old_c
        torch.backends.cudnn.benchmark=old_benchmark;torch.set_num_threads(old_threads)


def prepare_input(image, size):
    """Return actual RGB network canvas and reversible integer letterbox geometry."""
    width,height=image.size
    if size is None:
        return image,dict(canvas=[width,height],resized=[width,height],offset=[0,0],native=True)
    scale=min(size/width,size/height)
    w,h=max(1,round(width*scale)),max(1,round(height*scale))
    left,top=(size-w)//2,(size-h)//2
    canvas=Image.new('RGB',(size,size),PAD_COLOR)
    canvas.paste(image.resize((w,h),Image.Resampling.BILINEAR),(left,top))
    return canvas,dict(canvas=[size,size],resized=[w,h],offset=[left,top],native=False)


def restore_logits(logits, geometry, original_size):
    """Crop padding, interpolate continuous scores, then let the caller argmax."""
    if geometry['native']: return logits
    left,top=geometry['offset'];w,h=geometry['resized']
    width,height=original_size
    return F.interpolate(logits[:,:,top:top+h,left:left+w],size=(height,width),
                         mode='bilinear',align_corners=False)


def input_preview(working):
    """Draw only on a resized visualization copy, never on network RGB pixels."""
    preview=working.copy();preview.thumbnail((420,280))
    x=min(640,working.width)*preview.width/working.width
    y=min(640,working.height)*preview.height/working.height
    ImageDraw.Draw(preview).rectangle((0,0,max(0,round(x)-1),max(0,round(y)-1)),
                                     outline=(0,230,60),width=3)
    return preview


@torch.inference_mode()
def infer_float32(model, working, geometry, original_size, device, *, forward_fn=None):
    """Single-image actual forward; no high-resolution probabilities are retained."""
    device=torch.device(device);started=perf_counter();seen=[]
    if device.type=='cuda':
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    model=model.to(device=device,dtype=torch.float32).eval()
    handle=model.stem.register_forward_pre_hook(lambda module,args:seen.append(list(args[0].shape)))
    try:
        inputs=TF.normalize(TF.to_tensor(working).float(),MEAN,STD).unsqueeze(0).to(device)
        expected=[1,3,working.height,working.width]
        if inputs.dtype!=torch.float32:raise ValueError('Expected float32 input')
        if device.type=='cuda':torch.cuda.synchronize()
        forward_started=perf_counter()
        with torch.autocast(device.type,enabled=False):
            logits=(forward_fn or forward_compact_head)(model,inputs)
        if device.type=='cuda':torch.cuda.synchronize()
        forward_seconds=perf_counter()-forward_started
        if not seen or seen[-1]!=expected:raise ValueError('Actual encoder input shape mismatch')
        if logits.dtype!=torch.float32 or logits.shape[1]!=4:raise ValueError('Expected float32 four-class logits')
        logits=restore_logits(logits,geometry,original_size)
        if not torch.isfinite(logits).all():raise RuntimeError('Non-finite logits')
        mask=logits.argmax(1)[0].to(dtype=torch.uint8,device='cpu').numpy().copy()
        if device.type=='cuda':torch.cuda.synchronize()
        stats=dict(device=str(device),precision='float32',amp=False,tf32=False,
            actual_input_shape=seen[-1],forward_seconds=forward_seconds,
            inference_seconds=perf_counter()-started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else None,
            time_scope='Tensor normalization/transfer, forward, logit restoration and CPU class mask; PIL preprocessing measured separately')
        return mask,stats
    finally:handle.remove()


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_fixed_model(root):
    path=root/CHECKPOINT_REL;digest=sha(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if tuple(checkpoint['class_names'])!=CLASSES or checkpoint.get('num_classes')!=4:
        raise ValueError('The fixed checkpoint is not the required four-class model')
    if checkpoint['config']['kernel_size']!=3:raise ValueError('Expected kernel 3')
    model=ResNetUNet(num_classes=4,pretrained=False,kernel_size=3)
    model.load_state_dict(checkpoint['model_state'],strict=True);model.float().eval()
    convolutions={name:list(layer.kernel_size) for name,layer in model.named_modules()
                  if isinstance(layer,torch.nn.Conv2d)}
    if convolutions['stem.0']!=[7,7] or convolutions['head']!=[1,1]:
        raise ValueError('Unexpected stem or head')
    if any(shape not in ([3,3],[1,1]) for name,shape in convolutions.items() if name!='stem.0'):
        raise ValueError('Unexpected convolution structure')
    metadata=dict(checkpoint=str(path),checkpoint_sha256=digest,epoch=checkpoint['epoch'],
        class_names=list(CLASSES),convolutions=convolutions,
        state_sha256=state_digest(model),kernel_size=3,training_performed=False)
    del checkpoint
    return model,metadata


def prepare_protocol(root,run,backend):
    path=root/REFERENCE_REL;manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest['version']!='benchmark_v2' or len(manifest['entries'])!=9:
        raise ValueError('Expected unchanged benchmark_v2 with nine references')
    entries=[]
    for entry in manifest['entries']:
        image=root/entry['image_path'];mask=path.parent/entry['mask']
        if sha(image)!=entry['image_sha256'] or sha(mask)!=entry['mask_sha256']:
            raise ValueError('Frozen source image or reference mask changed')
        if entry['image'] in EXPECTED_ROI_IMAGES and [entry['width'],entry['height']]!=EXPECTED_ROI_IMAGES[entry['image']]:
            raise ValueError('Fixed ROI coordinates require the original image dimensions')
        entries.append(dict(image=entry['image'],image_path=str(image),image_sha256=entry['image_sha256'],
            group=entry['group'],private=False,mask_path=str(mask),mask_sha256=entry['mask_sha256']))
    if sum(e['group']=='landscapes' for e in entries)!=6:raise ValueError('Expected six frozen landscape negatives')
    for name in NJU_NAMES:
        image=root/'data/nju_testset'/name
        entries.append(dict(image=name,image_path=str(image),image_sha256=sha(image),
                            group='nju',private=True,mask_path=None,mask_sha256=None))
    point_path=root/POINTS_REL;points=json.loads(point_path.read_text(encoding='utf-8'))
    if len(points['coordinates'])!=16 or points['hit_rule']!='exact pixel; no radius':
        raise ValueError('The existing 16-point protocol or exact-pixel rule changed')
    protocol=dict(protocol=PROTOCOL,checkpoint=str(root/CHECKPOINT_REL),checkpoint_sha256=sha(root/CHECKPOINT_REL),
        class_names=list(CLASSES),settings=list(LABELS),target_sizes=list(SCALES),
        benchmark_version='benchmark_v2',reference_manifest_sha256=sha(path),
        far_point_protocol_sha256=sha(point_path),far_point_protocol=points,
        rois=ROIS,roi_coordinate_system='original image; [x0,y0,x1,y1), fixed from RGB before experiment-14 predictions',
        normalization=dict(mean=list(MEAN),std=list(STD)),padding_color=list(PAD_COLOR),
        resize='PIL bilinear RGB; centered integer letterbox; continuous bilinear logits before argmax',
        precision='float32',amp=False,tf32=False,backend=backend,torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),pillow_version=PIL_VERSION,
        cudnn_benchmark=False,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        inference_graph='Existing compact head: same weights, final affine 1x1 head before interpolation at every scale',
        inference_code_sha256=hashlib.sha256((inspect.getsource(prepare_input)+inspect.getsource(restore_logits)+
            inspect.getsource(infer_float32)+inspect.getsource(forward_compact_head)).encode()).hexdigest(),
        entries=entries,legend=REFERENCE_LEGEND)
    request=run/'protocol.json'
    if request.exists() and json.loads(request.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Existing protocol differs: use a new run directory')
    save_json(request,protocol)
    save_json(run/'roi_protocol.json',dict(coordinates=ROIS,source='Original RGB viewed before predictions',
        image_sizes=EXPECTED_ROI_IMAGES))
    save_json(run/'far_point_protocol.json',points)
    save_json(run/'reference_manifest_snapshot.json',manifest)
    return protocol,manifest


def run_case(model,image,entry,size,protocol,run):
    label='Native' if size is None else str(size)
    folder=run/'cases'/('nju' if entry['private'] else 'references')/Path(entry['image']).stem/label
    folder.mkdir(parents=True,exist_ok=True)
    prep_started=perf_counter();working,geometry=prepare_input(image,size)
    preprocessing_seconds=perf_counter()-prep_started
    input_hash=hashlib.sha256(working.tobytes()).hexdigest()
    common=dict(protocol=PROTOCOL,checkpoint_sha256=protocol['checkpoint_sha256'],image_sha256=entry['image_sha256'],
        label=label,geometry=geometry,input_rgb_sha256=input_hash,precision='float32',amp=False,tf32=False,
        mean=list(MEAN),std=list(STD),padding_color=list(PAD_COLOR),
        inference_code_sha256=protocol['inference_code_sha256'],torch_version=protocol['torch_version'],
        cuda_version=protocol['cuda_version'],cudnn_version=protocol['cudnn_version'],backend=protocol['backend'],
        pillow_version=PIL_VERSION,cudnn_deterministic=protocol['cudnn_deterministic'],
        deterministic_algorithms=protocol['deterministic_algorithms'])
    cache_key=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest()
    preview=input_preview(working);preview.save(folder/'input.jpg',quality=92);preview.close()
    record_path=folder/'case.json';mask_path=folder/'mask.png'
    if record_path.exists():
        cached=json.loads(record_path.read_text(encoding='utf-8'))
        if cached.get('status')=='complete' and cached.get('cache_key')==cache_key and mask_path.exists() and sha(mask_path)==cached['mask_sha256']:
            cached['cache_reused']=True
            if working is not image:working.close()
            return cached
    result=dict(image=entry['image'],group=entry['group'],private=entry['private'],setting=label,
        target_size=size,original_size=list(image.size),attempted_input_shape=[1,3,working.height,working.width],
        folder=str(folder),input_preview=str(folder/'input.jpg'),cache_key=cache_key,cache_reused=False,
        preprocessing_seconds=preprocessing_seconds,attempts=[])
    try:
        attempt_started=perf_counter()
        try:
            prediction,stats=infer_float32(model,working,geometry,image.size,'cuda:0')
            result['attempts'].append(dict(status='complete',**stats))
        except torch.OutOfMemoryError as error:
            result['attempts'].append(dict(device='cuda:0',status='OOM',reason=str(error),
                attempted_input_shape=result['attempted_input_shape'],seconds=perf_counter()-attempt_started,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
            model.cpu();release()
            estimate=working.width*working.height*350+2*2**30
            if estimate>psutil.virtual_memory().available*.9:
                raise MemoryError('Insufficient CPU memory for the same input; configuration not resized')
            prediction,stats=infer_float32(model,working,geometry,image.size,'cpu')
            result['attempts'].append(dict(status='complete',**stats))
        if prediction.shape!=(image.height,image.width):raise ValueError('Original-coordinate mask mismatch')
        Image.fromarray(prediction).save(mask_path)
        result.update(status='complete',mask=str(mask_path),mask_sha256=sha(mask_path),**stats)
    except Exception as error:
        result.update(status='failed',failure_type=type(error).__name__,failure_reason=str(error))
    finally:
        if working is not image:working.close()
        release()
    save_json(record_path,result)
    return result


def measurements(matrix):
    row=photo_metrics(matrix);denominator=int(matrix[1].sum())
    row.update(reference_penguin_pixels=denominator)
    for c,name in ((0,'background'),(2,'other_bird'),(3,'seal')):
        row['penguin_to_'+name+'_pixels']=int(matrix[1,c])
        row['penguin_to_'+name+'_fraction']=float(matrix[1,c]/denominator) if denominator else None
    return row


def evaluate_case(case,entry,points):
    rows=[];roi_rows=[];point_rows=[];matrix=None
    if case['status']!='complete' or entry['private']:return rows,roi_rows,point_rows,matrix
    with Image.open(case['mask']) as raw:prediction=np.array(raw,dtype=np.uint8)
    with Image.open(entry['mask_path']) as raw:target=convert_labels(np.asarray(raw))
    matrix=confusion4(prediction,target)
    rows.append(dict(image=entry['image'],group=entry['group'],setting=case['setting'],
        scope='image',benchmark_version='benchmark_v2',**measurements(matrix)))
    for name,box in ROIS.get(entry['image'],{}).items():
        x0,y0,x1,y1=box
        roi_rows.append(dict(image=entry['image'],roi=name,setting=case['setting'],coordinates=box,
            **measurements(confusion4(prediction[y0:y1,x0:x1],target[y0:y1,x0:x1]))))
    if entry['image']=='penguins_far.jpg':
        for index,(x,y) in enumerate(points,1):
            value=int(prediction[y,x])
            point_rows.append(dict(setting=case['setting'],point=index,x=x,y=y,predicted_class=CLASSES[value],
                                   penguin_hit=value==1,any_animal_hit=value>0))
    return rows,roi_rows,point_rows,matrix


def font(size=18):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def overlay(image,prediction,box=None):
    rgb=image.crop(tuple(box)) if box else image.copy()
    values=prediction[box[1]:box[3],box[0]:box[2]] if box else prediction
    rgb.thumbnail((420,280))
    small=np.asarray(Image.fromarray(values).resize(rgb.size,Image.Resampling.NEAREST))
    array=np.asarray(rgb).copy();visible=small>0
    array[visible]=(.5*array[visible]+.5*COLORS[small[visible]]).astype(np.uint8)
    return Image.fromarray(array)


def paste_center(canvas,image,x,y,width=440,height=285):
    canvas.paste(image,(x+(width-image.width)//2,y+(height-image.height)//2))


def failure_text(draw,case,x,y):
    message='FAILED: '+case.get('failure_reason','missing configuration')
    draw.multiline_text((x+10,y+12),'\n'.join(textwrap.wrap(message,43)[:10]),font=font(14),fill='red')


def render_image(image,entry,cases,points,run):
    """Every figure uses the same five columns, including failed configurations."""
    folder=run/'visualizations'/('nju' if entry['private'] else 'references')/Path(entry['image']).stem
    folder.mkdir(parents=True,exist_ok=True)
    main=Image.new('RGB',(2200,740),'white');draw=ImageDraw.Draw(main)
    draw.text((12,8),entry['image']+' | Actual network input / original-coordinate prediction',font=font(21),fill='black')
    draw.text((12,704),REFERENCE_LEGEND,font=font(17),fill=(0,135,40))
    draw.text((12,722),'Background: original RGB unchanged. Penguin: red. Other bird: orange. Seal: cyan.',font=font(15),fill='black')
    error_canvases={}
    target=None
    if not entry['private']:
        with Image.open(entry['mask_path']) as raw:target=convert_labels(np.asarray(raw))
        for name in ('penguin_to_bird','background_to_bird'):
            error_canvases[name]=Image.new('RGB',(2200,365),'white')
            label='GT penguin -> predicted bird (magenta)' if name=='penguin_to_bird' else 'GT background -> predicted bird (yellow)'
            ImageDraw.Draw(error_canvases[name]).text((10,4),entry['image']+' | '+label,font=font(20),fill='black')
    roi_canvases={name:Image.new('RGB',(2200,375),'white') for name in ROIS.get(entry['image'],{})}
    for name,canvas in roi_canvases.items():
        ImageDraw.Draw(canvas).text((10,4),entry['image']+' | '+name+' | original coordinates '+str(ROIS[entry['image']][name]),font=font(18),fill='black')
    for col,case in enumerate(cases):
        x=440*col;shape=case.get('actual_input_shape',case['attempted_input_shape'])
        label=case['setting']+' | '+('input ' if case['status']=='complete' else 'attempted ')+str(shape)
        draw.text((x+8,40),label,font=font(15),fill='black')
        with Image.open(case['input_preview']) as raw:paste_center(main,raw.convert('RGB'),x,65)
        draw.text((x+8,365),case['setting']+' | '+case.get('device','failed'),font=font(17),fill='black')
        if case['status']!='complete':
            failure_text(draw,case,x,400)
            for canvas in [*error_canvases.values(),*roi_canvases.values()]:failure_text(ImageDraw.Draw(canvas),case,x,50)
            continue
        with Image.open(case['mask']) as raw:prediction=np.array(raw,dtype=np.uint8)
        small=overlay(image,prediction);paste_center(main,small,x,395)
        for name,canvas in error_canvases.items():
            errors=(target==(1 if name=='penguin_to_bird' else 0))&(prediction==2)
            small=image.copy();small.thumbnail((420,280))
            visible=np.asarray(Image.fromarray(errors.astype(np.uint8)).resize(small.size,Image.Resampling.NEAREST))>0
            array=(np.asarray(small)*.55).astype(np.uint8)
            array[visible]=(255,0,220) if name=='penguin_to_bird' else (255,230,0)
            d=ImageDraw.Draw(canvas);d.text((x+8,36),case['setting']+' | error pixels: '+str(int(errors.sum())),font=font(16),fill='black')
            paste_center(canvas,Image.fromarray(array),x,62)
        for name,canvas in roi_canvases.items():
            box=ROIS[entry['image']][name];small=overlay(image,prediction,box)
            d=ImageDraw.Draw(canvas);d.text((x+8,34),case['setting'],font=font(17),fill='black')
            if name=='distant_region':
                # Centers remain exact; dots are display annotations only.
                sd=ImageDraw.Draw(small);hit=0;animal=0
                for px,py in points:
                    value=int(prediction[py,px]);hit+=int(value==1);animal+=int(value>0)
                    sx=(px-box[0])*small.width/(box[2]-box[0]);sy=(py-box[1])*small.height/(box[3]-box[1])
                    color=(0,230,60) if value==1 else ((255,180,0) if value>0 else (240,20,20))
                    sd.ellipse((sx-3,sy-3,sx+3,sy+3),fill=color,outline='black')
                d.text((x+8,55),f'Penguin {hit}/16; any animal {animal}/16',font=font(15),fill='black')
                d.text((x+8,350),'Dots: green=penguin; amber=other animal; red=BG',font=font(12),fill='black')
            paste_center(canvas,small,x,78)
    main.save(folder/'scale_comparison.jpg',quality=90)
    assets=[('Input and prediction',folder/'scale_comparison.jpg')]
    for name,canvas in {**error_canvases,**roi_canvases}.items():
        path=folder/(name+'.jpg');canvas.save(path,quality=92);assets.append((name,path));canvas.close()
    main.close()
    return assets


def resource_row(case):
    return {k:case.get(k) for k in ('image','group','private','setting','status','failure_type','failure_reason',
        'device','precision','actual_input_shape','attempted_input_shape','forward_seconds','inference_seconds',
        'preprocessing_seconds','peak_allocated_GiB','peak_reserved_GiB','cache_reused')}


def plot_curves(rows,landscapes,points,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions=[0,1,2,3,4.4];fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    near={r['setting']:r for r in rows if r['image']=='penguins_near.jpg'}
    far={r['setting']:r for r in points};bg={r['setting']:r for r in landscapes}
    series=[([100*near[l]['penguin_to_other_bird_fraction'] if l in near else np.nan for l in LABELS],
             'Near: reference penguin -> bird','Reference penguin pixels (%)'),
            ([far[l]['penguin_hits'] if l in far else np.nan for l in LABELS],
             'Far: exact penguin center hits','Hits out of 16'),
            ([100*bg[l]['background_fp_fraction'] if l in bg else np.nan for l in LABELS],
             'Six landscapes: pooled false animals','Reference background pixels (%)')]
    for ax,(values,title,ylabel) in zip(axes,series):
        ax.plot(positions[:4],values[:4],'o-')
        ax.scatter([positions[4]],[values[4]],marker='s',color='darkorange',label='Native (separate)')
        ax.axvline(3.7,color='gray',linestyle='--');ax.set(xticks=positions,xticklabels=LABELS,title=title,ylabel=ylabel)
        ax.set_xlabel('Letterbox canvas size; Native is not a shared resolution')
    axes[1].set_ylim(-.5,16.5)
    fig.text(.5,.01,'Native is isolated: original dimensions differ by image. Metrics use original coordinates; ignore=255 excluded.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(path,dpi=150);plt.close(fig)


def diagnosis(rows,landscapes,points,cases):
    near={r['setting']:r for r in rows if r['image']=='penguins_near.jpg'}
    far={r['setting']:r for r in points};bg={r['setting']:r for r in landscapes}
    lines=['## Diagnostic answers','']
    if all(l in near for l in LABELS[:4]):
        values=[near[l]['penguin_to_other_bird_fraction'] for l in LABELS[:4]]
        trend='non-decreasing' if all(b>=a for a,b in zip(values,values[1:])) else ('non-increasing' if all(b<=a for a,b in zip(values,values[1:])) else 'non-monotonic')
        lines.append(f'1. Near penguin-to-bird confusion across 640/960/1280/1920 is {trend}: '+', '.join(f'{100*v:.4f}%' for v in values)+'. Native is reported separately'+(f" ({100*near['Native']['penguin_to_other_bird_fraction']:.4f}%)." if 'Native' in near else ' and failed.'))
    else:lines.append('1. Near confusion trend is incomplete because one or more resized configurations failed.')
    if '640' in far:
        baseline=far['640']['penguin_hits'];improved=[l for l in LABELS[1:4] if l in far and far[l]['penguin_hits']>baseline]
        lines.append(f'2. Far point baseline is {baseline}/16 at 640. '+('First tested resized setting with more penguin hits: '+improved[0]+'.' if improved else 'No larger letterbox setting recovers more selected penguin centers.'))
        for label in LABELS:
            if label in far and label in bg:
                far_photo=next((r for r in rows if r['image']=='penguins_far.jpg' and r['setting']==label),None)
                lines.append(f"   - {label}: penguin {far[label]['penguin_hits']}/16; any animal {far[label]['any_animal_hits']}/16; pooled landscape FP {100*bg[label]['background_fp_fraction']:.4f}%"+(f"; far-photo background FP {100*far_photo['background_fp_fraction']:.4f}%." if far_photo else '.'))
    else:lines.append('2. The 640 far configuration failed; no recovery claim relative to it is made.')
    complete=[l for l in LABELS[:4] if l in near and l in far and l in bg]
    vectors={l:(near[l]['penguin_to_other_bird_fraction'],-far[l]['penguin_hits'],bg[l]['background_fp_fraction']) for l in complete}
    pareto=[l for l in complete if not any(other!=l and all(a<=b for a,b in zip(vectors[other],vectors[l])) and any(a<b for a,b in zip(vectors[other],vectors[l])) for other in complete)]
    middle=[l for l in ('960','1280') if l in pareto]
    lines.append('3. Resized-setting Pareto candidates on the three stated diagnostic axes: '+(', '.join(pareto) or 'not assessable')+'. '+('Intermediate non-dominated candidates: '+', '.join(middle)+'.' if middle else 'No intermediate setting is non-dominated on these axes.')+' This is descriptive, with no new classification thresholds or tuning. It does not establish a universal optimum.')
    conflicts=[]
    for a,b in zip(LABELS[:3],LABELS[1:4]):
        for group,metric in [('penguin_photos','penguin_iou'),('landscapes','background_fp_fraction')]:
            first={r['image']:r for r in rows if r['setting']==a and r['group']==group}
            second={r['image']:r for r in rows if r['setting']==b and r['group']==group}
            up=[n for n in first.keys()&second.keys() if second[n][metric]>first[n][metric]]
            down=[n for n in first.keys()&second.keys() if second[n][metric]<first[n][metric]]
            if up and down:conflicts.append(f'{a}->{b}, {metric}: increases in '+', '.join(sorted(up))+'; decreases in '+', '.join(sorted(down))+'.')
    lines.append('4. '+('Opposite per-image trends are present:' if conflicts else 'No opposite adjacent resized-setting changes were detected in the completed per-image IoU/landscape FP rows; this limited check does not prove uniform behavior.'))
    lines.extend('   - '+text for text in conflicts)
    failures=[r for r in cases if r['status']!='complete']
    lines+=['',f'Completed {len(cases)-len(failures)}/{len(cases)} configurations. Failed settings retain their names and reasons; no smaller substitute was used.',
        'Native uses different original sizes, so it is not included in the resized-resolution Pareto ordering. Device/precision records must be considered when reading timings.',
        'These model-assisted development references are not a blind independent test. No inference threshold, fusion, model structure or weights were tuned. Results cannot be attributed directly to a kernel size or crop boundary.','']
    return '\n'.join(lines)


def primary_table(rows,landscapes,points):
    near={r['setting']:r for r in rows if r['image']=='penguins_near.jpg'}
    bg={r['setting']:r for r in landscapes};far={r['setting']:r for r in points}
    text='| Setting | Near penguin -> bird | Far penguin point hits | Pooled landscape FP |\n|---|---|---|---|\n'
    for label in LABELS:
        if label in near and label in bg and label in far:
            text+=f"| {label} | {100*near[label]['penguin_to_other_bird_fraction']:.2f}% | {far[label]['penguin_hits']}/16 | {100*bg[label]['background_fp_fraction']:.2f}% |\n"
    return text+'\nNative uses a different original resolution for each photograph.\n\n'


def resource_note(cases):
    complete=[c for c in cases if c['status']=='complete']
    gpu=sum(c['device'].startswith('cuda') for c in complete)
    peaks=[c['peak_allocated_GiB'] for c in complete if c.get('peak_allocated_GiB') is not None]
    native=[c['inference_seconds'] for c in complete if c['setting']=='Native']
    text=f'Observed successful cases: GPU {gpu}, CPU {len(complete)-gpu}. '
    if peaks:text+=f'Maximum CUDA allocated peak: {max(peaks):.2f} GiB. '
    if native:text+=f'Native inference times: {min(native):.2f}-{max(native):.2f} seconds per photograph. '
    return text+'Times include device placement and first-use costs; they are diagnostic measurements, not steady-state latency benchmarks.\n\n'


def build_outputs(root,run,protocol,manifest,cases):
    rows=[];roi_rows=[];point_rows=[];matrices={};attempts=[];gallery=[];public={}
    points=protocol['far_point_protocol']['coordinates']
    for entry in protocol['entries']:
        group_cases=[next(r for r in cases if r['image']==entry['image'] and r['setting']==l) for l in LABELS]
        for case in group_cases:
            image_rows,roi,pt,matrix=evaluate_case(case,entry,points)
            rows.extend(image_rows);roi_rows.extend(roi);point_rows.extend(pt)
            for attempt in case['attempts']:attempts.append(dict(image=case['image'],setting=case['setting'],**attempt))
            if matrix is not None and entry['group']=='landscapes':
                matrices.setdefault(case['setting'],[]).append(matrix)
        with Image.open(entry['image_path']) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
        assets=render_image(image,entry,group_cases,points,run);image.close()
        for label,path in assets:gallery.append((entry['image']+' | '+label,path))
        if not entry['private']:public[entry['image']]=assets
    landscapes=[]
    for label,values in matrices.items():
        # A pooled six-photo result is only valid when all six configurations succeeded.
        if len(values)==6:landscapes.append(dict(setting=label,images=6,scope='pooled',**measurements(sum(values,np.zeros((4,4),np.int64)))))
    point_summary=[]
    for label in LABELS:
        values=[r for r in point_rows if r['setting']==label]
        if values:
            assert len(values)==16
            point_summary.append(dict(setting=label,points=16,penguin_hits=sum(r['penguin_hit'] for r in values),any_animal_hits=sum(r['any_animal_hit'] for r in values)))
    save_csv(run/'image_metrics.csv',rows);save_csv(run/'roi_metrics.csv',roi_rows)
    save_csv(run/'landscape_metrics.csv',[r for r in rows if r['group']=='landscapes'])
    save_csv(run/'landscape_summary.csv',landscapes)
    save_csv(run/'point_results.csv',point_rows);save_csv(run/'point_summary.csv',point_summary)
    save_csv(run/'resources.csv',[resource_row(c) for c in cases]);save_csv(run/'device_log.csv',attempts)
    save_json(run/'failures.json',[r for r in cases if r['status']!='complete'])
    html=write_gallery_html(run/'gallery.html','Experiment 14: fixed-weight inference-scale sensitivity',gallery,
        REFERENCE_LEGEND+'. Background is unchanged RGB; no NJU metrics. All columns preserve the requested setting.')
    plot_curves(rows,landscapes,point_summary,run/'error_curves.png')
    report=root/'reports/exp14';report.mkdir(parents=True,exist_ok=True)
    for name in ('image_metrics.csv','roi_metrics.csv','landscape_metrics.csv','landscape_summary.csv','point_results.csv','point_summary.csv','error_curves.png'):
        shutil.copy2(run/name,report/name)
    save_csv(report/'public_resources.csv',[resource_row(c) for c in cases if not c['private']])
    save_json(report/'protocol.json',{k:v for k,v in protocol.items() if k!='entries'})
    # Public overview contains only reference photographs, never NJU pixels.
    selected=['penguins_near.jpg','penguins_far.jpg','antarctic_landscape_untitled_2.jpg']
    canvases=[]
    for name in selected:
        with Image.open(public[name][0][1]) as image:canvases.append(image.convert('RGB'))
    overview=Image.new('RGB',(2200,sum(i.height for i in canvases)),'white');y=0
    for image in canvases:overview.paste(image,(0,y));y+=image.height
    overview.thumbnail((1600,1800));overview.save(report/'public_overview.jpg',quality=90)
    for name in ('penguins_near.jpg','penguins_far.jpg'):
        for label,path in public[name][1:]:shutil.copy2(path,report/(Path(name).stem+'_'+label+'.jpg'))
    answers=diagnosis(rows,landscapes,point_summary,cases)
    text='# Experiment 14: Inference Scale Sensitivity\n\n'
    text+=f"Fixed checkpoint SHA256: `{protocol['checkpoint_sha256']}`. No training.\n\n"
    text+='Actual single-image inputs: 640/960/1280/1920 centered letterbox canvases and original-pixel Native. Padding RGB=(124,116,104), ImageNet normalization, bilinear RGB resize. Remove padding from continuous logits, bilinearly restore original coordinates, then argmax. Same compact affine-head inference graph at every setting; no fusion, sliding windows or threshold changes.\n\n'
    text+='All model inputs, weights and logits are float32; autocast and both matmul/convolution TF32 are disabled. GPU is attempted first; only CUDA OOM permits CPU float32 at the unchanged requested size. CPU and GPU timings are not interchangeable. Forward and total inference times, preprocessing time, successful-device peaks and failed GPU attempts are separate in the resource logs.\n\n'
    text+='CUDA peaks are PyTorch allocator accounting, not a measurement of physically resident dedicated VRAM. On Windows/WDDM they can exceed card VRAM while execution becomes much slower; consult runtime_environment.json and per-case timings. No per-process GPU memory cap or input reduction is applied.\n\n'
    text+=resource_note(cases)
    text+='The green square is a 640x640 input-coordinate reference drawn only on preview copies, including padding at 640. It is not the receptive field and does not certify a training-distribution region. Background remains original RGB in overlays; penguin red, other bird orange, seal cyan.\n\n'
    text+='Head/chest and distant ROIs were fixed from original RGB before experiment-14 predictions; coordinates are in protocol.json. Error maps separately show GT-penguin->bird (magenta) and GT-background->bird (yellow). Metrics ignore label 255 and use original image coordinates. Exact existing 16-point hits are not instance recall. Reference bird/seal recall is undefined without positive labels; use animal validation for those classes.\n\n'
    text+=primary_table(rows,landscapes,point_summary)
    text+='![Public overview](public_overview.jpg)\n\n![Diagnostic curves](error_curves.png)\n\n'+answers
    text+=f'\nFull 15-image HTML gallery and class masks: `{html}`. NJU is visualization only, and all NJU pixels remain under ignored artifacts storage.\n'
    text+='\nReference protocol: unchanged benchmark_v2; near and far include model-assisted annotations. This is a fixed-weight scale diagnosis, not independent blind-test accuracy or a kernel/boundary causal ablation.\n'
    (report/'README.md').write_text(text,encoding='utf-8');(run/'diagnosis.md').write_text(answers,encoding='utf-8')
    return dict(gallery=str(html),report=str(report/'README.md'),answers=answers)


def run_experiment14(*,root=None,run_dir=None,execute=False):
    root=Path(root or PROJECT_ROOT).resolve()
    if not execute:
        print('PLAN ONLY: 15 images x 5 settings, fixed experiment-13 best, float32, no training.')
        return dict(status='prepared',settings=list(LABELS),configurations=75)
    if run_dir is None:raise ValueError('Pass a fixed run_<timestamp> directory to enable exact-cache resume')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the GPU-first experiment; no silent protocol change')
    run=Path(run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    device_info=torch.cuda.get_device_properties(0)
    save_json(run/'runtime_environment.json',dict(platform=platform.platform(),
        gpu=device_info.name,dedicated_vram_bytes=device_info.total_memory,
        system_ram_bytes=psutil.virtual_memory().total,cpu_threads=4,
        gpu_memory_cap=None,peak_memory_definition='PyTorch CUDA allocator accounting; not physical dedicated-VRAM residency'))
    with strict_float32() as backend:
        protocol,manifest=prepare_protocol(root,run,backend)
        model,metadata=load_fixed_model(root);save_json(run/'model.json',metadata)
        cases=[]
        for entry in protocol['entries']:
            with Image.open(entry['image_path']) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
            for size in SCALES:
                label='Native' if size is None else str(size)
                print(f"{len(cases)+1}/75 | {entry['image']} | {label}",flush=True)
                case=run_case(model,image,entry,size,protocol,run);cases.append(case)
                save_json(run/'progress.json',cases)
                print('  ',case['status'],case.get('device',''),case.get('actual_input_shape',case['attempted_input_shape']),flush=True)
            image.close()
        if sha(root/CHECKPOINT_REL)!=metadata['checkpoint_sha256'] or state_digest(model)!=metadata['state_sha256']:
            raise ValueError('Fixed checkpoint or in-memory model state changed')
        if sha(root/REFERENCE_REL)!=protocol['reference_manifest_sha256'] or sha(root/POINTS_REL)!=protocol['far_point_protocol_sha256']:
            raise ValueError('Frozen reference/point protocol changed')
        outputs=build_outputs(root,run,protocol,manifest,cases)
        result=dict(protocol=PROTOCOL,status='complete' if all(c['status']=='complete' for c in cases) else 'partial',
            folder=str(run),completed=sum(c['status']=='complete' for c in cases),total=75,
            checkpoint_sha256=metadata['checkpoint_sha256'],weights_unchanged=True,training_performed=False,**outputs)
        save_json(run/'result.json',result)
        save_json(root/'artifacts/experiments/exp14_results/latest.json',result)
        del model;release()
        print(outputs['answers'],flush=True)
        return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    run_experiment14(run_dir=args.run_dir,execute=args.run)
