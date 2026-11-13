"""Native-resolution float32 visualization of the completed experiment-16 model.

No training, resizing, tiles or fusion. Existing training code and 960 predictions
remain unchanged. Release consumed skip tensors early without changing operations.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import gc
import html
import inspect
import json
import os
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import psutil
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from project_paths import PROJECT_ROOT
from experiment13 import sha, save_json, save_csv
from experiment14 import strict_float32
from experiment15 import load_checkpoint, confusion5, metrics5
from experiment16 import metrics_row, PROTOCOL
from experiment9 import forward_compact_head, MEAN, STD
from gallery import write_gallery_html,save_visualization


def forward_releasing_skips(model, inputs):
    """Same full-image operations; do not retain consumed decoder temporaries."""
    size=inputs.shape[-2:]
    skips=[model.stem(inputs)]
    skips.append(model.enc1(model.pool(skips[-1])))
    skips.append(model.enc2(skips[-1]))
    skips.append(model.enc3(skips[-1]))
    current=model.enc4(skips[-1])
    for block in (model.dec4,model.dec3,model.dec2,model.dec1):
        skip=skips.pop()
        up=F.interpolate(current,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        del current
        current=torch.cat((up,skip),dim=1)
        del up,skip
        for layer in block.fuse.layers:
            current=layer(current)
    logits=model.head(current)
    del current
    return F.interpolate(logits,size=size,mode='bilinear',align_corners=False)


@torch.inference_mode()
def infer_native(model,image,device):
    device=torch.device(device);seen=[];started=perf_counter()
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    model.to(device=device,dtype=torch.float32).eval()
    handle=model.stem.register_forward_pre_hook(lambda module,args:seen.append(list(args[0].shape)))
    try:
        inputs=TF.normalize(TF.to_tensor(image).float(),MEAN,STD).unsqueeze(0).to(device)
        with torch.autocast(device.type,enabled=False):logits=forward_releasing_skips(model,inputs)
        if seen!=[[1,3,image.height,image.width]] or logits.shape!=(1,5,image.height,image.width):
            raise ValueError('Native input/output coordinates changed')
        if logits.dtype!=torch.float32 or not torch.isfinite(logits).all():raise ValueError('Invalid float32 logits')
        prediction=logits.argmax(1)[0].to(dtype=torch.uint8,device='cpu').numpy().copy()
        if device.type=='cuda':torch.cuda.synchronize()
        return prediction,dict(actual_input_shape=seen[0],device=str(device),precision='float32',amp=False,tf32=False,
            inference_seconds=perf_counter()-started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else None)
    finally:handle.remove()


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def comparison_gallery(path,records,old_root,out):
    parts=['<!doctype html><meta charset="utf-8"><title>Experiment 16: 960 vs Native</title>',
        '<style>body{font:16px system-ui;margin:24px;background:#eee}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}section{background:white;padding:16px;margin:20px 0}h2{font-size:18px}</style>',
        '<h1>Experiment 16: 960 vs Native</h1><p>Same best checkpoint; float32, no fusion. Background purple; penguin red; bird orange; seal cyan; whale green. NJU is qualitative only. Native preserves original pixels; dimensions differ across photographs.</p>']
    for record in records:
        parts.append('<section><h2>'+html.escape(record['group']+' / '+record['name'])+'</h2>')
        if record['status']!='complete':
            parts.append('<p>'+html.escape(record['failure_reason'])+'</p></section>');continue
        native=Path(record['folder'])/record['overlay']
        old=old_root/record['group']/Path(record['name']).stem/record['old_overlay']
        parts.append('<div class="row">')
        for label,p in [('960 letterbox',old),('Native '+str(record['actual_input_shape'][-2:])+' / '+record['device'],native)]:
            url=quote(os.path.relpath(p,path.parent).replace('\\','/'),safe='/')
            parts.append('<div><p>'+html.escape(label)+'</p><img loading="lazy" src="'+url+'"></div>')
        parts.append('</div></section>')
    path.write_text('\n'.join(parts),encoding='utf-8')
    examples=[r for r in records if r['group']=='examples' and r['status']=='complete']
    if examples:
        sheet=Image.new('RGB',(1500,560*len(examples)),(245,246,250));draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
        for index,r in enumerate(examples):
            for col,(label,p) in enumerate([('960 letterbox',old_root/r['group']/Path(r['name']).stem/r['old_overlay']),
                    ('Native',Path(r['folder'])/r['overlay'])]):
                with Image.open(p) as raw:
                    im=raw.convert('RGB');im.thumbnail((730,500))
                x=col*750+(750-im.width)//2;y=index*560+55
                sheet.paste(im,(x,y));draw.text((col*750+12,index*560+12),r['name']+' | '+label,font=font,fill=(20,30,50));im.close()
        sheet.save(out/'penguins_960_vs_native.jpg',quality=93);sheet.close()


def run_native(root=PROJECT_ROOT,run_dir=None):
    root=Path(root).resolve()
    run=Path(run_dir) if run_dir else Path(json.loads((root/'artifacts/experiments/exp16_results/latest.json').read_text())['run_dir'])
    digest=sha(run/'best.pt');old_root=run/'predictions'/('best_'+digest[:12])
    old=json.loads((old_root/'result.json').read_text());cases=old['records']
    if len(cases)!=40:raise ValueError('Expected the same 40 completed 960 cases')
    out=run/'predictions'/('native_best_'+digest[:12]);out.mkdir(parents=True,exist_ok=True)
    model,saved=load_checkpoint(run/'best.pt')
    if saved['config'].get('protocol')!=PROTOCOL:raise ValueError('Unexpected checkpoint')
    del saved
    request=dict(protocol='exp16_native_float32_v1',checkpoint_sha256=digest,classes=5,
        implementation_sha256=sha(Path(__file__)),torch_version=str(torch.__version__),
        precision='float32',amp=False,tf32=False,normalization=dict(mean=list(MEAN),std=list(STD)),
        input='original full image; no resize, padding, tiles or fusion',images=40,
        early_release='Same compact-head operations; consumed skip and interpolation tensors freed early')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=request:
        # A source refactor must not overwrite the already recorded inference.
        out=run/'predictions'/('native_best_'+digest[:12]+'_'+request['implementation_sha256'][:8])
        out.mkdir(parents=True,exist_ok=True)
        if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=request:
            raise ValueError('Native inference protocol changed; choose a new run')
    save_json(out/'protocol.json',request)
    records=[]
    with strict_float32(),torch.inference_mode():
        probe=torch.randn((1,3,127,159),generator=torch.Generator().manual_seed(42))
        expected=forward_compact_head(model,probe);actual=forward_releasing_skips(model,probe)
        difference=float((expected-actual).abs().max())
        if not torch.equal(expected,actual):raise ValueError('Early-release graph changed float32 outputs')
        save_json(out/'graph_parity.json',dict(shape=list(probe.shape),maximum_absolute_difference=difference,bitwise_equal=True))
        del probe,expected,actual
        for index,old_case in enumerate(cases,1):
            case={key:old_case[key] for key in ('name','source','group','image_sha256')}
            folder=out/case['group']/Path(case['name']).stem;folder.mkdir(parents=True,exist_ok=True)
            path=folder/'record.json';mask_path=folder/'class_ids.png'
            if sha(root/case['source'])!=case['image_sha256'] or old_case['checkpoint_sha256']!=digest:
                raise ValueError('960 and Native source/checkpoint identities do not match')
            if path.exists():
                cached=json.loads(path.read_text())
                if cached.get('status')=='complete' and cached['image_sha256']==case['image_sha256'] and cached['checkpoint_sha256']==digest and sha(mask_path)==cached['mask_sha256']:
                    records.append(cached);print(f'[{index:02d}/40] CACHED {case["name"]}',flush=True);continue
            with Image.open(root/case['source']) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
            record=dict(case,folder=str(folder),checkpoint_sha256=digest,old_overlay=old_case['overlay'],attempts=[])
            print(f'[{index:02d}/40] START {case["name"]} | Native {image.width}x{image.height}',flush=True)
            try:
                prediction=stats=None
                if torch.cuda.is_available():
                    started=perf_counter()
                    try:prediction,stats=infer_native(model,image,'cuda:0')
                    except torch.OutOfMemoryError as error:
                        record['attempts'].append(dict(device='cuda:0',status='OOM',reason=str(error),seconds=perf_counter()-started))
                    # Leave the exception scope before releasing CUDA tensors.
                if prediction is None:
                    model.cpu();release()
                    estimate=image.width*image.height*300+int(1.5*2**30)
                    if estimate>psutil.virtual_memory().available*.95:raise MemoryError('Insufficient available RAM for unchanged native float32 input')
                    print(f'[{index:02d}/40] CPU FALLBACK | preserving native dimensions',flush=True)
                    prediction,stats=infer_native(model,image,'cpu')
                record['attempts'].append(dict(status='complete',**stats))
                Image.fromarray(prediction).save(mask_path)
                record.update(status='complete',mask=str(mask_path),mask_sha256=sha(mask_path),**stats,
                    **save_visualization(root,folder,case,prediction,mask_path))
                if old_case.get('reference_mask'):
                    reference=root/old_case['reference_mask']
                    if sha(reference)!=old_case['reference_mask_sha256']:raise ValueError('SAM3 reference changed')
                    with Image.open(reference) as gt:record['metrics']=metrics5(confusion5(prediction,np.asarray(gt)))
                    record['reference_origin']='User-confirmed SAM3 validation reference; not independent blind test'
                print(f'[{index:02d}/40] DONE {case["name"]} | {stats["device"]} | {stats["inference_seconds"]:.1f}s',flush=True)
            except (MemoryError,RuntimeError) as error:
                record.update(status='failed',failure_reason=str(error),failure_type=type(error).__name__)
                print(f'[{index:02d}/40] FAILED {case["name"]} | {error}',flush=True)
            finally:
                prediction=None;image.close();release()
            save_json(path,record);records.append(record)
            save_json(out/'progress.json',dict(completed=len(records),total=40,successful=sum(r['status']=='complete' for r in records)))
        model.cpu();del model;release()
    gallery=write_gallery_html(out/'gallery.html','Experiment 16: Native float32',
        [(r['group']+' / '+r['name'],Path(r['folder'])/r['overlay']) for r in records if r['status']=='complete'],
        'Original full-image inference; no resize, tiles or fusion. Background purple; penguin red; bird orange; seal cyan; whale green. NJU qualitative only.')
    comparison_gallery(out/'comparison.html',records,old_root,out)
    save_csv(out/'resources.csv',[dict(image=r['name'],group=r['group'],status=r['status'],
        **{key:r.get(key) for key in ('actual_input_shape','device','precision','inference_seconds','peak_allocated_GiB','failure_reason')}) for r in records])
    save_csv(out/'image_metrics.csv',[dict(image=r['name'],group=r['group'],**metrics_row(r['metrics'])) for r in records if 'metrics' in r])
    result=dict(gallery=str(gallery),comparison=str(out/'comparison.html'),successful=sum(r['status']=='complete' for r in records),
        failed=sum(r['status']!='complete' for r in records),records=records,training_performed=False)
    save_json(out/'result.json',result);return {key:value for key,value in result.items() if key!='records'}


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args();print(json.dumps(run_native(run_dir=args.run_dir),indent=2))
