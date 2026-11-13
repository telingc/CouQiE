"""Experiment-15 source audit and RGB-only target inventory; no model inference.

Historical source-snapshot pinning is LLM-assisted: Codex, 2026-11-12.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from scipy.fft import dctn
import torch
from torch.utils.data import Dataset,Subset

from animal_data import build_datasets,fit_image_and_mask
from experiment13 import sha,save_json,save_csv
from gallery import write_gallery_html

CLASSES=('background','penguin','other_bird','seal','whale')
ALLOWED=(0,1,2,3,4,255)


class SourceView(Dataset):
    """Unchanged five-class labels; separate audited image-sampling groups."""
    def __init__(self,base,records):
        self.base,self.records=base,records
        self.groups=np.asarray([r['group'] for r in records],dtype=np.int64)
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return self.base[index]


def source_data(root,run):
    folder=Path(run)/'source_audit';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'audit.json';records_path=folder/'records.json'
    from dataset_paths import historical_source_root
    source_root=historical_source_root(root,json.loads(path.read_text())['source_metadata']) if path.exists() else Path(root)
    datasets,metadata=build_datasets(source_root,640,include_background=True)
    if path.exists():
        audit=json.loads(path.read_text());records=json.loads(records_path.read_text())
        if audit['source_metadata']!=metadata or sha(records_path)!=audit['records_sha256']:
            raise ValueError('Source dataset identity changed; choose a new run')
        for split,base in datasets.items():
            names=[str(child.image_path(i).relative_to(source_root).as_posix()) for child in base.datasets for i in range(len(child))]
            if names!=[r['image'] for r in records[split]]:raise ValueError('Source record order changed')
            for row in records[split]:
                if sha(source_root/row['image'])!=row['image_sha256']:raise ValueError('Audited source image changed')
                if row.get('mask') and sha(source_root/row['mask'])!=row['mask_sha256']:raise ValueError('Audited source mask changed')
    else:
        records={};statistics=[]
        for split,base in datasets.items():
            records[split]=[]
            for child in base.datasets:
                def inspect_one(index):
                    image,mask=child.get_raw(index);canvas,target=fit_image_and_mask(image,mask,640)
                    labels=np.asarray(target)
                    if not np.isin(labels,ALLOWED).all():raise ValueError('Invalid five-class mask')
                    counts=np.bincount(labels.ravel(),minlength=256)
                    if not counts[:5].sum():raise ValueError('Fully ignored source image')
                    group=int(np.argmax(counts[1:5])+1) if counts[1:5].sum() else 0
                    source=child.image_path(index);record=child.records[index]
                    row=dict(image=source.relative_to(source_root).as_posix(),source=child.root.name,
                        image_sha256=sha(source),loader_group=int(record['sample_class']),group=group,
                        pixel_counts={str(c):int(counts[c]) for c in ALLOWED})
                    if child.prepared:
                        p=child.split_dir/record['mask'];row.update(mask=p.relative_to(source_root).as_posix(),mask_sha256=sha(p))
                    for item in (image,mask,canvas,target):item.close()
                    return row
                with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect_one,range(len(child))))
                records[split].extend(rows);groups=Counter(r['group'] for r in rows)
                loader_groups=Counter(r['loader_group'] for r in rows)
                statistics.append(dict(split=split,source=child.root.name,images=len(rows),
                    **{name+'_images':groups[c] for c,name in enumerate(CLASSES)},
                    **{name+'_pixels':sum(r['pixel_counts'][str(c)] for r in rows) for c,name in enumerate(CLASSES)},
                    loader_whale_images=loader_groups[4],ignored_pixels=sum(r['pixel_counts']['255'] for r in rows)))
                print('SOURCE AUDIT',split,child.root.name,len(rows),'whale loader/mask:',loader_groups[4],groups[4],flush=True)
            if len(records[split])!=len(base):raise ValueError('Source audit length mismatch')
        loader_whales={s:sum(r['loader_group']==4 for r in rows) for s,rows in records.items()}
        layout_path=source_root/'data/dataset_layout.json'
        expected_whales=({s:json.loads(layout_path.read_text())['expected_class_samples'][s].get('whale',0) for s in records}
                         if layout_path.exists() else {'train':41,'valid':12,'test':6})
        if loader_whales!=expected_whales:raise ValueError('Whale primary counts differ from the selected dataset version: '+str(loader_whales))
        counts=Counter(r['group'] for r in records['train'])
        if set(counts)!=set(range(5)):raise ValueError('All five sampling groups must be present')
        save_json(records_path,records)
        audit=dict(class_names=list(CLASSES),allowed_labels=list(ALLOWED),whale_mapping='identity: 4 remains whale',
            group_rule='largest foreground pixel area on 640 letterbox mask; background if no foreground',
            loader_whale_primary_counts=loader_whales,
            mask_whale_primary_counts={s:sum(r['group']==4 for r in rows) for s,rows in records.items()},
            source_metadata=metadata,records_sha256=sha(records_path),statistics=statistics,
            sampling=[dict(group=CLASSES[c],images=counts[c],image_weight=1/math.sqrt(counts[c])) for c in range(5)])
        save_json(path,audit);save_csv(folder/'statistics.csv',statistics)
    views={s:SourceView(datasets[s],records[s]) for s in datasets}
    animals=[i for i,r in enumerate(records['valid']) if r['source']!='dataset_nature']
    nature=[i for i,r in enumerate(records['valid']) if r['source']=='dataset_nature']
    return views['train'],Subset(views['valid'],animals),Subset(views['valid'],nature),audit


def _target_record(path,root,thumbs):
    with Image.open(path) as source:
        source.load();exif=source.getexif();timestamp=str(exif.get(306,''));camera=str(exif.get(272,''))
        nested=exif.get_ifd(34665) if 34665 in exif else {}
        timestamp=str(nested.get(36867,timestamp))
        image=ImageOps.exif_transpose(source).convert('RGB')
    width,height=image.size
    gray=np.asarray(image.resize((32,32),Image.Resampling.LANCZOS).convert('L'),dtype=np.float32)
    spectrum=dctn(gray,norm='ortho')[:8,:8];bits=(spectrum>np.median(spectrum.ravel()[1:])).ravel()
    phash=int.from_bytes(np.packbits(bits).tobytes(),'big')
    rgb=np.asarray(image.resize((16,16),Image.Resampling.BILINEAR),dtype=np.float32)
    hist=np.concatenate([np.histogram(rgb[:,:,c],bins=8,range=(0,256))[0] for c in range(3)]).astype(float)
    hist/=np.linalg.norm(hist)
    thumb=image.copy();thumb.thumbnail((320,220));thumb.save(thumbs/(path.stem+'.jpg'),quality=88)
    # Hash decoded RGB as well as JPEG bytes, so metadata-only variants are found.
    import hashlib
    pixel_sha=hashlib.sha256(image.tobytes()).hexdigest();image.close();thumb.close()
    return dict(id=path.stem,image=path.name,source=path.relative_to(root).as_posix(),image_sha256=sha(path),
        rgb_sha256=pixel_sha,width=width,height=height,capture_time=timestamp,camera=camera,
        phash=f'{phash:016x}',rgb_hist=hist.tolist())


def target_inventory(root,run):
    folder=Path(run)/'target_audit';folder.mkdir(parents=True,exist_ok=True);thumbs=folder/'thumbnails';thumbs.mkdir(exist_ok=True)
    path=folder/'inventory.json'
    if path.exists():
        result=json.loads(path.read_text())
        for row in result['entries']:
            if sha(Path(root)/row['source'])!=row['image_sha256']:raise ValueError('Target photo identity changed')
        return result
    paths=sorted(p for p in (Path(root)/'data/nju_weaklabel').rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff','.webp'))
    if len(paths)!=167:raise ValueError('Expected 167 target photos; found '+str(len(paths)))
    with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(lambda p:_target_record(p,Path(root),thumbs),paths))
    parent=list(range(len(entries)));edges=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j,reason,distance):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
        edges.append(dict(first=entries[i]['image'],second=entries[j]['image'],reason=reason,phash_distance=distance))
    times=[]
    for row in entries:
        try:times.append(datetime.strptime(row['capture_time'],'%Y:%m:%d %H:%M:%S'))
        except ValueError:times.append(None)
    for i,a in enumerate(entries):
        for j in range(i):
            b=entries[j];distance=(int(a['phash'],16)^int(b['phash'],16)).bit_count()
            similarity=float(np.dot(a['rgb_hist'],b['rgb_hist']))
            if a['image_sha256']==b['image_sha256'] or a['rgb_sha256']==b['rgb_sha256']:join(i,j,'exact duplicate',distance)
            elif distance<=6 and similarity>.92:join(i,j,'high perceptual similarity',distance)
            elif times[i] and times[j] and a['camera']==b['camera'] and abs((times[i]-times[j]).total_seconds())<=20 and similarity>.90:
                join(i,j,'nearby capture and similar RGB distribution',distance)
    groups={}
    for i,row in enumerate(entries):groups.setdefault(find(i),[]).append(row)
    group_rows=[]
    for number,rows in enumerate(sorted(groups.values(),key=lambda g:g[0]['image']),1):
        group=f'group_{number:03d}'
        for row in rows:row['group']=group
        group_rows.append(dict(group=group,images=[r['image'] for r in rows],count=len(rows)))
    test_names=('5N0A5174_1.jpg','5N0A5204_1.jpg','5N0A5965_1.jpg','5N0A7706_1.jpg','5N0A7727_1.jpg','5N0A9908_1.jpg')
    test_hashes={sha(Path(root)/'data/nju_testset'/name):name for name in test_names}
    overlaps=[dict(image=r['image'],test=test_hashes[r['image_sha256']]) for r in entries if r['image_sha256'] in test_hashes]
    if overlaps:raise ValueError('Target/test exact-photo overlap requires exclusion: '+str(overlaps))
    result=dict(expected_images=167,decoded_images=len(entries),integrity='passed',entries=entries,groups=group_rows,
        similarity_edges=edges,test_exact_overlaps=overlaps,
        notes='RGB/EXIF-only grouping proposal; inspect contact sheets before freezing. No predictions or class labels used.')
    save_json(path,result);save_csv(folder/'inventory.csv',[{k:v for k,v in r.items() if k!='rgb_hist'} for r in entries])
    contact_sheets(run,entries,'inventory')
    return result


def contact_sheets(run,entries,name):
    folder=Path(run)/'target_audit';assets=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for start in range(0,len(entries),30):
        canvas=Image.new('RGB',(1500,1580),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,7),f'Experiment 15 | RGB-only {name} | page {start//30+1} | no model predictions',font=font,fill='black')
        for k,row in enumerate(entries[start:start+30]):
            x=(k%5)*300;y=(k//5)*255+38
            with Image.open(folder/'thumbnails'/(row['id']+'.jpg')) as im:
                im=im.copy();im.thumbnail((290,205));canvas.paste(im,(x+(300-im.width)//2,y))
            draw.text((x+5,y+207),f"{start+k+1:03d} {row['image']}",font=small,fill='black')
            draw.text((x+5,y+226),row['group']+' | '+row.get('split','unassigned'),font=small,fill='navy')
        path=folder/f'{name}_{start//30+1:02d}.jpg';canvas.save(path,quality=92);assets.append((f'Page {start//30+1}',path));canvas.close()
    html=write_gallery_html(folder/(name+'.html'),'Experiment 15: RGB-only split review',assets,
        'Inspect scene/burst groups and class coverage. These are original RGB photos only, not model masks.')
    return html


def propose_split(root,run):
    """Add conservative scene merges selected from the RGB contact sheets only."""
    if (Path(run)/'split_manifest.json').exists():return json.loads((Path(run)/'split_manifest.json').read_text())
    inventory=target_inventory(root,run);entries=[dict(r) for r in inventory['entries']]
    parents=list(range(len(entries)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def merge(indices):
        indices=[i-1 for i in indices]
        for i in indices[1:]:parents[find(i)]=find(indices[0])
    by_group={}
    for i,row in enumerate(entries):by_group.setdefault(row['group'],[]).append(i+1)
    for values in by_group.values():merge(values)
    # Numbers refer to the fixed, hashed inventory ordering, not inferred classes.
    scenes=[([1,2,4],'beach colony'),([5,6],'jetty birds'),([9,10,11],'rock seal closeups'),
        ([12,13],'flying bird burst'),([14,15,16,17],'seals on snow/ice'),([18,19],'rock penguins'),
        ([20,21,22],'near Adelie penguin'),(list(range(24,38)),'first rocky/snow colony'),
        ([38,39,40],'swimming penguin burst'),(list(range(41,53)),'second shore colony'),
        (list(range(53,66)),'Gentoo nesting rocks'),(list(range(66,75)),'lagoon rock penguins'),
        ([76,77,78],'flying bird and roof/rock sequence'),(list(range(79,89)),'penguin ice-floe sequence'),
        ([89,90,91,92,96],'water penguins and ice'),([93,94,95],'bird at ice cliff'),
        (list(range(97,111)),'swimming seal and ice/bird sequence'),([111,112],'penguin nest closeup'),
        (list(range(113,119)),'snow hillside colony'),(list(range(119,123)),'penguin on blue ice'),
        ([123,124,127,128,129,130,131],'rock-shore landing sequence'),
        ([125,126,132,133],'flying bird second sequence'),([134,135,136],'Gentoo snow nest sequence'),
        (list(range(140,145)),'Adelie nest closeups'),([147,148,149,150],'seal and penguin rock sequence'),
        (list(range(154,168)),'late Chinstrap rookery visit')]
    for indices,label in scenes:merge(indices)
    groups={}
    for i,row in enumerate(entries):groups.setdefault(find(i),[]).append(row)
    for number,rows in enumerate(sorted(groups.values(),key=lambda rows:rows[0]['image']),1):
        for row in rows:row['group']=f'scene_{number:03d}'
    selected=set([9,10,11,12,13,20,21,22,23,145,153,*range(154,168)])
    valid_groups={entries[i-1]['group'] for i in selected}
    for index,row in enumerate(entries,1):
        row['split']='valid' if row['group'] in valid_groups else 'train'
        if index in [9,10,11]:hint='seal'
        elif index in [12,13,167]:hint='other_bird'
        elif index==153:hint='whale'
        elif index in [23,145]:hint='background'
        elif index in [20,21,22,*range(154,167)]:hint='penguin'
        else:hint='not assigned'
        row['rgb_coverage_hint']=hint
    valid=[r for r in entries if r['split']=='valid']
    if not 20<=len(valid)<=30:raise ValueError('Group-preserving validation size is outside 20-30')
    proposal=dict(status='pending_user_review',frozen=False,inventory_sha256=sha(Path(run)/'target_audit/inventory.json'),
        seed=42,entries=entries,counts=dict(Counter(r['split'] for r in entries)),
        validation_scene_groups=len({r['group'] for r in valid}),
        validation_rgb_hints=dict(Counter(r['rgb_coverage_hint'] for r in valid)),
        rgb_scene_merges=[dict(numbers=indices,description=label) for indices,label in scenes],
        only_visible_whale='5N0A8885_1.jpg is the only clearly visible whale in RGB review; proposed for validation. Target training then has no identified whale photo; source replay retains whale supervision.',
        notes='Groups include automatic RGB/EXIF similarity edges plus conservative manual RGB scene merges. Hints are not labels. NJU final-test photos are excluded.')
    save_json(Path(run)/'split_proposal.json',proposal)
    contact_sheets(run,entries,'proposed_split');contact_sheets(run,valid,'proposed_validation')
    return proposal


def freeze_split(run,*,confirmed=False):
    run=Path(run)
    if not confirmed:raise PermissionError('Review RGB contact sheets and explicitly confirm the proposed split first')
    source=run/'split_proposal.json';proposal=json.loads(source.read_text())
    frozen={**proposal,'status':'frozen','frozen':True,'review_confirmation':'User explicitly confirmed the RGB-only split proposal',
        'proposal_sha256':sha(source)}
    target=run/'split_manifest.json'
    if target.exists() and json.loads(target.read_text())!=frozen:raise ValueError('Frozen split cannot be overwritten')
    save_json(target,frozen);return frozen
