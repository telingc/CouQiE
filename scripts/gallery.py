"""Small local HTML galleries that reference images without embedding copies."""
from pathlib import Path
import html
import os
from urllib.parse import quote


def save_visualization(root,folder,entry,labels,path):
    """Save the established five-class palette and overlay without editor code.

    Extracted unchanged from the local annotation tool for public experiments.
    LLM-assisted: Codex, 2026-11-13.
    """
    import hashlib
    from PIL import Image,ImageOps,ImageDraw
    def sha(file):
        with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    palette=[0]*768
    for c,rgb in {0:(166,133,245),1:(255,65,77),2:(255,167,38),3:(0,190,220),4:(61,190,90),255:(155,163,175)}.items():
        palette[c*3:c*3+3]=rgb
    colored=Image.fromarray(labels).convert('P');colored.putpalette(palette)
    color=path.with_suffix('.color.png');colored.convert('RGB').save(color)
    with Image.open(Path(root)/entry['source']) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
    image.thumbnail((1920,1920));colors=colored.convert('RGB').resize(image.size,Image.Resampling.NEAREST)
    overlay=Image.blend(image,colors,.42)
    draw=ImageDraw.Draw(overlay);draw.rectangle((0,0,min(overlay.width,690),30),fill=(20,28,40))
    draw.text((8,8),'Background purple | Penguin red | Bird orange | Seal cyan | Whale green | Ignore gray',fill='white')
    preview=path.with_suffix('.overlay.jpg');overlay.save(preview,quality=92)
    return dict(color_mask=str(color.relative_to(folder).as_posix()),color_mask_sha256=sha(color),
        overlay=str(preview.relative_to(folder).as_posix()),overlay_sha256=sha(preview))


def write_gallery_html(path,title,items,description="Local predictions; no ground-truth photo IoU."):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    parts=["<!doctype html><meta charset='utf-8'>",f"<title>{html.escape(title)}</title>",
        "<style>body{max-width:1500px;margin:24px auto;font:16px system-ui;background:#f5f6fa;color:#182038}section{background:white;padding:16px;margin:18px 0}img{width:100%;height:auto}h2{font-size:18px}</style>",
        f"<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p>"]
    for name,image in items:
        relative=os.path.relpath(Path(image).resolve(),path.parent.resolve()).replace("\\","/")
        parts.append(f"<section><h2>{html.escape(str(name))}</h2><img loading='lazy' src='{quote(relative,safe='/')}'></section>")
    path.write_text("\n".join(parts),encoding="utf-8")
    return path
