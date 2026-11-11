"""Small local HTML galleries that reference images without embedding copies."""
from pathlib import Path
import html
import os
from urllib.parse import quote


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
