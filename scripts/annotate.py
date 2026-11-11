"""Local mask review editor. Uses saved predictions; never trains a model."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import base64
import hashlib
import io
import json
import threading

import numpy as np
from PIL import Image, ImageOps

from project_paths import PROJECT_ROOT

CLASS_NAMES = {0: "background", 1: "penguin", 2: "other_bird", 3: "seal", 4: "whale", 255: "ignore"}
UI_ROOT = Path(__file__).resolve().parent / "annotation_ui"
LOCK = threading.Lock()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_references(root=PROJECT_ROOT):
    root = Path(root)
    folder = root / "data/reference_masks/review_v1"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return folder, json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads((root / "artifacts/evaluation/latest.json").read_text(encoding="utf-8"))
    records = {r["image"]: r for r in evaluation["records"] if not r["private"]}
    names = ["penguins_far.jpg", "penguins_near.jpg", "penguin_colony.jpg"]
    names += sorted(name for name in records if name not in names)
    entries = []
    for number, name in enumerate(names, 1):
        source = root / "data/examples" / name
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw)
            width, height = image.size
        predictions = Path(records[name]["output"])
        if name == "penguins_far.jpg":
            labels = np.array(Image.open(predictions / "resize.png"), copy=True)
            native = np.asarray(Image.open(predictions / "native.png"))
            y0, y1 = int(height * .5), int(height * .625)
            labels[y0:y1] = native[y0:y1]
            rule = "Resize 0-50%; native 50-62.5%; resize 62.5-100%. Class IDs preserved."
            note = "Check distant birds, seams, missing penguins, and any remaining non-penguin labels."
        elif name == "penguins_near.jpg":
            labels = (np.asarray(Image.open(predictions / "union.png")) > 0).astype(np.uint8)
            rule = "Native/resize foreground union, with all proposed animals mapped to penguin."
            note = "Penguin-only proposal. Check back-row penguins, missing bodies, and ground false positives."
        elif name == "penguin_colony.jpg":
            labels = (np.asarray(Image.open(predictions / "union.png")) > 0).astype(np.uint8)
            labels[:int(height * .36)] = 0
            labels[int(height * .60):] = 0
            rule = "Foreground union mapped to penguin inside 36-60% height; background outside."
            note = "Check every bird and its feet. The spatial band is a draft correction, not a detection rule."
        else:
            labels = np.zeros((height, width), dtype=np.uint8)
            rule = "Proposed full-background mask based on RGB scene review."
            note = "Confirm no target animals. People, boats, vegetation, snow, and rocks are background for this task."
        if labels.shape != (height, width) or not np.isin(labels, list(CLASS_NAMES)).all():
            raise ValueError(f"Invalid seed for {name}")
        identifier = source.stem
        seed = folder / (identifier + ".seed.png")
        Image.fromarray(labels).save(seed)
        entries.append(dict(id=identifier, number=number, image=name,
            source=str(source.relative_to(root).as_posix()), image_sha256=digest(source),
            width=width, height=height, seed=seed.name, seed_sha256=digest(seed),
            seed_rule=rule, review_note=note, status="pending_review", revision=0,
            mask=None, notes=""))
    manifest = dict(schema_version=1, purpose="Human review candidates; not approved evaluation ground truth",
        class_names=CLASS_NAMES, source_evaluation=evaluation["config"]["run"],
        checkpoint_sha256=evaluation["config"]["checkpoint_sha256"], entries=entries)
    write_json(manifest_path, manifest)
    return folder, manifest


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root):
        self.root = Path(root)
        self.folder, self.manifest = prepare_references(self.root)
        super().__init__(address, AnnotationHandler)


class AnnotationHandler(BaseHTTPRequestHandler):
    def reply(self, content, mime="application/json", status=200):
        if isinstance(content, dict):
            content = json.dumps(content).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def entry(self, identifier):
        return next(e for e in self.server.manifest["entries"] if e["id"] == identifier)

    def do_GET(self):
        query = urlparse(self.path)
        try:
            if query.path == "/api/catalog":
                return self.reply(self.server.manifest)
            if query.path in ("/api/image", "/api/mask"):
                entry = self.entry(parse_qs(query.query)["id"][0])
                if query.path == "/api/image":
                    path = self.server.root / entry["source"]
                    with Image.open(path) as raw:
                        if raw.getexif().get(274, 1) == 1:
                            mime = Image.MIME.get(raw.format, "application/octet-stream")
                            return self.reply(path.read_bytes(), mime)
                        output = io.BytesIO()
                        ImageOps.exif_transpose(raw).convert("RGB").save(output, format="PNG")
                    return self.reply(output.getvalue(), "image/png")
                path = self.server.folder / (entry["mask"] or entry["seed"])
                return self.reply(path.read_bytes(), "image/png")
            files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if query.path in files:
                path = UI_ROOT / files[query.path]
                mime = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}[path.suffix]
                return self.reply(path.read_bytes(), mime)
            self.reply({"error": "Not found"}, status=404)
        except (KeyError, StopIteration, FileNotFoundError) as error:
            self.reply({"error": str(error)}, status=404)

    def do_POST(self):
        if self.path != "/api/save":
            return self.reply({"error": "Not found"}, status=404)
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
            return self.reply({"error": "Local origin required"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100 * 1024 * 1024:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(length))
            with LOCK:
                entry = self.entry(value["id"])
                if int(value["revision"]) != entry["revision"]:
                    return self.reply({"error": "This mask changed in another tab. Reload before saving."}, status=409)
                image = Image.open(io.BytesIO(base64.b64decode(value["mask_png"].split(",", 1)[1], validate=True)))
                if image.size != (entry["width"], entry["height"]):
                    raise ValueError("Mask dimensions differ from the original image")
                values = np.asarray(image.convert("RGB"))
                if not np.array_equal(values[:, :, 0], values[:, :, 1]) or not np.array_equal(values[:, :, 0], values[:, :, 2]):
                    raise ValueError("Upload a class-ID grayscale mask, not a colored overlay")
                labels = values[:, :, 0]
                if not np.isin(labels, list(CLASS_NAMES)).all():
                    raise ValueError("Mask contains an unknown class ID")
                revision = entry["revision"] + 1
                target = self.server.folder / f"{entry['id']}.edit_{revision:03d}.png"
                temporary = target.with_suffix(".tmp")
                Image.fromarray(labels).save(temporary, format="PNG")
                temporary.replace(target)
                entry.update(mask=target.name, mask_sha256=digest(target), revision=revision,
                    status="reviewed" if value.get("reviewed") is True else "manual_draft",
                    notes=str(value.get("notes", ""))[:10000], saved_at=datetime.now(timezone.utc).isoformat())
                write_json(self.server.folder / "manifest.json", self.server.manifest)
                self.reply({"status": "saved", "entry": entry})
        except (ValueError, KeyError, StopIteration, OSError) as error:
            self.reply({"error": str(error)}, status=400)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Build a standalone editor with seed masks, without embedding source photos")
    args = parser.parse_args()
    if args.offline:
        folder, manifest = prepare_references()
        seeds = {e["id"]: "data:image/png;base64," + base64.b64encode((folder / e["seed"]).read_bytes()).decode("ascii") for e in manifest["entries"]}
        bundle = json.dumps(dict(catalog=manifest, seeds=seeds)).replace("<", "\\u003c")
        html = (UI_ROOT / "index.html").read_text(encoding="utf-8")
        html = html.replace('<link rel="stylesheet" href="/style.css">', "<style>" + (UI_ROOT / "style.css").read_text(encoding="utf-8") + "</style>")
        html = html.replace('<script src="/app.js"></script>', "<script>window.REFERENCE_BUNDLE=" + bundle + ";</script><script>" + (UI_ROOT / "app.js").read_text(encoding="utf-8") + "</script>")
        target = PROJECT_ROOT / "artifacts/reference_review/editor.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        print(f"Offline annotation editor: {target} ({target.stat().st_size} bytes)")
        return
    if args.prepare_only:
        folder, manifest = prepare_references()
        print(f"Prepared {len(manifest['entries'])} pending reference masks in {folder}")
        return
    server = AnnotationServer(("127.0.0.1", args.port), PROJECT_ROOT)
    print(f"Annotation editor: http://127.0.0.1:{args.port}", flush=True)
    print(f"Local masks: {server.folder}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
