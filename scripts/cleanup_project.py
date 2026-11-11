"""Reversible layout migration. Historical files and source snapshots are kept."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MOVES = {
    "experiments": "artifacts/experiments", "runs": "artifacts/runs",
    "examples": "data/examples", "nju_testset": "data/nju_testset",
    **{name: "data/"+name for name in ("dataset_birds", "dataset_penguins", "dataset_seals", "dataset_wildlife", "dataset_nature")},
    "images": "data/reference_images",
}
RETAINED = []


def relocate(source, target):
    """Move unlocked entries; retain byte-verified copies of locked originals."""
    for path in (source, target):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Unsafe relocation: {path}")
    try:
        source.rename(target)
        return
    except PermissionError:
        pass
    if source.is_dir():
        target.mkdir(exist_ok=True)
        for child in list(source.iterdir()):
            relocate(child,target/child.name)
        try:
            source.rmdir()
        except PermissionError:
            RETAINED.append(str(source.relative_to(ROOT)))
    else:
        shutil.copy2(source,target)
        with source.open("rb") as a,target.open("rb") as b:
            if hashlib.file_digest(a,"sha256").digest()!=hashlib.file_digest(b,"sha256").digest():
                raise ValueError(f"Relocation verification failed: {source}")
        try:
            source.unlink()
        except PermissionError:
            RETAINED.append(str(source.relative_to(ROOT)))


def checked(relative):
    path = (ROOT/relative).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Path outside the workspace: {path}")
    return path


def remap_string(value):
    for old, new in MOVES.items():
        for sep in ("/", "\\"):
            prefix = str(ROOT/old).replace("\\", sep).replace("/", sep)
            if value == prefix or value.startswith(prefix+sep):
                return str(ROOT/new).replace("\\", sep).replace("/", sep)+value[len(prefix):]
        for sep in ("/", "\\"):
            if value.startswith(old+sep):
                return new.replace("/", sep)+value[len(old):]
    return value


def remap(value):
    if isinstance(value, str):
        return remap_string(value)
    if isinstance(value, list):
        return [remap(v) for v in value]
    if isinstance(value, dict):
        return {k: remap(v) for k, v in value.items()}
    return value


def migrate():
    archive = checked("artifacts/archive")
    archive.mkdir(parents=True, exist_ok=True)
    if (archive/"layout_migration.json").exists():
        raise RuntimeError("Migration already recorded; do not run twice")
    subprocess.run(["git", "bundle", "create", str(archive/"before_cleanup.bundle"), "--all"], cwd=ROOT, check=True)
    originals = list(ROOT.glob("*.py"))+[ROOT/"launch.ipynb",ROOT/"README.md",ROOT/"DATASETS.md",ROOT/".gitignore"]
    with zipfile.ZipFile(archive/"original_sources.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in originals:
            if path.exists():
                z.write(path, path.name)
    records=[]
    for old,new in MOVES.items():
        source,target=checked(old),checked(new)
        if target.exists():
            raise FileExistsError(target)
        if source.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            relocate(source,target)
            records.append({"from":old,"to":new})
    # Preserve original JSON byte sequences before updating relocated paths.
    metadata=list(checked("artifacts/experiments").rglob("*.json"))+list(checked("artifacts/runs").rglob("*.json"))
    with zipfile.ZipFile(archive/"original_metadata.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in metadata:
            z.write(path,str(path.relative_to(ROOT)))
    changed=0
    for path in metadata:
        data=json.loads(path.read_text(encoding="utf-8-sig"))
        relocated=remap(data)
        if relocated!=data:
            path.write_text(json.dumps(relocated,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            changed+=1
    scripts=checked("scripts")
    for path in list(ROOT.glob("*.py")):
        text=path.read_text(encoding="utf-8-sig")
        # Historical prose logs are redundant with structured metadata.
        tree=ast.parse(text)
        lines=text.splitlines(keepends=True)
        edits=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=="note":
                segment=ast.get_source_segment(text,node) or ""
                if re.search(r"[\u3400-\u9fff]",segment):
                    folder=ast.get_source_segment(text,node.value.args[0])
                    indent=" "*node.col_offset
                    edits.append((node.lineno-1,node.end_lineno,indent+f'note({folder}, "Run status updated. See structured run metadata.")\n'))
        for start,end,replacement in sorted(edits,reverse=True):
            lines[start:end]=[replacement]
        text="".join(lines)
        text=text.replace("Path(__file__).resolve().parent", "PROJECT_ROOT")
        text=text.replace("Path(__file__).parent", "PROJECT_ROOT")
        text="from project_paths import PROJECT_ROOT\n"+text
        # Insert after module docstring so API documentation remains intact.
        tree=ast.parse(text)
        if len(tree.body)>1 and isinstance(tree.body[1],ast.Expr) and isinstance(tree.body[1].value,ast.Constant) and isinstance(tree.body[1].value.value,str):
            doc=tree.body[1];ls=text.splitlines(keepends=True);ls.pop(0);ls.insert(doc.end_lineno-1,"\nfrom project_paths import PROJECT_ROOT\n");text="".join(ls)
        for old,new in (("experiments/","artifacts/experiments/"),("runs/","artifacts/runs/"),("examples/","data/examples/"),("nju_testset/","data/nju_testset/")):
            text=text.replace(old,new)
        for old,new in (("experiments","artifacts/experiments"),("examples","data/examples"),("nju_testset","data/nju_testset")):
            # Path operands only: source/group identifiers must remain unchanged.
            text=re.sub(r"(/\s*)([\"'])"+old+r"\2",lambda m:m[1]+m[2]+new+m[2],text)
        text=text.replace("Gentoo_Penguin_colony.jpg","penguin_colony.jpg")
        if path.name=="animal_data.py":
            text=text.replace("root / s, split",'root / "data" / s, split')
            text=text.replace('root / "dataset_nature", split','root / "data" / "dataset_nature", split')
            text=text.replace("root / source / split",'root / "data" / source / split')
        if path.name=="script.py":
            text=text.replace("def prepare_all_datasets():\n    root = PROJECT_ROOT",'def prepare_all_datasets():\n    root = PROJECT_ROOT / "data"')
        if path.name=="train_multiclass.py":
            text=text.replace('"experiments" / f"exp{number}_results"','"artifacts/experiments" / f"exp{number}_results"')
        if path.name=="experiment13.py":
            text=text.replace("return _run_stages(12,","return _run_stages(13,")
        if re.search(r"[\u3400-\u9fff]",text):
            raise ValueError(f"Untranslated active source: {path.name}")
        compile(text,str(scripts/path.name),"exec")
        (scripts/path.name).write_text(text,encoding="utf-8")
        path.unlink()
    helper=checked("data/dataset_nature/download_and_prepare.py")
    if helper.exists():
        text=helper.read_text(encoding="utf-8").replace('ROOT = Path(__file__).resolve().parent','ROOT = Path(__file__).resolve().parents[1] / "data/dataset_nature"')
        (scripts/"download_nature.py").write_text(text,encoding="utf-8")
        helper.unlink()
    for name in ("data_preparation_report.json","training_data_report.json"):
        path=ROOT/name
        if path.exists():
            target=checked("artifacts/data_reports/"+name);target.parent.mkdir(parents=True,exist_ok=True);path.rename(target)
    (archive/"layout_migration.json").write_text(json.dumps({"moves":records,"retained_locked_originals":RETAINED,"relocated_json_files":changed,"historical_checkpoint_payloads_unchanged":True,"backup":"before_cleanup.bundle","note":"Original sources and metadata are archived. Old source hashes refer to pre-migration files."},indent=2),encoding="utf-8")
    print(json.dumps({"moves":records,"updated_json_files":changed},indent=2))


if __name__=="__main__":
    migrate()
