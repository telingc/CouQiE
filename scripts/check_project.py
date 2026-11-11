"""Read-only closeout checks; no torch import, downloads, training or inference."""
from argparse import ArgumentParser
import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from project_paths import PROJECT_ROOT

RETIRED={'cleanup_project','finalize_cleanup','modernize_galleries','audit_cleanup','preview_exp10_scales'}
PACKAGES=('torch','torchvision','numpy','Pillow','matplotlib','pandas','psutil','scipy','requests','nbformat')
LINK=re.compile(r'!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+)(?:\s+"[^"]*")?)\)')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def module_exports(tree):
    names=set()
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):names.add(node.name)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            names.update(a.asname or a.name.split('.')[0] for a in node.names)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:names.update(n.id for n in ast.walk(target) if isinstance(n,ast.Name))
    return names


def check_project(root=None,*,local_results=False):
    root=Path(root or PROJECT_ROOT).resolve();errors=[];warnings=[]
    trees={}
    for path in sorted((root/'scripts').glob('*.py')):
        try:trees[path.stem]=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        except SyntaxError as error:errors.append(f'Syntax: {path.name}: {error}')
    exports={name:module_exports(tree) for name,tree in trees.items()}
    for name,tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module:
                module=node.module.split('.')[0]
                if module in RETIRED:errors.append(f'{name} imports archived module {module}')
                if module in exports:
                    for alias in node.names:
                        if alias.name!='*' and alias.name not in exports[module]:errors.append(f'{name}: missing local export {module}.{alias.name}')
    for name in RETIRED:
        if name in trees:errors.append(f'Archived source remains an active entry point: {name}')
    if list(root.glob('*.py')):errors.append('Python entry points must remain under scripts/')
    notebook=json.loads((root/'launch.ipynb').read_text(encoding='utf-8'))
    switches={}
    for index,cell in enumerate(notebook['cells']):
        source=''.join(cell.get('source',[]))
        if cell.get('attachments') or cell.get('outputs'):errors.append(f'Notebook cell {index} embeds saved content')
        if cell['cell_type']=='markdown' and '![' in source:errors.append(f'Notebook cell {index} renders a large image; use a plain link')
        if cell['cell_type']=='code':
            try:tree=ast.parse(source)
            except SyntaxError as error:errors.append(f'Notebook cell {index}: {error}');continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id.startswith('RUN_'):
                            value=node.value.value if isinstance(node.value,ast.Constant) else None
                            switches[target.id]=value
                            if value is not False:errors.append(f'Notebook switch {target.id} must default to False')
    documents=sorted(root.glob('*.md'))+sorted((root/'reports').rglob('*.md'));links=0
    for path in documents:
        text=re.sub(r'```.*?```','',path.read_text(encoding='utf-8'),flags=re.S)
        for match in LINK.finditer(text):
            raw=match.group(1) or match.group(2)
            if raw.startswith('#') or urlsplit(raw).scheme in ('http','https','mailto','data','codex'):continue
            target=unquote(raw.split('#',1)[0].split('?',1)[0]);target=re.sub(r':\d+$','',target)
            destination=Path(target) if Path(target).is_absolute() else path.parent/target
            destination=destination.resolve();links+=1
            if not destination.exists():
                message=f'{path.relative_to(root)}: missing local link {raw}'
                if destination.is_relative_to(root/'artifacts') or destination.is_relative_to(root/'data'):warnings.append(message+' (local-only storage)')
                else:errors.append(message)
            if match.group(0).startswith('!') and ('nju_testset' in destination.parts or 'nju' in destination.parts or destination.name.startswith('5N0A')):
                errors.append(f'Public Markdown references private NJU pixels: {path.relative_to(root)}')
    with (root/'reports/index.csv').open(encoding='utf-8-sig',newline='') as stream:index=list(csv.DictReader(stream))
    ids=[int(row['experiment']) for row in index]
    if sorted(ids)!=list(range(1,15)):errors.append('Formal index must contain experiments 1-14 exactly once')
    for row in index:
        if not (root/row['report']).is_file():errors.append(f"Missing experiment report: {row['report']}")
    try:
        tracked=subprocess.check_output(['git','ls-files','-z','--','data','artifacts'],cwd=root).decode().split('\0')
        if any(tracked):errors.append('Private data or artifacts are tracked by Git')
    except (FileNotFoundError,subprocess.CalledProcessError):warnings.append('Git tracking check unavailable')
    versions={}
    for package in PACKAGES:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None;warnings.append(f'Runtime package not installed: {package}')
    result_checks={}
    if local_results:
        # Read metadata and hash files only. Never instantiate a neural network.
        try:
            parent=root/'artifacts/experiments'
            baseline=Path(json.loads((parent/'exp13_results/latest_4class.json').read_text())['folder'])
            training=json.loads((baseline/'training_result.json').read_text())
            if training['completed_epochs']!=10:errors.append('Experiment 13 did not complete ten epochs')
            scale=json.loads((parent/'exp14_results/latest.json').read_text())
            if scale['status']!='complete' or scale['completed']!=75:errors.append('Experiment 14 is not complete (75 cases)')
            protocol=json.loads((Path(scale['folder'])/'protocol.json').read_text())
            if Path(protocol['checkpoint']).resolve()!=(baseline/'best.pt').resolve():errors.append('Current experiment-13 baseline differs from experiment-14 fixed weights')
            if protocol['class_names']!=['background','penguin','other_bird','seal']:errors.append('Experiment 14 class schema changed')
            if digest(Path(protocol['checkpoint']))!=protocol['checkpoint_sha256']:errors.append('Experiment 14 checkpoint SHA256 mismatch')
            manifest=root/'data/reference_masks/benchmark_v2/manifest.json'
            if digest(manifest)!=protocol['reference_manifest_sha256']:errors.append('Frozen reference manifest changed')
            for entry in protocol['entries']:
                if digest(Path(entry['image_path']))!=entry['image_sha256']:errors.append(f"Changed source photo: {entry['image']}")
                if entry['mask_path'] and digest(Path(entry['mask_path']))!=entry['mask_sha256']:errors.append(f"Changed reference mask: {entry['image']}")
            point_path=baseline/'far_point_protocol.json'
            if digest(point_path)!=protocol['far_point_protocol_sha256']:errors.append('Frozen far-point protocol changed')
            records=json.loads((Path(scale['folder'])/'progress.json').read_text())
            if len(records)!=75 or any(r['status']!='complete' or r['precision']!='float32' or r['amp'] or r['tf32'] for r in records):errors.append('Experiment 14 case protocol mismatch')
            result_checks=dict(exp13_epochs=training['completed_epochs'],exp13_best_epoch=training['best_epoch'],
                exp14_completed=scale['completed'],checkpoint_sha256=protocol['checkpoint_sha256'],reference_version='benchmark_v2')
        except (OSError,KeyError,ValueError) as error:errors.append(f'Local result metadata: {error}')
    return dict(status='passed' if not errors else 'failed',python_modules=len(trees),documentation_files=len(documents),
        checked_local_links=links,formal_experiments=ids,notebook_switches=switches,
        local_results=result_checks,package_versions=versions,errors=errors,warnings=warnings,
        training_performed=False,inference_performed=False)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--local-results',action='store_true')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=check_project(local_results=args.local_results)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='passed' else 1)
