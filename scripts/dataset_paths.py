"""Resolve active datasets and immutable pre-migration source snapshots.

LLM-assisted: Codex, 2026-11-12.
"""
from pathlib import Path
import json

from project_paths import PROJECT_ROOT


def data_location(name,root=PROJECT_ROOT):
    """Read an active dataset, or find its archived original without recreating it."""
    root=Path(root).resolve();current=root/'data'/name
    if current.exists():return current
    index=root/'artifacts/data_archive/index.json'
    if index.exists():
        for snapshot in reversed(json.loads(index.read_text())['snapshots']):
            candidate=Path(snapshot['root'])/'data'/name
            if candidate.exists():return candidate
    return current


def historical_source_root(root,expected_metadata):
    """Pin an existing run by its source manifest hashes; never substitute new data."""
    root=Path(root).resolve();index=root/'artifacts/data_archive/index.json'
    if not index.exists():return root
    expected=expected_metadata['annotation_manifest_sha256']
    for snapshot in reversed(json.loads(index.read_text())['snapshots']):
        folder=Path(snapshot['root']);metadata=json.loads((folder/'legacy_metadata.json').read_text())
        available=metadata['annotation_manifest_sha256']
        if all(available.get(key)==value for key,value in expected.items()):return folder
    # A run audited against the active merged layout should keep using that view.
    if 'dataset_wildlife' not in {name for values in expected_metadata['sources'].values() for name in values}:return root
    raise ValueError('No archived source snapshot matches this historical run')
