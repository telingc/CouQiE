"""Experiment 11 and shared staged-training helpers. Nothing trains on import.

Fine-tune the complete five-class 5x5 checkpoint, retaining its head and BN
buffers. A new optimizer starts each stage; an interrupted stage resumes its
own optimizer/scaler/RNG state. LLM-assisted teaching code, 2026-11-10.
"""

from project_paths import PROJECT_ROOT
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Subset, RandomSampler, WeightedRandomSampler

from animal_data import CLASS_NAMES
from script import Trainer, ResNetUNet, load_initial_animal_model
from train_multiclass import (experiment_data, experiment_folder, make_loader,
    close_loader, cleanup_cuda, prepare_runtime, seed_everything, write_json, note)


DEFAULT_REPLAY_PROBABILITIES = {
    'background': .20, 'penguin': .50, 'other_bird': .10, 'seal': .10, 'whale': .10,
}


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def best_5x5_checkpoint(root=None):
    """Explicit kernel-5 index; latest.json may instead point to kernel 7."""
    root = Path(root or PROJECT_ROOT).resolve()
    result = json.loads((root/'artifacts/experiments/exp8_results/latest_kernel5.json').read_text(encoding='utf-8'))
    return Path(result['selected']['checkpoint'])


def _records(train):
    return [(child.root.name, record['sample_class'])
            for child in train.datasets for record in child.records]


def _stage_plan(records, stage):
    """Select by source OR dominant-image group; preserve all pixel labels."""
    if not isinstance(stage['epochs'], int) or stage['epochs'] < 1 or stage['learning_rate'] <= 0:
        raise ValueError('epochs must be a positive integer and learning_rate positive')
    sources, groups = stage.get('sources'), stage.get('groups')
    if sources and groups:
        raise ValueError('Use either sources or groups in one stage')
    unknown = set(groups or ()) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f'Unknown sampling groups: {unknown}')
    probabilities = stage.get('probabilities')
    if probabilities is not None:
        if sources or groups or set(probabilities) != set(CLASS_NAMES):
            raise ValueError('Weighted replay requires probabilities for all five groups')
        if any(p <= 0 for p in probabilities.values()) or abs(sum(probabilities.values())-1) > 1e-9:
            raise ValueError('Group probabilities must be positive and sum to one')
    indices = [i for i, (source, c) in enumerate(records)
               if (not sources or source in sources) and (not groups or CLASS_NAMES[c] in groups)]
    counts = Counter(CLASS_NAMES[records[i][1]] for i in indices)
    if not indices or (probabilities is not None and set(counts) != set(CLASS_NAMES)):
        raise ValueError(f"Empty/incomplete stage: {stage['name']}")
    if sources and set(sources) - {records[i][0] for i in indices}:
        raise ValueError('A requested dataset source is missing')
    draws = stage.get('samples_per_epoch') if probabilities else len(indices)
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('samples_per_epoch must be positive for weighted replay')
    plan = {**stage, 'train_images': len(indices), 'draws_per_epoch': draws,
        'total_draws': draws*stage['epochs'], 'sampling_group_counts': dict(counts),
        'source_counts': dict(Counter(records[i][0] for i in indices)),
        'sampling': 'weighted_with_replacement' if probabilities else 'shuffle_without_replacement',
        'indices_sha256': hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()}
    if probabilities:
        plan['expected_draws_per_image_per_epoch'] = {
            name: draws*probabilities[name]/counts[name] for name in CLASS_NAMES}
    return indices, plan


def _sampler(subset, records, indices, stage, seed):
    generator = torch.Generator().manual_seed(seed)
    if stage.get('probabilities'):
        counts = stage['sampling_group_counts']
        weights = [stage['probabilities'][CLASS_NAMES[records[i][1]]]/counts[CLASS_NAMES[records[i][1]]]
                   for i in indices]
        return WeightedRandomSampler(weights, stage['draws_per_epoch'], replacement=True, generator=generator)
    return RandomSampler(subset, replacement=False, generator=generator)


def _load_model(path, mode, seed):
    seed_everything(seed)
    if mode == 'binary_features_fresh_head':
        return load_initial_animal_model(path, num_classes=5, kernel_size=5, seed=seed)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if tuple(checkpoint['class_names']) != CLASS_NAMES or checkpoint['config']['kernel_size'] != 5:
        raise ValueError('Expected a complete five-class 5x5 checkpoint')
    model = ResNetUNet(num_classes=5, pretrained=False, kernel_size=5)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model


def _run_stages(number, stages, *, root, run_dir, initial_checkpoint,
                initial_mode, run_training=False, batch_size=6, seed=42,
                workers=2, bn_policy='normal', show_plots=True, device=None):
    root = Path(root).resolve()
    if not stages or any(not s['name'].replace('_','').isalnum() for s in stages):
        raise ValueError('Provide named stages using letters/digits/underscores')
    if len({s['name'] for s in stages}) != len(stages):
        raise ValueError('Stage names must be unique')
    if not isinstance(batch_size,int) or batch_size<1 or workers<0 or bn_policy not in ('normal','frozen_stats'):
        raise ValueError('Invalid batch_size, workers or bn_policy')
    folder = experiment_folder(root, number)
    run = Path(run_dir).resolve() if run_dir else folder/'run1'
    run.mkdir(parents=True, exist_ok=True)
    data, metadata = experiment_data(str(root), 640, True)
    records = _records(data['train'])
    compiled = [_stage_plan(records, stage) for stage in stages]
    initial = Path(initial_checkpoint).resolve()
    request = dict(experiment=number, initial_checkpoint=str(initial), initial_mode=initial_mode,
        initial_checkpoint_sha256=_hash(initial), kernel_size=5, class_names=list(CLASS_NAMES),
        batch_size=batch_size, seed=seed, workers=workers, bn_policy=bn_policy,
        image_size=640, device=str(device or 'cuda:0'),
        amp=torch.device(device or 'cuda:0').type=='cuda',
        gradient_clip=1., stages=[p for _, p in compiled],
        stage_handoff='previous last.pt model only; fresh optimizer/scaler at next stage',
        annotation_manifest_sha256=metadata['annotation_manifest_sha256'])
    previous_request = run/'request.json'
    if previous_request.exists():
        previous = json.loads(previous_request.read_text(encoding='utf-8'))
        # Editing a plan before any optimization is allowed. Trained runs stay immutable.
        if previous != request and (any(run.glob('stage_*/*.pt')) or (run/'result.json').exists()):
            raise ValueError('This run already has training state. Use a different run_dir for changed settings.')
    elif any(run.glob('stage_*/*.pt')):
        raise ValueError('Found checkpoints without their request.json; choose another run_dir')
    write_json(previous_request, request)
    write_json(run/'data.json', metadata)
    write_json(run/'plan.json', request)
    print(pd.DataFrame([{k:p[k] for k in ('name','epochs','learning_rate','train_images','draws_per_epoch','total_draws')}
                        for _, p in compiled]).to_string(index=False), flush=True)
    print('Result folder:', run, flush=True)
    if not run_training:
        print('PREPARATION ONLY: no model optimization or GPU inference.', flush=True)
        write_json(folder/'latest_plan.json', {'folder':str(run), 'plan':request})
        return dict(status='prepared', experiment=number, folder=str(run), plan=request)
    if (run/'result.json').exists():
        result = json.loads((run/'result.json').read_text(encoding='utf-8'))
        if show_plots:
            plot_staged_results(result)
        return result
    device = torch.device(device or 'cuda:0')
    if device.type == 'cuda':
        prepare_runtime()
    else:
        torch.set_num_threads(4)
    sources = ('experiment11.py', 'experiment12.py', 'script.py', 'animal_data.py', 'train_multiclass.py')
    if not (run/'training_source_hashes.json').exists():
        write_json(run/'training_source_hashes.json', {n:_hash(root/'scripts'/n) for n in sources})
    animal_valid = ConcatDataset([d for d in data['valid'].datasets if d.root.name != 'dataset_nature'])
    background_valid = next(d for d in data['valid'].datasets if d.root.name == 'dataset_nature')
    valid_loader = make_loader(animal_valid, batch_size, workers=workers)
    background_loader = make_loader(background_valid, batch_size, workers=workers)
    results = []
    source, mode = initial, initial_mode
    note(folder, "Run status updated. See structured run metadata.")
    try:
        for index, (indices, plan) in enumerate(compiled, 1):
            stage_dir = run/f"stage_{index:02d}_{plan['name']}"
            stage_dir.mkdir(exist_ok=True)
            if (stage_dir/'result.json').exists():
                stage_result = json.loads((stage_dir/'result.json').read_text(encoding='utf-8'))
                if not Path(stage_result['checkpoint']).exists() or not Path(stage_result['last_checkpoint']).exists():
                    raise FileNotFoundError('Completed-stage checkpoints are missing')
                results.append(stage_result)
                source, mode = Path(stage_result['last_checkpoint']), 'full_five_class_weights'
                continue
            print(f"\n=== Stage {index}/{len(compiled)}: {plan['name']} | {plan['epochs']} epochs ===", flush=True)
            stage_seed = seed+index-1
            last_path = stage_dir/'last.pt'
            model = _load_model(last_path if last_path.exists() else source,
                                'full_five_class_weights' if last_path.exists() else mode, stage_seed).to(device)
            subset = Subset(data['train'], indices)
            sampler = _sampler(subset, records, indices, plan, stage_seed)
            train_loader = make_loader(subset, batch_size, sampler=sampler, workers=workers)
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'])
            trainer = Trainer(model, train_loader, valid_loader, torch.nn.CrossEntropyLoss(ignore_index=255),
                optimizer, device, stage_dir, class_names=CLASS_NAMES, amp=device.type=='cuda',
                bn_policy=bn_policy, gradient_clip=1., log_every=0, background_loader=background_loader,
                dataset_metadata={**metadata, 'stage_plan':plan, 'stage_input_checkpoint':str(source)})
            trainer.config.update(epochs=plan['epochs'], seed=stage_seed, learning_rate=plan['learning_rate'],
                image_size=640, experiment=number, stage_index=index, stage_name=plan['name'],
                samples_per_epoch=plan['draws_per_epoch'], initial_checkpoint=str(source))
            try:
                if last_path.exists():
                    trainer.resume(last_path)
                    print('Resuming completed stage epochs:', trainer.epoch, flush=True)
                elif not (stage_dir/'start_metrics.json').exists():
                    trainer.validate()
                    write_json(stage_dir/'start_metrics.json', trainer.validation_metrics)
                write_json(stage_dir/'config.json', trainer.config)
                if device.type=='cuda':
                    torch.cuda.reset_peak_memory_stats()
                trainer.fit(max(0, plan['epochs']-trainer.epoch))
                best = torch.load(stage_dir/'best.pt', map_location='cpu', weights_only=True, mmap=True)
                stage_result = dict(status='complete', stage_index=index, stage_name=plan['name'],
                    folder=str(stage_dir), checkpoint=str(stage_dir/'best.pt'), last_checkpoint=str(last_path),
                    best_epoch=best['epoch'], metrics=best['valid_metrics'], config=trainer.config,
                    completed_epochs=trainer.epoch, seconds=sum(r['epoch_seconds'] for r in trainer.history),
                    final_metrics=trainer.history[-1]['valid_metrics'],
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None)
                del best
                write_json(stage_dir/'result.json', stage_result)
                results.append(stage_result)
                # Deliberately transfer the LAST, not the best, stage weights.
                source, mode = last_path, 'full_five_class_weights'
                note(folder, "Run status updated. See structured run metadata.")
            finally:
                close_loader(train_loader)
                del trainer, optimizer, model
                cleanup_cuda()
    finally:
        close_loader(valid_loader)
        close_loader(background_loader)
        cleanup_cuda()
    best = max(results, key=lambda r:r['metrics']['foreground_miou'])
    result = dict(status='complete', experiment=number, folder=str(run), selected=best,
        stages=results, final_checkpoint=results[-1]['last_checkpoint'],
        final_metrics=results[-1]['final_metrics'], total_epochs=sum(r['completed_epochs'] for r in results),
        initial_metrics=json.loads((Path(results[0]['folder'])/'start_metrics.json').read_text(encoding='utf-8')))
    result['improved_over_initial'] = best['metrics']['foreground_miou'] > result['initial_metrics']['foreground_miou']
    write_json(run/'result.json', result)
    write_json(run/'best_overall.json', best)
    write_json(folder/'latest.json', result)
    _export_history(result)
    note(folder, "Run status updated. See structured run metadata.")
    if show_plots:
        plot_staged_results(result)
    return result


def _export_history(result):
    rows=[];global_epoch=0
    for stage in result['stages']:
        history=json.loads((Path(stage['folder'])/'history.json').read_text(encoding='utf-8'))
        for r in history:
            global_epoch+=1
            rows.append(dict(global_epoch=global_epoch, stage=stage['stage_name'], stage_epoch=r['epoch'],
                train_loss=r['train_loss'], valid_loss=r['valid_loss'], foreground_miou=r['foreground_miou'],
                background_fp_fraction=r['valid_metrics']['background_holdout']['false_positive_fraction'],
                **{name+'_iou':r['iou_per_class'][i] for i,name in enumerate(CLASS_NAMES)}))
    table=pd.DataFrame(rows)
    table.to_csv(Path(result['folder'])/'all_stages_history.csv',index=False)
    return table


def plot_staged_results(result):
    """Plot saved metrics only. No model construction, inference or training."""
    import matplotlib.pyplot as plt
    table=_export_history(result)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    axes[0].plot(table.global_epoch,table.foreground_miou,'o-')
    axes[0].set(title='All-animal validation foreground mIoU',xlabel='Global epoch',ylim=(0,1))
    for name in CLASS_NAMES[1:]:
        axes[1].plot(table.global_epoch,table[name+'_iou'],label=name)
    axes[1].set(title='Per-class validation IoU',xlabel='Global epoch',ylim=(0,1));axes[1].legend()
    axes[2].plot(table.global_epoch,100*table.background_fp_fraction,'o-')
    axes[2].set(title='Nature validation false foreground',xlabel='Global epoch',ylabel='Labeled background pixels (%)')
    boundary=0
    for stage in result['stages'][:-1]:
        boundary+=stage['completed_epochs']
        for ax in axes:ax.axvline(boundary+.5,color='gray',linestyle='--')
    fig.tight_layout();fig.savefig(Path(result['folder'])/'training_curves.png',dpi=140)
    plt.show();plt.close(fig)
    print('Best trained stage:',result['selected']['stage_name'],'epoch',result['selected']['best_epoch'])
    print('Best trained checkpoint:',result['selected']['checkpoint'])
    print('Final checkpoint:',result['final_checkpoint'])
    print('Foreground mIoU exceeded starting model:',result['improved_over_initial'])


def run_experiment11(*, root=None, run_dir=None, run_training=False, epochs=5,
                     learning_rate=1e-5, samples_per_epoch=2000, probabilities=None,
                     initial_checkpoint=None, batch_size=6, workers=2, seed=42,
                     bn_policy='normal', show_plots=True, device=None):
    """Penguin-focused replay fine-tuning; defaults are an untested candidate.

    run_training=False only prepares the plan. Set True locally to optimize.
    All encoder/decoder/head weights remain trainable; no head reinitialization.
    """
    root=Path(root or PROJECT_ROOT).resolve()
    stage=dict(name='penguin_replay', epochs=epochs, learning_rate=learning_rate,
        probabilities=dict(DEFAULT_REPLAY_PROBABILITIES if probabilities is None else probabilities),
        samples_per_epoch=samples_per_epoch)
    return _run_stages(11,[stage],root=root,run_dir=run_dir,
        initial_checkpoint=initial_checkpoint or best_5x5_checkpoint(root),
        initial_mode='full_five_class_weights',run_training=run_training,
        batch_size=batch_size,workers=workers,seed=seed,bn_policy=bn_policy,
        show_plots=show_plots,device=device)
