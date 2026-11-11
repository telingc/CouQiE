"""Experiment 12: strict 10 + 15 + 5 + 5 + 5 epoch sequence, no replay.

Sources in stage 1; dominant-image animal groups in later stages. Each chosen
image is visited once per epoch, shuffled without replacement. LLM-assisted,
2026-11-10. Default call prepares a plan only, without training.
"""

from project_paths import PROJECT_ROOT
from pathlib import Path
from experiment11 import _run_stages, plot_staged_results, best_5x5_checkpoint


def default_stages():
    """Unvalidated candidate learning rates; edit these explicitly if desired."""
    return [
        dict(name='background_wildlife',sources=['dataset_nature','dataset_wildlife'],
             epochs=10,learning_rate=1e-4),
        dict(name='other_bird',groups=['other_bird'],epochs=15,learning_rate=3e-5),
        dict(name='whale',groups=['whale'],epochs=5,learning_rate=1e-5),
        dict(name='seal',groups=['seal'],epochs=5,learning_rate=1e-5),
        dict(name='penguin',groups=['penguin'],epochs=5,learning_rate=1e-5),
    ]


def run_experiment12(*, root=None, run_dir=None, run_training=False, stages=None,
                     initial_checkpoint=None, batch_size=6, workers=2, seed=42,
                     bn_policy='normal', show_plots=True, device=None):
    """Independent experiment, not a continuation of experiment 11.

    None starts with the original binary feature checkpoint plus a fresh
    five-class head, matching prior from-initialization experiments. To start
    with the old complete 5x5 model instead, explicitly pass
    initial_checkpoint=best_5x5_checkpoint(root).

    All stages preserve the five output classes and existing labels. Later
    single-group stages do not silently add replay; forgetting is measurable
    through the unchanged full-animal and background validation after each epoch.
    """
    root=Path(root or PROJECT_ROOT).resolve()
    checkpoint=Path(initial_checkpoint) if initial_checkpoint is not None else root/'artifacts/runs/penguin_baseline/best.pt'
    mode='full_five_class_weights' if initial_checkpoint is not None else 'binary_features_fresh_head'
    return _run_stages(12,default_stages() if stages is None else stages,
        root=root,run_dir=run_dir,initial_checkpoint=checkpoint,initial_mode=mode,
        run_training=run_training,batch_size=batch_size,workers=workers,seed=seed,
        bn_policy=bn_policy,show_plots=show_plots,device=device)
