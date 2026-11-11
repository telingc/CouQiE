"""Experiment 5: five-class training with the prepared animal datasets.

LLM-assisted: Codex, 2026-11-09. Reuses the existing training builder/Trainer;
run_training=False only prepares the configuration and prints a short summary.
"""

from project_paths import PROJECT_ROOT

from train_multiclass import build_multiclass_trainer


def show_saved_experiment5(root=None):
    """Show the completed three-epoch historical baseline without training."""
    import json
    from pathlib import Path
    import pandas as pd
    from IPython.display import display, Image
    root = Path(root or PROJECT_ROOT)
    folder = root / "artifacts/runs/antarctic_batch_test_20261109_192421"
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    display(pd.DataFrame([{k: r[k] for k in ("epoch", "train_loss", "valid_loss", "foreground_miou")}
                          for r in history]).round(4))
    for name in ("training_curves.png", "validation_examples_1.png", "validation_examples_2.png"):
        display(Image(filename=str(folder / name)))
    print("Historical best checkpoint:", folder / "best_antarctic_multiclass_tuned.pt")


def run_experiment5(
    *,
    epochs=10,
    batch_size=1,
    learning_rate=1e-4,
    run_training=False,
    root=None,
):
    """Return the Trainer; start optimization only when explicitly enabled."""
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    trainer, train_dataset, valid_dataset = build_multiclass_trainer(
        root=root, batch_size=batch_size, learning_rate=learning_rate,
    )
    print("Experiment 5: five-class animal segmentation")
    print("Classes:", ", ".join(trainer.class_names))
    print(f"Train / valid images: {len(train_dataset)} / {len(valid_dataset)}")
    print(f"Epochs: {epochs} | batch size: {batch_size} | learning rate: {learning_rate:g}")
    print("Checkpoint:", trainer.checkpoint_dir / trainer.checkpoint_name)
    if run_training:
        trainer.fit(epochs)
    else:
        print("Setup only. Training disabled; no new weights were saved.")
    return trainer
