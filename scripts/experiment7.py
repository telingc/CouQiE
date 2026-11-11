"""Experiment 7: matched 10-epoch normal/frozen-statistics BN comparison.

LLM-assisted, 2026-11-09. Each policy starts from the same binary initialization.
"""

from project_paths import PROJECT_ROOT
from pathlib import Path
import pandas as pd
from IPython.display import display
from train_multiclass import experiment_folder,new_run,note,write_json,train_run,prepare_runtime


def run_experiment7(batch_result,*,root=None,epochs=10,run_dir=None):
    root=Path(root or PROJECT_ROOT).resolve();prepare_runtime()
    folder=experiment_folder(root,7);run=Path(run_dir) if run_dir else new_run(folder)
    batch=batch_result["selected"]
    note(folder, "Run status updated. See structured run metadata.")
    results=[]
    for policy in ["normal","frozen_stats"]:
        print(f"\nEXPERIMENT 7 | {policy} | {epochs} epochs",flush=True)
        try:
            result=train_run(root,run/policy,epochs=epochs,bn_policy=policy,
                             batch_size=batch["batch_size"],channels_last=batch["channels_last"])
        except Exception as error:
            note(folder, "Run status updated. See structured run metadata.")
            raise
        results.append(result)
        note(folder, "Run status updated. See structured run metadata.")
        write_json(run/"partial_results.json",results)
    selected=max(results,key=lambda r:r["metrics"]["foreground_miou"])
    summary=dict(experiment=7,folder=str(run),results=results,selected=selected)
    write_json(run/"result.json",summary);write_json(folder/"latest.json",summary)
    display(pd.DataFrame([dict(policy=r["config"]["bn_policy"],best_epoch=r["best_epoch"],
        foreground_miou=r["metrics"]["foreground_miou"],**dict(zip(["background","penguin","bird","seal","whale"],r["metrics"]["iou_per_class"]))) for r in results]).round(4))
    note(folder, "Run status updated. See structured run metadata.")
    return summary
