"""Experiment 6: resource/throughput measurement, not an accuracy training run.

LLM-assisted, 2026-11-09. Actual results, including failures, are recorded.
"""

from project_paths import PROJECT_ROOT
from pathlib import Path
from train_multiclass import prepare_runtime,benchmark,experiment_folder,new_run,note,write_json


def run_experiment6(root=None):
    root=Path(root or PROJECT_ROOT).resolve()
    prepare_runtime()
    folder=experiment_folder(root,6);run=new_run(folder)
    note(folder, "Run status updated. See structured run metadata.")
    result=benchmark(root)
    result.update(folder=str(run),experiment=6)
    write_json(run/"result.json",result);write_json(folder/"latest.json",result)
    best=result["selected"]
    note(folder, "Run status updated. See structured run metadata.")
    return result
