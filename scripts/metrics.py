"""Compatibility imports. Metric implementations live in script.py.

LLM-assisted consolidation, 2026-11-09. Existing notebooks/imports remain valid.
"""

from project_paths import PROJECT_ROOT
from script import (binary_confusion_counts, metrics_from_counts, compute_binary_metrics,
                    metrics_from_confusion, evaluate_segmentation)

__all__ = ["binary_confusion_counts", "metrics_from_counts", "compute_binary_metrics",
           "metrics_from_confusion", "evaluate_segmentation"]
