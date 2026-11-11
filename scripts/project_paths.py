"""Canonical workspace locations; independent of the caller's current directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
REPORT_ROOT = PROJECT_ROOT / "reports"
