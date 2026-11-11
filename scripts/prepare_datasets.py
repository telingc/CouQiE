"""Compatibility CLI. Dataset preparation implementations live in script.py.

LLM-assisted consolidation, 2026-11-09. Importing this file never prepares data.
"""

from project_paths import PROJECT_ROOT
from script import (SEED, SIZE, save_sample, save_manifests, bird_mask, prepare_birds,
                    largest_filled_region, recover_seal_mask, prepare_seals,
                    prepare_all_datasets)

main = prepare_all_datasets

if __name__ == "__main__":
    main()
