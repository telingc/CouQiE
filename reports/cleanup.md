# Historical layout migration

This table records the earlier layout migration, not the current launcher size or final experiment count. For current state see [project closeout](CLOSEOUT.md).

Code is under scripts/, data under ignored data/, and historical large outputs under ignored artifacts/. The original source notebook and Git history were backed up before migration. No model training was performed.

| Check | Result |
|---|---:|
| Launcher size | 2702 bytes |
| Embedded notebook images | 0 |
| Python modules at root | 0 |
| Wildlife training files / source groups | 7858 / 2627 |
| Wildlife retained target training records | 650 (previously 1938) |
| Wildlife cross-split source-prefix overlap | 0 |
| Inference-only weight file | 270.9 MB |
| Oversized blobs reachable from main | 0 |

The single unpushed commit was undone with a mixed reset. Working files are preserved and have not been recommitted or pushed. Historical model optimizer states remain archived; physical disk usage was not reduced by deleting history.

Fresh local gallery: artifacts/evaluation/20261111_072147/gallery.html.
Current counts: datasets.csv. Raw source-group audit: wildlife.csv. The before-cleanup Git bundle is artifacts/archive/before_cleanup.bundle.


## Final closeout organization

The active project retains formal experiments 1-14. The rejected unnumbered 5x5
preview source and outputs remain under local artifacts/previews/. Four one-time
migration/audit scripts were moved from scripts/ to the closeout archive. Their
contents were verified unchanged, and a complete before-closeout source/report
backup was made. Historical sources, checkpoints, input data and frozen masks
were not removed.

The current [closeout archive pointer](../artifacts/maintenance/closeout/latest.json)
identifies the archive manifest, backup and read-only verification. The active
replacement is [check_project.py](../scripts/check_project.py), which never trains,
infers, downloads, rewrites datasets or exports models. Report generation no
longer edits the notebook, and experiment-13 curves regenerate from saved logs.
