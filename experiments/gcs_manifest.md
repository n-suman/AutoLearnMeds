# GCS Manifest

Generated: `2026-05-12T19:47:11+00:00`

Bucket: `gs://auto_learn_meds`

Total inventoried objects: **160**
Total inventoried size: **21.01 GiB**

## Cross-Check Summary

- Ledger run IDs: **10**
- Ledger entries: **11**
- GCS run IDs: **10**
- Local run IDs: **6**
- Ledger missing from GCS: **0**
- GCS orphan runs: **0**
- Ledger missing locally: **4**

## Alarms

No ledger run IDs are missing from GCS.

## Runs

| run_id | in_ledger | in_gcs | in_local | kind | objects | has_best_pt | has_stdout_log | has_metrics_json | has_config_yaml | has_predictions | total_size |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-seed42` | yes | yes | yes | track_a | 1 | no | no | no | no | no | 401 B |
| `baseline-seed43` | yes | yes | yes | track_a | 1 | no | no | no | no | no | 401 B |
| `baseline-seed44` | yes | yes | yes | track_a | 1 | no | no | no | no | no | 401 B |
| `baseline-seed44-rerun` | yes | yes | yes | track_a | 5 | no | yes | yes | yes | no | 37.03 KiB |
| `baseline_mae_init-seed44` | yes | yes | no | track_c | 4 | no | yes | yes | yes | no | 32.35 KiB |
| `baseline_mae_tapt_init-seed44` | yes | yes | no | track_c | 4 | no | yes | yes | yes | no | 32.50 KiB |
| `baseline_mae_text_aware_init-seed44` | yes | yes | no | track_c | 4 | no | yes | yes | yes | no | 32.51 KiB |
| `qwen-baseline-seed42` | yes | yes | yes | track_b | 5 | no | yes | yes | yes | no | 33.14 KiB |
| `randaug-m5-seed42` | yes | yes | yes | track_a | 5 | no | yes | yes | yes | no | 31.51 KiB |
| `track_d-seed44` | yes | yes | no | track_d | 4 | no | yes | yes | yes | no | 14.06 KiB |

## Raw Data Summary

- Raw image objects, including non-images: **3061**
- Raw image files: **3060**
- Raw image prefix size: **7.60 GiB**

### Raw Manifests

| path | size | last_modified |
|---|---:|---|
| `gs://auto_learn_meds/raw/golden_set/gold_standard.jsonl` | 1.07 MiB | `2026-05-04T21:17:01Z` |
| `gs://auto_learn_meds/raw/golden_set/splits.json` | 17.46 KiB | `2026-05-04T21:17:00Z` |

## Other Prefixes

- Pretraining objects under `experiments/pretraining/`: **13**, 122.54 KiB
- Checkpoint objects under `checkpoints/`: **78**, 21.01 GiB

### Checkpoint Groups

| prefix | objects | has_best_pt | has_model_safetensors | has_trainer_state | total_size |
|---|---:|---:|---:|---:|---:|
| `mae/run-seed44` | 10 | no | yes | no | 1.73 GiB |
| `mae/tapt-seed44` | 11 | no | yes | yes | 3.76 GiB |
| `mae/text-aware-seed44` | 43 | no | yes | yes | 15.12 GiB |
| `runs/baseline` | 2 | yes | no | no | 101.28 MiB |
| `runs/baseline_mae_init` | 2 | yes | no | no | 101.28 MiB |
| `runs/baseline_mae_tapt_init` | 2 | yes | no | no | 101.28 MiB |
| `runs/baseline_mae_text_aware_init` | 2 | yes | no | no | 101.28 MiB |
| `runs/qwen_baseline` | 3 | no | no | no | 8.35 MiB |

## Notes

- This manifest intentionally records metadata only; checkpoint/model files were not downloaded.
- Apparent GCS orphans are listed for review only. Nothing was deleted.
