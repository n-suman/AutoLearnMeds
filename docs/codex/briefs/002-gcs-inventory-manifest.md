# Brief 002: GCS inventory — offline manifest of all experiment artifacts

**Owner:** Codex
**Created:** 2026-05-13 by Claude
**Estimated effort:** S (under an hour)
**Estimated cost:** ~$0 (GCS list calls are free; downloads not needed)

## Context (self-contained)

The `AutoLearnMeds` project records experiments in two places:

1. **In git:** `experiments/ledger.jsonl` (one row per run, with metrics + git_sha + config), plus 6 local `experiments/runs/*/` directories.
2. **In GCS:** `gs://auto_learn_meds/experiments/` mirrors checkpoints, stdout.log, and predictions for every run — including the ~5 runs whose local directories don't exist in git (Track C-DAPT / C-text-aware / C-TAPT / Track D, twice).

The paper at `paper/main.md` depends on all 11 ledger entries staying reproducible. Today we can rebuild the paper's numbers from the ledger alone, but if a GCS object is missing, silently or otherwise, we won't notice until we need it. **Goal: produce a versioned offline manifest of GCS state so we can detect any future loss.**

The user (Suman) has been burned once before by a `sync_to_gcs.sh -d` flag that silently destroyed Track A's checkpoint (fixed at commit `75e9ad2`). Defensive inventory is the cheap insurance.

## Prerequisites Codex needs

Before starting, verify these are available. If any are missing, ask the user before proceeding:

1. **`gsutil` installed** — run `gsutil version` to confirm. If missing, install via `gcloud components install gsutil` or the standalone gsutil tarball.
2. **`gcloud` authenticated** to an account with read access to `gs://auto_learn_meds/`. Run `gcloud auth list` to confirm. The user's account is `suman@zanier.one` per memory; if not authenticated, ask the user to run `gcloud auth login` themselves (don't trigger an interactive browser flow from inside Codex — write a status note instead).
3. **Python 3.11+** for JSON manipulation. Run `python3 --version`.
4. **Local checkout of `AutoLearnMeds` repo** on branch `phase-0-plumbing` (where Claude is also working). Run `git branch --show-current && git status --short`. If on a different branch, switch and pull; if working tree is dirty in a way that conflicts, stop and report in the status file.
5. **No write access needed to GCS** — this is a read-only inventory.

## Goal

One sentence: produce `experiments/gcs_manifest.json` and `experiments/gcs_manifest.md` that describe the complete state of `gs://auto_learn_meds/experiments/` and cross-check it against `experiments/ledger.jsonl`.

## Step-by-step actions

### Step 1 — Recursive listing of the bucket

```bash
gsutil ls -l -r 'gs://auto_learn_meds/experiments/**' > /tmp/gcs_raw_listing.txt
wc -l /tmp/gcs_raw_listing.txt   # sanity check: should be hundreds to low thousands of lines
```

If the bucket has other top-level prefixes worth recording (`raw/`, `checkpoints/`, `pseudo_labels/`), also list those:

```bash
gsutil ls -l 'gs://auto_learn_meds/' > /tmp/gcs_top_level.txt
gsutil ls -l -r 'gs://auto_learn_meds/raw/' | head -200 > /tmp/gcs_raw_subset.txt
```

(`raw/` is multi-GB of image files; we don't want every JPG in the manifest, just the top-level prefixes and `*.jsonl` manifests.)

### Step 2 — Parse the listing into structured rows

For each object record one row:

```json
{
  "gs_path": "gs://auto_learn_meds/experiments/runs/baseline-seed44/best.pt",
  "size_bytes": 312449024,
  "last_modified": "2026-04-29T13:14:22Z",
  "md5_hash": "<from gsutil if cheap; skip if it requires extra fetch>"
}
```

Group rows by run-id (the path component after `experiments/runs/`). Each run gets a struct like:

```json
{
  "run_id": "baseline-seed44",
  "kind": "track_a",
  "object_count": 7,
  "total_size_bytes": 524288000,
  "objects": [...],
  "has_best_pt": true,
  "has_stdout_log": true,
  "has_metrics_json": true,
  "has_config_yaml": true,
  "has_predictions_jsonl": true
}
```

Use Python or `jq` — your call. Write a small reusable script at `scripts/codex/build_gcs_manifest.py` so future inventories are one command.

### Step 3 — Cross-check against the ledger

Load `experiments/ledger.jsonl` and compute three sets:

- **`ledger_run_ids`** — every `run_id` in the ledger (11 entries).
- **`gcs_run_ids`** — every distinct directory under `gs://auto_learn_meds/experiments/runs/`.
- **`local_run_ids`** — every directory under local `experiments/runs/`.

Then surface:

- `ledger_missing_from_gcs`: any ledger entry whose run-id has no GCS objects. **CRITICAL** if non-empty — paper artifacts that aren't backed up.
- `gcs_orphans`: any GCS run-id not in the ledger. These are likely failed runs that got auto-finalized or are debug artifacts — flag for cleanup decision (but don't delete).
- `ledger_missing_locally`: ledger entries with no local `experiments/runs/<id>/` directory. **EXPECTED** for ~5 entries (Track C × 3, Track D × 2). Just confirm via the table.

### Step 4 — Inventory the MAE pretraining + Track D YOLO weights

These live in separate prefixes; check each:

- `gs://auto_learn_meds/experiments/pretraining/` — should contain `mae-pretrain-seed44/` with the MAE encoder checkpoint.
- `gs://auto_learn_meds/checkpoints/` (if it exists) — sometimes YOLO/SAHI weights and intermediate models live here.

Record file presence + size for each.

### Step 5 — Inventory the `raw/` prefix (top-level only)

We don't want to list every JPG. Just record:

- Size of `gs://auto_learn_meds/raw/raw_images/` (one `gsutil du -s` call).
- File-count via `gsutil ls gs://auto_learn_meds/raw/raw_images/ | wc -l`.
- Existence of `train.jsonl`, `val.jsonl`, `test.jsonl` and their sizes / last-modified.
- Existence of `gold_standard.jsonl` if any.

### Step 6 — Write the manifest files

**Machine-readable:** `experiments/gcs_manifest.json` with this top-level structure:

```json
{
  "generated_at": "2026-05-13T20:00:00Z",
  "generated_by": "codex (brief 002)",
  "bucket": "gs://auto_learn_meds",
  "total_objects": ...,
  "total_size_bytes": ...,
  "runs": [ {<per-run struct from step 2>}, ... ],
  "pretraining": [...],
  "raw_summary": {...},
  "cross_check": {
    "ledger_missing_from_gcs": [],
    "gcs_orphans": [],
    "ledger_missing_locally": [...]
  }
}
```

**Human-readable:** `experiments/gcs_manifest.md` — a markdown table of run-ids × (in_ledger, in_gcs, in_local, has_best_pt, total_size_gb), plus prose blocks for cross-check findings.

### Step 7 — Commit

```bash
git add scripts/codex/build_gcs_manifest.py experiments/gcs_manifest.json experiments/gcs_manifest.md
git commit -m "chore(codex): GCS inventory manifest (brief 002)"
```

(Don't push without checking with the user first — `phase-0-plumbing` is the active branch and the user pushes manually.)

## Acceptance criteria

1. ✅ `experiments/gcs_manifest.json` exists and parses as JSON.
2. ✅ `experiments/gcs_manifest.md` is human-readable and lists every ledger run-id with its status.
3. ✅ Any `ledger_missing_from_gcs` entries are surfaced loudly in `status/002-...md` — this is the alarm condition Claude is looking for.
4. ✅ `scripts/codex/build_gcs_manifest.py` is reusable: re-running it produces an updated manifest without manual intervention.
5. ✅ Status file `status/002-gcs-inventory-manifest.md` contains: total objects, total size, count of GCS-only runs, count of any alarms, and a one-paragraph summary.

## Anti-goals — explicitly DO NOT

- ❌ Do not download checkpoint files (`*.pt`, `*.bin`) — they're large and we only need metadata. Use `gsutil ls -l` not `gsutil cp`.
- ❌ Do not delete any GCS objects, even apparent orphans. Cleanup is a separate decision by the user.
- ❌ Do not modify `experiments/ledger.jsonl` — that's Claude's append-only history.
- ❌ Do not commit the raw `gsutil ls` output (the `.txt` files in `/tmp`). Only the parsed JSON + markdown go in git.
- ❌ Do not enumerate `gs://auto_learn_meds/raw/raw_images/` to the object level — the bucket has ~3,000 JPGs and listing them per-object bloats the manifest. Just the top-level summary.

## Where to write

| File | Purpose |
|---|---|
| `scripts/codex/build_gcs_manifest.py` | Reusable inventory script |
| `experiments/gcs_manifest.json` | Machine-readable manifest |
| `experiments/gcs_manifest.md` | Human-readable manifest |
| `status/002-gcs-inventory-manifest.md` | Status updates + final alarm summary |

## If alarms fire

If `ledger_missing_from_gcs` is non-empty after step 3, **stop**. Don't try to fix it yourself. Write a detailed alarm into the status file:

- Which ledger entries are missing GCS backup
- When were they recorded (timestamp from ledger row)
- What the last-known local artifacts are (cross-check `experiments/runs/<id>/` and `experiments/per_field_<id>.json`)

The user will decide whether to re-upload from a local copy or accept the loss.

## Coordination with Brief 001

If Brief 001 (pseudo-labeling) is in flight: Brief 002 can run concurrently — it doesn't touch any of the same files. Both write to git, but on different paths. Sequence the two `git commit`s.
