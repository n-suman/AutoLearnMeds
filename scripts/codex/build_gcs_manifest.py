#!/usr/bin/env python3
"""Build an offline manifest for AutoLearnMeds GCS experiment artifacts.

Read-only by design: this script only invokes `gsutil ls` and `gsutil du`.
It writes a machine-readable JSON manifest and a Markdown summary that can be
committed with the repo for later drift/loss checks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


BUCKET = "gs://auto_learn_meds"


def run_gsutil(args: list[str], *, allow_failure: bool = False) -> str:
    """Run a gsutil command and return stdout."""
    proc = subprocess.run(
        ["gsutil", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"gsutil {' '.join(args)} failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def parse_ls_l(output: str) -> list[dict[str, Any]]:
    """Parse `gsutil ls -l` / `gsutil ls -l -r` object rows."""
    objects: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("TOTAL:") or line.endswith(":"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            size = int(parts[0])
        except ValueError:
            continue
        last_modified = parts[1]
        gs_path = parts[2]
        if not gs_path.startswith("gs://"):
            continue
        objects.append(
            {
                "gs_path": gs_path,
                "size_bytes": size,
                "last_modified": last_modified,
            }
        )
    return objects


def parse_du_s(output: str) -> dict[str, Any]:
    """Parse `gsutil du -s` output."""
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            return {"size_bytes": int(parts[0]), "gs_path": parts[1]}
    return {"size_bytes": 0, "gs_path": ""}


def read_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def local_run_ids(runs_dir: Path) -> set[str]:
    if not runs_dir.exists():
        return set()
    return {p.name for p in runs_dir.iterdir() if p.is_dir()}


def infer_kind(run_id: str) -> str:
    if run_id.startswith("qwen"):
        return "track_b"
    if run_id.startswith("track_d"):
        return "track_d"
    if "mae" in run_id:
        return "track_c"
    if run_id.startswith("baseline") or run_id.startswith("randaug"):
        return "track_a"
    if run_id.startswith("yolo"):
        return "track_d_detection"
    return "unknown"


def group_runs(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    marker = "/experiments/runs/"
    for obj in objects:
        path = obj["gs_path"]
        if marker not in path:
            continue
        tail = path.split(marker, 1)[1]
        if "/" not in tail:
            continue
        run_id = tail.split("/", 1)[0]
        if run_id:
            grouped.setdefault(run_id, []).append(obj)

    runs: list[dict[str, Any]] = []
    for run_id, rows in sorted(grouped.items()):
        paths = [r["gs_path"] for r in rows]
        total_size = sum(int(r["size_bytes"]) for r in rows)
        runs.append(
            {
                "run_id": run_id,
                "kind": infer_kind(run_id),
                "object_count": len(rows),
                "total_size_bytes": total_size,
                "objects": sorted(rows, key=lambda r: r["gs_path"]),
                "has_best_pt": any(p.endswith("/best.pt") or "/best.pt" in p for p in paths),
                "has_stdout_log": any(p.endswith("/stdout.log") for p in paths),
                "has_metrics_json": any(p.endswith("/metrics.json") for p in paths),
                "has_config_yaml": any(p.endswith("/config.yaml") for p in paths),
                "has_predictions_jsonl": any(
                    p.endswith("/predictions.jsonl")
                    or p.endswith("/predictions.json")
                    or "prediction" in Path(p).name
                    for p in paths
                ),
            }
        )
    return runs


def summarize_prefix(prefix: str) -> list[dict[str, Any]]:
    listing = run_gsutil(["ls", "-l", "-r", prefix], allow_failure=True)
    return parse_ls_l(listing)


def group_by_prefix(objects: list[dict[str, Any]], marker: str, depth: int = 1) -> list[dict[str, Any]]:
    """Group objects by path components after `marker`."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        path = obj["gs_path"]
        if marker not in path:
            continue
        tail = path.split(marker, 1)[1]
        if not tail or "/" not in tail:
            continue
        parts = tail.split("/")
        key = "/".join(parts[:depth])
        grouped.setdefault(key, []).append(obj)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        rows.append(
            {
                "prefix": key,
                "object_count": len(group),
                "total_size_bytes": sum(o["size_bytes"] for o in group),
                "has_best_pt": any(o["gs_path"].endswith("/best.pt") for o in group),
                "has_model_safetensors": any(o["gs_path"].endswith("/model.safetensors") for o in group),
                "has_trainer_state": any(o["gs_path"].endswith("/trainer_state.pt") for o in group),
                "objects": sorted(group, key=lambda r: r["gs_path"]),
            }
        )
    return rows


def build_raw_summary() -> dict[str, Any]:
    raw_top_level = run_gsutil(["ls", "-l", f"{BUCKET}/raw/"], allow_failure=True)
    raw_images_listing = run_gsutil(["ls", f"{BUCKET}/raw/raw_images/"], allow_failure=True)
    raw_images_du = run_gsutil(["du", "-s", f"{BUCKET}/raw/raw_images/"], allow_failure=True)

    raw_image_paths = [
        line.strip()
        for line in raw_images_listing.splitlines()
        if line.strip().startswith("gs://")
    ]
    image_like = [
        p
        for p in raw_image_paths
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))
    ]

    manifest_candidates = [
        f"{BUCKET}/raw/train.jsonl",
        f"{BUCKET}/raw/val.jsonl",
        f"{BUCKET}/raw/test.jsonl",
        f"{BUCKET}/raw/gold_standard.jsonl",
        f"{BUCKET}/raw/golden_set/gold_standard.jsonl",
        f"{BUCKET}/raw/golden_set/splits.json",
    ]
    manifests: list[dict[str, Any]] = []
    for candidate in manifest_candidates:
        objects = parse_ls_l(run_gsutil(["ls", "-l", candidate], allow_failure=True))
        if objects:
            manifests.extend(objects)

    return {
        "top_level_listing": raw_top_level.splitlines(),
        "raw_images": {
            "object_count_including_non_images": len(raw_image_paths),
            "image_file_count": len(image_like),
            "du": parse_du_s(raw_images_du),
        },
        "manifests": sorted(manifests, key=lambda r: r["gs_path"]),
    }


def build_manifest(repo_root: Path) -> dict[str, Any]:
    experiments_objects = summarize_prefix(f"{BUCKET}/experiments/")
    pretraining = [
        obj for obj in experiments_objects if "/experiments/pretraining/" in obj["gs_path"]
    ]
    checkpoints = summarize_prefix(f"{BUCKET}/checkpoints/")
    checkpoint_groups = group_by_prefix(checkpoints, "/checkpoints/", depth=2)
    runs = group_runs(experiments_objects)

    ledger_rows = read_ledger(repo_root / "experiments" / "ledger.jsonl")
    ledger_run_ids = {str(row.get("run_id", "")) for row in ledger_rows if row.get("run_id")}
    gcs_run_ids = {row["run_id"] for row in runs}
    local_ids = local_run_ids(repo_root / "experiments" / "runs")

    total_objects = len(experiments_objects) + len(checkpoints)
    total_size = sum(o["size_bytes"] for o in experiments_objects) + sum(
        o["size_bytes"] for o in checkpoints
    )

    return {
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "generated_by": "codex (brief 002)",
        "bucket": BUCKET,
        "total_objects": total_objects,
        "total_size_bytes": total_size,
        "experiments_object_count": len(experiments_objects),
        "experiments_total_size_bytes": sum(o["size_bytes"] for o in experiments_objects),
        "runs": runs,
        "pretraining": sorted(pretraining, key=lambda r: r["gs_path"]),
        "checkpoints": sorted(checkpoints, key=lambda r: r["gs_path"]),
        "checkpoint_groups": checkpoint_groups,
        "raw_summary": build_raw_summary(),
        "cross_check": {
            "ledger_entry_count": len(ledger_rows),
            "ledger_run_ids": sorted(ledger_run_ids),
            "gcs_run_ids": sorted(gcs_run_ids),
            "local_run_ids": sorted(local_ids),
            "ledger_missing_from_gcs": sorted(ledger_run_ids - gcs_run_ids),
            "gcs_orphans": sorted(gcs_run_ids - ledger_run_ids),
            "ledger_missing_locally": sorted(ledger_run_ids - local_ids),
        },
    }


def fmt_size(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size_bytes} B"


def write_markdown(manifest: dict[str, Any], path: Path) -> None:
    cc = manifest["cross_check"]
    runs_by_id = {run["run_id"]: run for run in manifest["runs"]}
    all_run_ids = sorted(set(cc["ledger_run_ids"]) | set(cc["gcs_run_ids"]) | set(cc["local_run_ids"]))

    lines = [
        "# GCS Manifest",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        f"Bucket: `{manifest['bucket']}`",
        "",
        f"Total inventoried objects: **{manifest['total_objects']}**",
        f"Total inventoried size: **{fmt_size(manifest['total_size_bytes'])}**",
        "",
        "## Cross-Check Summary",
        "",
        f"- Ledger run IDs: **{len(cc['ledger_run_ids'])}**",
        f"- Ledger entries: **{cc['ledger_entry_count']}**",
        f"- GCS run IDs: **{len(cc['gcs_run_ids'])}**",
        f"- Local run IDs: **{len(cc['local_run_ids'])}**",
        f"- Ledger missing from GCS: **{len(cc['ledger_missing_from_gcs'])}**",
        f"- GCS orphan runs: **{len(cc['gcs_orphans'])}**",
        f"- Ledger missing locally: **{len(cc['ledger_missing_locally'])}**",
        "",
    ]

    if cc["ledger_missing_from_gcs"]:
        lines.extend(
            [
                "## Alarm: Ledger Entries Missing From GCS",
                "",
                "These ledger runs have no corresponding objects under `experiments/runs/` in GCS:",
                "",
                *[f"- `{run_id}`" for run_id in cc["ledger_missing_from_gcs"]],
                "",
            ]
        )
    else:
        lines.extend(["## Alarms", "", "No ledger run IDs are missing from GCS.", ""])

    lines.extend(
        [
            "## Runs",
            "",
            "| run_id | in_ledger | in_gcs | in_local | kind | objects | has_best_pt | has_stdout_log | has_metrics_json | has_config_yaml | has_predictions | total_size |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id in all_run_ids:
        run = runs_by_id.get(run_id, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run_id}`",
                    "yes" if run_id in cc["ledger_run_ids"] else "no",
                    "yes" if run_id in cc["gcs_run_ids"] else "no",
                    "yes" if run_id in cc["local_run_ids"] else "no",
                    str(run.get("kind", "")),
                    str(run.get("object_count", 0)),
                    "yes" if run.get("has_best_pt") else "no",
                    "yes" if run.get("has_stdout_log") else "no",
                    "yes" if run.get("has_metrics_json") else "no",
                    "yes" if run.get("has_config_yaml") else "no",
                    "yes" if run.get("has_predictions_jsonl") else "no",
                    fmt_size(int(run.get("total_size_bytes", 0))),
                ]
            )
            + " |"
        )

    raw = manifest["raw_summary"]
    lines.extend(
        [
            "",
            "## Raw Data Summary",
            "",
            f"- Raw image objects, including non-images: **{raw['raw_images']['object_count_including_non_images']}**",
            f"- Raw image files: **{raw['raw_images']['image_file_count']}**",
            f"- Raw image prefix size: **{fmt_size(raw['raw_images']['du']['size_bytes'])}**",
            "",
            "### Raw Manifests",
            "",
            "| path | size | last_modified |",
            "|---|---:|---|",
        ]
    )
    for obj in raw["manifests"]:
        lines.append(
            f"| `{obj['gs_path']}` | {fmt_size(obj['size_bytes'])} | `{obj['last_modified']}` |"
        )

    pretraining = manifest["pretraining"]
    checkpoints = manifest["checkpoints"]
    checkpoint_groups = manifest["checkpoint_groups"]
    lines.extend(
        [
            "",
            "## Other Prefixes",
            "",
            f"- Pretraining objects under `experiments/pretraining/`: **{len(pretraining)}**, {fmt_size(sum(o['size_bytes'] for o in pretraining))}",
            f"- Checkpoint objects under `checkpoints/`: **{len(checkpoints)}**, {fmt_size(sum(o['size_bytes'] for o in checkpoints))}",
            "",
            "### Checkpoint Groups",
            "",
            "| prefix | objects | has_best_pt | has_model_safetensors | has_trainer_state | total_size |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in checkpoint_groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{group['prefix']}`",
                    str(group["object_count"]),
                    "yes" if group["has_best_pt"] else "no",
                    "yes" if group["has_model_safetensors"] else "no",
                    "yes" if group["has_trainer_state"] else "no",
                    fmt_size(group["total_size_bytes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This manifest intentionally records metadata only; checkpoint/model files were not downloaded.",
            "- Apparent GCS orphans are listed for review only. Nothing was deleted.",
        ]
    )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("experiments/gcs_manifest.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("experiments/gcs_manifest.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = build_manifest(repo_root)

    json_out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
    md_out = args.md_out if args.md_out.is_absolute() else repo_root / args.md_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_markdown(manifest, md_out)

    cc = manifest["cross_check"]
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    print(f"total_objects={manifest['total_objects']}")
    print(f"total_size_bytes={manifest['total_size_bytes']}")
    print(f"ledger_missing_from_gcs={len(cc['ledger_missing_from_gcs'])}")
    print(f"gcs_orphans={len(cc['gcs_orphans'])}")


if __name__ == "__main__":
    main()
