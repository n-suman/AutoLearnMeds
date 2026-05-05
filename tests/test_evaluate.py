"""Tests for prepare.py public eval API: evaluate() + evaluate_test() firewall."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


class _StubModelAlwaysEmpty:
    """Returns an empty (BOS+EOS) string for every input — simulates an
    untrained model."""

    def predict_text(self, batch_images, max_new_tokens: int = 256) -> list[str]:
        return [""] * len(batch_images)


def test_evaluate_runs_on_stub_returns_zero(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    metrics = prepare_mod.evaluate(
        model=_StubModelAlwaysEmpty(),
        jsonl_path=tiny_processed_dir / "val.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=4,
    )
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] == 0.0  # stub predicts nothing
    assert "per_field_f1" in metrics
    assert metrics["n_examples"] == 1


def test_evaluate_runs_on_stub_with_partial_predictions(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    """A stub model that always returns the correct brand_name text gets a
    boost on that field's F1."""
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )

    class _StubBrandOnly:
        def predict_text(self, batch_images, max_new_tokens=256):
            xml = (
                prepare_mod.BOS_TOKEN
                + "<drug_name>Synthcol</drug_name>"
                + prepare_mod.EOS_TOKEN
            )
            return [xml] * len(batch_images)

    metrics = prepare_mod.evaluate(
        model=_StubBrandOnly(),
        jsonl_path=tiny_processed_dir / "val.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=4,
    )
    # val has 1 record (IMG_synth_004) with drug_name="Synthcol".
    # Stub's prediction matches → drug_name F1 is 1.0.
    assert metrics["per_field_f1"]["drug_name"] == pytest.approx(1.0)


def test_evaluate_test_is_gated(prepare_mod, tiny_processed_dir: Path) -> None:
    """evaluate_test must require an explicit confirmation flag."""
    with pytest.raises(RuntimeError, match="explicit_consent"):
        prepare_mod.evaluate_test(
            model=_StubModelAlwaysEmpty(),
            jsonl_path=tiny_processed_dir / "test.jsonl",
            images_root=tiny_processed_dir,
            path_strip_prefix="raw/",
        )


def test_evaluate_test_logs_audit_entry(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
    tmp_path: Path,
) -> None:
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    audit = tmp_path / "evaluate_test_audit.log"
    prepare_mod.evaluate_test(
        model=_StubModelAlwaysEmpty(),
        jsonl_path=tiny_processed_dir / "test.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        explicit_consent="i_understand_this_consumes_the_test_set",
        audit_log=audit,
    )
    assert audit.is_file()
    assert "evaluate_test invoked" in audit.read_text()
