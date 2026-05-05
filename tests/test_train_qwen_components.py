"""Lightweight (non-Colab) unit tests for train_qwen.py pure-logic helpers."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def train_qwen_mod():
    """Import train_qwen.py from the project root, fresh per test.
    Lazy ml imports inside train_qwen.py mean this works in the torch-less Mac venv.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train_qwen" in sys.modules:
        importlib.reload(sys.modules["train_qwen"])
    import train_qwen
    return train_qwen


def test_qwen_config_loads_baseline_yaml(train_qwen_mod, project_root: Path) -> None:
    cfg = train_qwen_mod.QwenConfig.from_yaml(
        project_root / "experiments" / "configs" / "qwen_baseline.yaml"
    )
    assert cfg.track == "B"
    assert cfg.model_name == "Qwen/Qwen2-VL-2B-Instruct"
    assert cfg.lora_rank == 8
    assert cfg.adam_betas == (0.9, 0.999)
    assert cfg.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_qwen_module_imports_without_torch() -> None:
    """train_qwen.py must import in a torch-less venv (lazy ml imports).
    Belt-and-suspenders: confirm torch and transformers aren't in the module's
    top-level namespace.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train_qwen" in sys.modules:
        del sys.modules["train_qwen"]
    import train_qwen
    assert not hasattr(train_qwen, "torch")
    assert not hasattr(train_qwen, "transformers")
