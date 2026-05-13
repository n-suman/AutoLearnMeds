"""Tests for the stage scheduler."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def train_mod(project_root: Path):
    import importlib
    sys.path.insert(0, str(project_root))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_empty_schedule_returns_default_stage(train_mod) -> None:
    """No schedule = single 'baseline' stage with all data, lr_mult=1.0, for cfg.max_steps."""
    cfg = train_mod.Config(max_steps=1000, stage_schedule=[])
    sched = train_mod.StageScheduler(cfg)
    assert sched.stage_at_step(0).name == "baseline"
    assert sched.stage_at_step(999).name == "baseline"
    assert sched.stage_at_step(0).lr_mult == 1.0
    assert sched.stage_at_step(0).data == "all"


def test_three_stage_schedule_transitions(train_mod) -> None:
    """500 + 2000 + 500 = 3000 step schedule transitions cleanly at boundaries."""
    schedule = [
        {"name": "warmup", "steps": 500, "data": "pseudo", "lr_mult": 1.0},
        {"name": "joint", "steps": 2000, "data": "all", "lr_mult": 1.0},
        {"name": "finetune", "steps": 500, "data": "gold", "lr_mult": 0.1},
    ]
    cfg = train_mod.Config(max_steps=3000, stage_schedule=schedule)
    sched = train_mod.StageScheduler(cfg)

    assert sched.stage_at_step(0).name == "warmup"
    assert sched.stage_at_step(499).name == "warmup"
    assert sched.stage_at_step(500).name == "joint"
    assert sched.stage_at_step(2499).name == "joint"
    assert sched.stage_at_step(2500).name == "finetune"
    assert sched.stage_at_step(2999).name == "finetune"

    assert sched.stage_at_step(0).data == "pseudo"
    assert sched.stage_at_step(2500).lr_mult == pytest.approx(0.1)


def test_total_steps_from_schedule(train_mod) -> None:
    """total_steps() returns the sum of stage steps when schedule is set."""
    schedule = [
        {"name": "warmup", "steps": 500, "data": "pseudo", "lr_mult": 1.0},
        {"name": "joint", "steps": 2000, "data": "all", "lr_mult": 1.0},
        {"name": "finetune", "steps": 500, "data": "gold", "lr_mult": 0.1},
    ]
    cfg = train_mod.Config(stage_schedule=schedule)
    sched = train_mod.StageScheduler(cfg)
    assert sched.total_steps() == 3000


def test_total_steps_from_max_steps_when_no_schedule(train_mod) -> None:
    """No schedule = total_steps() reflects cfg.max_steps."""
    cfg = train_mod.Config(max_steps=1500, stage_schedule=[])
    sched = train_mod.StageScheduler(cfg)
    assert sched.total_steps() == 1500


def test_example_mask_gold_only_stage(train_mod) -> None:
    """Stage data='gold' should mask out pseudo examples in the batch."""
    rows = [{"image_path": "g.jpg", "is_pseudo": False}, {"image_path": "p.jpg", "is_pseudo": True}]
    sched = train_mod.StageScheduler(train_mod.Config(stage_schedule=[{"name": "s1", "steps": 100, "data": "gold", "lr_mult": 1.0}]))
    mask = sched.example_mask_at_step(0, rows)
    assert mask == [True, False]


def test_example_mask_pseudo_only_stage(train_mod) -> None:
    """Stage data='pseudo' should mask out gold examples."""
    rows = [{"image_path": "g.jpg", "is_pseudo": False}, {"image_path": "p.jpg", "is_pseudo": True}]
    sched = train_mod.StageScheduler(train_mod.Config(stage_schedule=[{"name": "s1", "steps": 100, "data": "pseudo", "lr_mult": 1.0}]))
    mask = sched.example_mask_at_step(0, rows)
    assert mask == [False, True]


def test_example_mask_all_stage(train_mod) -> None:
    """Stage data='all' (or default) keeps every example."""
    rows = [{"image_path": "g.jpg", "is_pseudo": False}, {"image_path": "p.jpg", "is_pseudo": True}]
    sched = train_mod.StageScheduler(train_mod.Config(stage_schedule=[]))
    mask = sched.example_mask_at_step(0, rows)
    assert mask == [True, True]
