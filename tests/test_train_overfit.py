"""Colab-marked overfitting test: trains for 50 steps on the 5 synthetic
fixtures and asserts loss decreases by at least 50%. Catches gross architectural
bugs (wrong loss reduction, broken backward graph, dead gradients, etc.).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


@pytest.fixture
def train_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_overfit_on_synthetic_fixtures(
    train_mod,
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
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

    import torch
    import prepare

    tok = prepare.get_tokenizer(project_root / "data" / "processed" / "tokenizer.json")
    model = train_mod.PharmaVLM(
        encoder_model="google/siglip-base-patch16-224",
        vocab_size=tok.get_vocab_size(),
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ffn_ratio=4,
        dropout=0.0,
        tied_embeddings=True,
        tokenizer=tok,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dl = prepare.get_dataloader(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=3,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-3)
    train_iter = train_mod._infinite(dl)
    losses: list[float] = []
    for step in range(50):
        model.train(True)
        batch = next(train_iter)
        images = batch["image"].to(device).float()
        in_ids, lab_ids = train_mod.encode_targets(tok, batch["target_text"], 256)
        in_ids = in_ids.to(device)
        lab_ids = lab_ids.to(device)
        memory = model._encode_images(images)
        logits = model.decoder(in_ids, memory)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            lab_ids.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    print(f"[overfit_test] initial_loss={initial:.4f} final_loss={final:.4f}")
    assert final < 0.5 * initial, (
        f"Expected loss to drop >50% in 50 steps on 5 synthetic examples, "
        f"got {initial:.4f} -> {final:.4f}"
    )
