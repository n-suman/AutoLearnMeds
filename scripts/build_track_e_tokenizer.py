"""Train a BPE tokenizer on the union of gold + pseudo XML targets.

Output: a tokenizers JSON file usable by prepare.get_tokenizer().

Usage:
    python scripts/build_track_e_tokenizer.py \
        --train-jsonl data/processed/train.jsonl \
        --pseudo-jsonl data/pseudo_labels/round_001.jsonl \
        --out experiments/tokenizers/track_e_bpe.json \
        --vocab-size 8192
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare  # noqa: E402


def _read_xml_targets(path: Path) -> list[str]:
    targets: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            targets.append(row.get("xml_label", ""))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--pseudo-jsonl", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=prepare.TOKENIZER_VOCAB_SIZE)
    args = ap.parse_args()

    corpus = _read_xml_targets(Path(args.train_jsonl))
    if args.pseudo_jsonl:
        corpus.extend(_read_xml_targets(Path(args.pseudo_jsonl)))

    print(f"Training BPE on {len(corpus)} XML targets (vocab={args.vocab_size})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # prepare.train_tokenizer signature: (corpus, out_path, vocab_size)
    prepare.train_tokenizer(corpus=corpus, out_path=out, vocab_size=args.vocab_size)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
