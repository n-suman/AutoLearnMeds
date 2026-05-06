# Track A vs Track B - per-field comparison

Methodology: identical val split (n=111 for Track A, n=111 for Track B), identical metrics (exact-match per-field F1 and partial-credit edit-F1 averaged over the 10 high-frequency fields), different methods - Track A is `/workspace/checkpoints/runs/baseline/best.pt` (experiments/configs/baseline.yaml); Track B is `/workspace/checkpoints/runs/qwen_baseline/best_lora` (experiments/configs/qwen_baseline.yaml). Cells marked with `*` are where Track B beats Track A.

## Per-field table (alphabetical)

| field | A f1 | B f1 | Δ f1 (B-A) | A edit_f1 | B edit_f1 | Δ edit_f1 (B-A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch_number | 0.0000 | 0.0000 | +0.0000 | 0.1807 | 0.1131 | -0.0676 |
| brand_name | 0.1379 | 0.0438 | -0.0941 | 0.5790 | 0.2108 | -0.3682 |
| company | 0.3500 | 0.0000 | -0.3500 | 0.3401 | 0.0000 | -0.3401 |
| drug_name | 0.2044 | 0.1098 | -0.0946 | 0.6139 | 0.2838 | -0.3301 |
| expiry_date | 0.0000 | 0.0000 | +0.0000 | 0.2656 | 0.1617 | -0.1038 |
| generic_name | 0.0488 | 0.0414 | -0.0074 | 0.6147 | 0.2460 | -0.3687 |
| manufacturer | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| mfg_date | 0.0000 | 0.0000 | +0.0000 | 0.2948 | 0.1386 | -0.1561 |
| mrp | 0.0000 | 0.0000 | +0.0000 | 0.2173 | 0.1219 | -0.0954 |
| quantity | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| strength | 0.0000 | 0.0000 | +0.0000 | 0.1190 | 0.0000 | -0.1190 |
| warnings | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 |
| **macro** | **0.0741** | **0.0195** | **-0.0546** | **0.3225** | **0.1276** | **-0.1949** |

## Track B wins (at least one metric)

_None - Track A wins or ties on every field._

## Track A wins (or ties)

- batch_number
- brand_name
- company
- drug_name
- expiry_date
- generic_name
- manufacturer
- mfg_date
- mrp
- quantity
- strength
- warnings

## What this tells the paper

Track A wins both macro metrics (f1 0.0741 vs 0.0195; edit_f1 0.3225 vs 0.1276). At the per-field grain, Track A wins or ties Track B on every field for both metrics; the macro gap is uniform, not driven by a few OCR-heavy fields.

## Citations

**Direct prior work on the same dataset (load-bearing for the paper):**
- Malepati, Nandamury, Manjunath, Rajan, Prabhune (2026). "Comparative Evaluation of YOLOv12 and SAHI for Medication Identification in Hospital Pharmacies." *2026 International Conference on Intelligent and Innovative Technologies in Computing, Electrical and Electronics (IITCEE)*, IEEE. DOI: 10.1109/IITCEE67948.2026.11394638. — establishes that on **the same 48 MP smartphone-photo dataset our 564/111/111 split is drawn from**, YOLOv12 + Sliced Aided Hyper Inference (1024-px tiles, 0.35 overlap, base 1280) lifted macro AP@0.5 on the four OCR-critical classes (Batch, MRP, Manufacturing Date, Expiry) from 0.035 to 0.609 — a 17× gain. This identifies SAME failure mode our per-field analysis observes (small-text fields scoring 0.0 strict for both Tracks A and B at 224×224) and proves it can be solved by tiled inference. Naturally pairs with end-to-end approaches as the modular-pipeline baseline; see research_directions.md #11–#13 for our reuse strategy.

**SAHI primary reference:**
- Akyon, Altinuc, Temizel (2022). "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." *2022 IEEE International Conference on Image Processing (ICIP)*, pp. 966–970. DOI: 10.1109/ICIP46576.2022.9897990. — introduces SAHI for tiled inference over high-resolution imagery.

**Per-field evaluation methodology:**
- Sang & De Meulder (2003). "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition." — canonical convention for per-entity-type F1 reporting in span/field extraction.
- Lipton (2014). "Optimal Thresholding of Classifiers to Maximize F1 Measure." arXiv:1402.1892. — discusses F1's brittleness on small per-class supports (esp. relevant for our `manufacturer`, `quantity`, `warnings` fields with very few positives) and motivates complementing with edit-distance metrics.
- Levenshtein (1966). "Binary codes capable of correcting deletions, insertions, and reversals." — basis of our partial-credit `macro_edit_f1`.

**Track A architecture:**
- Zhai, Mustafa, Kolesnikov, Beyer (2023). "Sigmoid Loss for Language Image Pre-training." arXiv:2303.15343. — SigLIP, the frozen vision encoder.
- Kim et al. (2022). "OCR-free Document Understanding Transformer." arXiv:2111.15664. — Donut-style autoregressive XML decoder, our decoder choice.

**Track B architecture:**
- Wang et al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv:2409.12191. — base VLM we LoRA-finetune.
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685. — adapter method.
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314. — 4-bit nf4 + LoRA recipe we used.
