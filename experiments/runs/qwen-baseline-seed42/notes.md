# Run: qwen-baseline-seed42

**Phase:** baseline (Track B kickoff)
**Track:** B (Qwen2-VL-2B + LoRA)
**Parent:** none (first Track B run; head-to-head against Track A's baseline-seed44 floor 0.0804)

## Hypothesis

Fine-tuning Qwen2-VL-2B with LoRA r=8 (q/k/v/o_proj) on the same 564-image train split will outperform Track A's custom hybrid by **+0.20 to +0.40 macro_f1** and **+0.40 to +0.70 macro_edit_f1**, despite training only ~5M of ~2.2B parameters.

**Why expect a large jump:**

1. **Pretrained vision-language alignment.** Qwen2-VL was pretrained on hundreds of millions of image-text pairs including OCR-heavy data; reading product labels is in-distribution. Track A's encoder is frozen SigLIP (also strong) but the decoder is initialized from scratch and has to learn the field grammar from 564 images.

2. **In-context decoding.** Qwen2-VL generates from a chat-formatted prompt that includes the full system instruction listing the 12 fields. The model can "look up" which field it should be emitting; Track A's decoder has no such grounding signal.

3. **Lower trainable parameter count fights overfitting.** Track A overfits hard at 26.5M trainable params on 564 images. LoRA at r=8 trains ~5M params (5× fewer) on the same data — closer to a healthy ratio for small-data regimes.

4. **OCR-grade visual tokens.** Qwen2-VL uses Naive Dynamic Resolution and a vision encoder tuned for fine text. Expected to help especially on `batch_number` / `expiry_date` / `mrp` fields where Track A's 224×224 SigLIP loses small-text legibility.

**Why expect edit_f1 to jump even more than macro_f1:** Qwen's outputs will likely have minor formatting drift (extra spaces, dropped punctuation, partial dates like "Jan 26" vs "JAN-2026"). Strict macro_f1 punishes this; edit_f1 gives partial credit. The gap between the two metrics is itself an interesting paper finding — it tells us how much of Track A's apparent failure is *content* vs *format*.

## Risks / null-result conditions

- **5M params still overfit** — possible but unlikely at this scale. If train loss → 0 and val plateaus low, drop r to 4 or add LoRA dropout 0.1.
- **Output format drift dominates strict metric** — would manifest as edit_f1 ≫ macro_f1 (>0.30 gap). Diagnostic, not failure: paper-worthy in itself.
- **Qwen ignores system prompt.** 2B-class instruction-tuned models occasionally drift from system instructions on out-of-distribution tasks. If outputs are conversational ("Sure, here's the info..."), need to add few-shot examples or switch to a stricter prompt format (e.g., `<|object_ref_start|>` style).
- **Tokenizer mismatch on field tags.** `<brand_name>` is multi-token in Qwen; if generation drops one token mid-tag it breaks the parser. Edit_f1 is robust to this; macro_f1 isn't. If macro_f1 tanks but edit_f1 looks reasonable, we'll know.

## Configuration

See `config.yaml` (snapshot of `experiments/configs/qwen_baseline.yaml`).

Key knobs:
- `Qwen/Qwen2-VL-2B-Instruct` loaded in 4-bit (nf4 + double-quant + bf16 compute).
- LoRA r=8, alpha=16, dropout=0.05 on attention projections only.
- Effective batch 16 (4 × grad_accum 4); peak_lr 2e-4 with cosine schedule and 50-step warmup; 500 max_steps.
- Eval every 100 steps with the same `prepare.evaluate(...)` used by Track A — apples-to-apples.

## Citation

- Wang et al. 2024, "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" — primary architecture reference; Naive Dynamic Resolution and OCR-tuned vision encoder are the proximal reasons we expect a label-reading boost.
- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models" — adapter method; r=8/alpha=16 follows their prescribed-strength heuristic for tasks within the model's pretraining domain.
- Dettmers et al. 2023, "QLoRA: Efficient Finetuning of Quantized LLMs" — 4-bit nf4 + LoRA combination; we lift the recipe (compute_dtype=bfloat16, double-quant, paged optimizers not strictly needed at this scale).

## Result

_(filled in after the run lands)_

```
final_macro_f1=
final_macro_edit_f1=
wall_clock_seconds=
exit_code=
```

## Retrospective

_(filled in after the run lands — what surprised us, what to try next)_
