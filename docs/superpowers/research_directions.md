# Research Directions — Creative Paper-Reading Backlog

**Discipline:** for every paper in our citation graph, separately track (a) what we used as the headline method and (b) techniques in the same paper or its follow-ups that we COULD repurpose for our specific goal — extracting structured text from pharmaceutical label images at the 564-train regime. Ranked by expected paper-impact, not by ease.

**Why this doc exists:** user-stated principle 2026-05-05; the project's success criterion is "interesting paper", not "shipped autoresearch loop." Cross-pollination of ideas is an explicit goal, not a nice-to-have.

---

## High-leverage queue

### 1. Text-aware MAE masking — He et al. 2021 (MAE), creative reapplication

**Headline used:** uniform random 75% patch masking.

**Repurposable:** mask text-region patches at higher rate than background. Use a tiny detector (CRAFT, EAST, or even Otsu+horizontal-projection on a thresholded grayscale version) to produce a binary text-likelihood mask per image. In `random_masking`, sample with weighted Bernoulli — 90% mask rate over text patches, 60% over background. The encoder is forced to learn *text-from-context* filling, which is the downstream skill.

Closely related: AttMask (Cao et al. 2022) uses attention-rollout from a partially-trained ViT to guide masking. Our text-detector approach is a simpler precursor — we know which regions matter without bootstrapping from attention.

**Expected:** macro_f1 0.07 → 0.09–0.10 over vanilla-MAE-init.
**Effort:** 1 day (text detector + mask weighting in dataset).
**Paper-leverage:** high — domain-justified twist on a generic recipe; differentiates our MAE from a literal copy of He et al.

### 2. Field-conditional decoding — Zhai et al. 2023 (SigLIP), creative reapplication

**Headline used:** SigLIP-base-patch16-224's vision tower as Track A's encoder.

**Repurposable:** SigLIP also has a strong text encoder. We have 12 field names. Use the SigLIP text encoder *once at training start* to embed each field name; pass the field embedding to Track A's decoder as a conditioning prefix; have the decoder generate just that field's value (or empty for absent). Instead of emitting `<brand_name>X</brand_name>` as one autoregressive sequence, the decoder is queried 12 times per image, once per field.

Why it's better:
- Factorizes "learn 12-field XML grammar from 564 images" into "learn one extraction strategy parameterized by 12 fields". Tighter generalization.
- Trivial multi-task extension — adding a 13th field is one new embedding, no decoder retraining.
- Decoder no longer has to learn the XML format itself; it just generates field values.

**Expected:** macro_f1 0.08 → 0.10–0.13. Highest expected paper impact of the queue.
**Effort:** 3 days (decoder architecture change, retraining).

### 3. Native-resolution Qwen2-VL — Wang et al. 2024, missed strength

**Headline used:** Qwen2-VL-2B-Instruct at 224×224 input.

**Repurposable:** Qwen2-VL's distinguishing feature is "Naive Dynamic Resolution" — variable input resolution from low to ~1280×1280. Pharma photos from phones are ~3024×4032; at 224×224 we discard ~99% of pixel content. The fields where Track B currently scores 0.0 (`batch_number`, `expiry_date`, `mrp`) are exactly where small printed text matters; they should benefit most from higher resolution.

**Effort:** 4 hours A100 (config change + re-run, no code).
**Paper integrity:** before claiming "Track A beats Track B in this regime", we owe at least one Track B run that uses Track B's actual architectural strengths. The current 224×224 comparison is unfair to Track B. Even if Track B still loses, the gap likely shrinks meaningfully.

### 4. TAPT (Task-Adaptive Pretraining) — Gururangan et al. 2020, follow-up not used

**Headline used:** DAPT (Domain-Adaptive Pretraining) — MAE on 2787 in-domain images.

**Missed from same paper:** TAPT consistently beats DAPT-alone in their experiments. After current MAE finishes, run a SECOND short MAE pass (50 epochs) on JUST the 564 labeled train images. Same code, different `images_root` / `exclude_jsonls`.

**Effort:** 30 min extra A100, no code change.
**Paper-leverage:** moderate; lifts the citation from "we used MAE" to "we used the full DAPT+TAPT recipe."

### 5. Heterogeneous-rank LoRA — Hu et al. 2021, missed prescription

**Headline used:** uniform LoRA r=8 across all 12 attention layers in Qwen2-VL.

**Missed:** the LoRA paper notes lower layers benefit less from adaptation. Try r=2 (bottom 4 layers), r=8 (middle 4), r=16 (top 4) — same total trainable params, focused capacity where it helps.

**Effort:** 1 day.
**Paper-leverage:** modest. Probably closes the Track B gap from 5× to 2–3× but doesn't flip the conclusion. Useful for "we genuinely tried Track B" credibility.

---

## Lower-leverage but documented

### 6. BEiT-style discrete-token reconstruction — Bao et al. 2021

**Repurposable for pretraining:** instead of pixel MSE, predict VQ-VAE codes of masked patches. Stronger loss signal for text-heavy regions where pixel reconstruction is essentially impossible.

**Why deprioritized:** requires training a VQ-VAE first on label crops; second-order dependency. Better-after-(1)-and-(2)-land.

### 7. SimMIM block masking — Xie et al. 2022

**Repurposable:** large rectangular block masks (32×32 patch blocks) instead of random patches. For text images, this forces longer-range context — text continues across the mask.

**Why deprioritized:** less novel than text-aware masking (#1); same direction.

### 8. CoT prompting for Qwen2-VL — Wang et al. 2024 (training data hint)

**Repurposable:** "First describe the visible label structure. Then list the fields you can read." 2-step prompt instead of direct XML emission. Tests whether Qwen2-VL has implicit structure-reading capabilities accessible via prompt-engineering.

**Why deprioritized:** zero-shot prompt tweaks are easy; if (3) shows Qwen-native-resolution still loses, this won't save it. If (3) flips the comparison, this becomes more interesting.

### 9. SigLIP-anchored continual learning — Zhai et al. 2023, creative

**Repurposable:** during MAE continue-pretraining, add an auxiliary contrastive loss that keeps the encoder's features close to vanilla SigLIP on natural images. Prevents catastrophic forgetting of general visual knowledge while we adapt for pharma.

**Why deprioritized:** speculative — we haven't actually observed catastrophic forgetting. Wait for the MAE-init Track A result; if it hurts on out-of-distribution evaluation we'll know.

### 10. CoNLL B-/I-/O- field-boundary F1 — Sang & De Meulder 2003, missed metric

**Repurposable:** treat the XML output as a tagged sequence; compute F1 on field BOUNDARIES separately from interior content. Different signal: rewards getting where each field starts/ends right even if the value inside is slightly wrong.

**Why deprioritized:** evaluation-only; doesn't change any model. Useful for the per-field paper table only.

---

## Highest-priority creative reuses from prior published work on THIS dataset

### 11. SAHI-style tiled inference for Track A — Malepati et al. 2026 IITCEE, direct application

**Headline finding from Malepati 2026 (co-authored by us):** YOLOv12 + Sliced Aided Hyper Inference (SAHI; Akyon et al. 2022) on 1024-px tiles with 0.35 overlap and base inference size 1280 lifted macro AP@0.5 on the four OCR-critical classes (Batch, MRP, Manufacturing Date, Expiry) from **0.035 → 0.609** — a **17× gain** — over the same YOLOv12 model without tiling. This is on the SAME 837-image dataset that our 564+111+111 train/val/test split is drawn from.

**Repurposable for Track A inference (no retraining required):**

At inference, slice the high-res input photo into N overlapping 224×224 tiles, run frozen-encoder + Donut decoder per tile, merge per-field predictions:

- For each field, take the highest-confidence prediction across tiles (NMS-style, but on the *XML field* axis rather than spatial bounding box).
- Track-A's decoder already outputs autoregressive tag-value pairs; per-tile we get a partial XML; we union the partial XMLs with confidence-tiebreak.
- Tile geometry: copy Malepati 2026's 1024×1024 tiles with 0.35 overlap, downsampled to 224 each. On a 4032×3024 native photo that's a 4×3 grid of overlapping tiles.

**Expected:** macro_f1 lift on OCR-critical fields proportional to Malepati 2026's 17× detection gain; we predict our current Track A field-level f1=0.0 → 0.10–0.20 on Batch/MRP/Mfg/Expiry, and macro_f1 0.0741 → ~0.10–0.13.

**Effort:** 1 day (inference-time-only change to per_field_eval.py; no retraining). Highest leverage in the queue.

### 12. Multi-tile MAE pretraining — same paper, repurposed for our Phase 7

**Repurposable for MAE training (replaces the current single-random-crop recipe):**

Currently `_ImagePathDataset.__getitem__` returns one random 224×224 crop per image per epoch. Replace with N tiles per image per epoch, computed via the same SAHI tiling scheme (1024-px tiles → 224 each). Each training step sees more native-resolution patch coverage, especially of small text. Forces the encoder to see the same image at multiple zoom levels, similar to how SAHI gives YOLOv12 17× lift via the inference-time analog.

**Expected:** smaller marginal lift than #11 because pretraining is already learning multi-scale via random_resized_crop, but cleaner small-text feature learning. Combine with text-aware masking (#1) for the strongest in-domain encoder.

**Effort:** 2 days (dataset rewrite + a separate Phase 7d MAE run).

### 13. Track D = YOLOv12+SAHI pipeline as a competitor track — full paper realignment

**The biggest creative move:** replicate Malepati 2026's YOLOv12+SAHI pipeline as **Track D** in our comparative paper. The pipeline is detect → crop → small text recognizer (TrOCR or similar) → aggregate into XML. Evaluate via the SAME `prepare.evaluate()` we use for Tracks A/B/C, on the SAME val/test split.

**Why this changes the paper:**

The current paper framing is "Track A (custom hybrid) vs Track B (Qwen+LoRA) vs Track C (MAE-init)." Every one of these is end-to-end. Adding Track D reframes the contribution as **"end-to-end VLMs vs modular detection-then-extraction in small-data pharma extraction"** — a more honest, more paper-grade framing.

Malepati 2026 publishes AP@0.5 = 0.61 macro for SAHI on the OCR-critical classes. If our end-to-end approaches match or beat this on per-field strict F1 we have a clear "end-to-end is competitive even at small data" story; if not, we have a clear "modular wins for OCR-critical fields, end-to-end wins for content fields" story. **Either way the paper is stronger.**

**Effort:** ~1 day to wire YOLOv12+SAHI inference + add a TrOCR or SigLIP-text-head per-region recognizer. ~6 hours A100 to retrain YOLOv12 if Malepati's weights aren't reusable; if they are, just inference. **Phase 8.**

---

## How this list updates

After each experiment lands:
1. Move the entry from queue to "done" with the result delta.
2. Re-rank the queue based on what we learned.
3. Add new ideas only when we find a paper-grounded justification (not "what if we tried X").

Default principle: prefer the paper-cited creative reuse over a novel ad-hoc idea. Easier to defend in review, easier to build on.
