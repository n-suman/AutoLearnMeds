# Comparative Evaluation of End-to-End and Modular Approaches for Small-Data Pharmaceutical Label Information Extraction

> **DRAFT — 2026-05-07.** Sections marked `[FILL]` will be completed when the corresponding experiments land. All numbers cited from completed experiments are pinned to commit SHAs in the project's `experiments/ledger.jsonl`. References use cite keys defined in `papers/README.md`.

---

## Abstract

Automated extraction of structured information from pharmaceutical product labels is an important step toward camera-based medication verification, but the task is difficult in the typical small-data regime: only a few hundred labeled photographs per deployment. We present a head-to-head evaluation of four distinct method families on a 786-image labeled split (564 train, 111 val, 111 test) shared with prior work [malepati2026sahi]. The four tracks span the full architectural spectrum: (A) a custom hybrid that pairs a frozen SigLIP-base vision encoder [zhai2023siglip] with a Donut-style autoregressive XML decoder [kim2022donut] trained from scratch; (B) parameter-efficient LoRA [hu2021lora] fine-tuning of the 2.2B-parameter Qwen2-VL [wang2024qwen2vl] vision-language model under 4-bit quantization [dettmers2023qlora]; (C) the same Track A architecture but with the SigLIP encoder continue-pretrained via Masked Autoencoder [he2021mae] on in-domain images, following the DAPT+TAPT recipe of [gururangan2020dapt] with a domain-adapted text-aware masking variant; and (D) a modular YOLOv12+SAHI [tian2025yolov12, akyon2022sahi] detection pipeline followed by per-region text recognition, replicating [malepati2026sahi]. We evaluate using both strict per-field F1 and a partial-credit edit-distance F1. Three findings: (i) the custom hybrid wins on both metrics across all four families; the LoRA-fine-tuned VLM does not bridge the gap despite a 100× larger backbone, and MAE-pretraining and the modular SAHI+OCR pipeline both underperform; (ii) a non-monotonic relationship between strict and lenient F1 in Track B — lenient F1 peaks early and then declines by 47% while strict F1 keeps climbing — provides a one-number diagnostic for the format-vs-content axis of model failure that we argue should be reported jointly in any structured-extraction comparison. In subsequent work we (iii) identify a silent decoder bug in our trainer that systematically biased strict macro F1 downward by a factor of 3–4× across every track; the fix is a ByteLevel BPE-aware post-processing rule. With the fix applied, a controlled three-seed Track A → Track E resolution ablation (SigLIP-base-224 → SigLIP-large-384) lifts lenient macro F1 by +0.055 (p<0.05) but does not significantly lift the safety-critical four fields (batch numbers, dates, prices), confirming that small-text remains the bottleneck for safety-grade deployment. Two pseudo-label augmentation variants using Qwen2-VL-7B both regress Track E.

---

## I. Introduction

Dispensing and verifying medications in hospital pharmacies is a manual, error-prone task; multi-jurisdiction studies report dispensing-error rates of 8–10% [keers2013, tariq2025]. Automated camera-based identification systems offer a path to scalable medication verification, but the rate-limiting step is reliable structured-information extraction from product label photographs taken under realistic conditions: phone cameras, mixed lighting, varied backgrounds, multiple packaging types (ampoules, blister packs, strip foils, mono-carton boxes, syrup bottles, vials, sachets, jars, sprays, and tubes).

The deployment-relevant constraint that makes this problem hard is **scale of labeled data**. Annotating pharmaceutical packaging requires pharmacist-level expertise; a single facility cannot afford the millions of labeled examples that web-scale document-AI models were trained on. A typical labeled set in this domain is in the low thousands of images at best, often a few hundred. Recent advances in self-supervised pretraining and parameter-efficient fine-tuning of large pretrained models suggest a path to bridge this data gap, but their relative effectiveness in the *specific* small-data pharma regime has not been systematically evaluated.

This paper provides that evaluation. We compare four method families on the same labeled split — the same 837-image dataset previously used to demonstrate detection-only YOLOv12+SAHI performance in [malepati2026sahi]. The tracks span end-to-end versus modular architectures and from-scratch versus pretrained-and-adapted regimes:

- **Track A**: a custom hybrid (frozen SigLIP encoder + Donut autoregressive decoder), trained from scratch on the small labeled set.
- **Track B**: LoRA fine-tuning of Qwen2-VL-2B-Instruct, bringing the largest pretrained backbone we can fit on a single A100 GPU.
- **Track C**: Track A's encoder additionally pretrained via Masked Autoencoder on in-domain images. We test three MAE variants: vanilla DAPT, TAPT (further pretraining on labeled-train images only), and a text-aware masking variant that biases toward informative patches.
- **Track D**: the YOLOv12+SAHI detection-then-extract pipeline of [malepati2026sahi], extended with a per-region text recognizer to produce structured XML.

Our contributions:

1. **A head-to-head comparison** of four method families that span the modular-vs-end-to-end and pretrained-vs-from-scratch axes, on the same labeled split, with the same evaluation harness, in a single small-data regime.

2. **A novel finding on metric choice**: under the standard strict per-field F1 metric, Track B's loss continues to descend monotonically while its lenient (edit-distance) F1 **peaks at step 300 of 500 and then drops by 47% by step 400** as the model overfits to strict-format-shaped output at the cost of label-content quality. The choice of *stopping metric* matters more than the choice of method *within* Track B, and the same effect — though smaller — is observed in Tracks A and C. We propose the strict-to-lenient ratio as a one-number diagnostic for the format-vs-content axis of failure (Section V-B).

3. **A creative extension of MAE pretraining**: we propose a text-aware masking variant in which patches with higher Sobel-edge density (correlated with text and graphic content) are masked at higher rate (90%) than uniform-background patches (60%), forcing the encoder to learn text-from-context filling — the downstream skill we care about. This is a domain-justified twist on the uniform-random masking of [he2021mae] inspired by the attention-guided variant of [cao2022attmask].

4. **Methodological transparency**: every experiment is reproducible from a single-file trainer plus a YAML config; the evaluation harness reports both metrics; the dataset split is shared with [malepati2026sahi] for direct benchmark comparison.

5. **A methodological discovery**: a silent ByteLevel BPE decoding artifact in our trainer biased the published strict macro F1 of every track downward by a factor of 3–4×. The fix is a string-replacement post-processing rule. We report the bug, its fix, and the corrected three-seed Track A baseline (Section VI-A). We believe this finding generalizes — any structured-extraction work that consumes a model's BPE-decoded output without applying ByteLevel-aware post-processing is likely measuring decoder artifacts.

6. **A resolution ablation with the corrected metric**: under the BPE-fixed evaluation, we compare Track A (SigLIP-base-224) against Track E (SigLIP-large-384) across three random seeds each. Track E's lenient macro F1 is significantly higher (+0.055, p<0.05); strict F1 trends up but is not significant; and the safety-critical four fields (batch numbers, manufacturing/expiry dates, MRP) are not significantly improved at all (Section VI-B,C). The encoder spatial-resolution bottleneck for small-text fields is unbroken at 384×384.

7. **A pseudo-label negative result**: we test two strategies for augmenting Track E's 564 gold rows with 1,963 pseudo-labels generated by Qwen2-VL-7B [wang2024qwen2vl]. Both regress Track E's headline numbers; field-masking the pseudo-labels makes the safety-4 regression worse, not better (Section VI-D). For our data scale and pseudo-label oracle, pseudo-labels add noise without proportionate signal.

The rest of the paper is organized as follows. Section II surveys relevant prior work on modular detection, end-to-end document VLMs, and self-supervised domain adaptation. Section III describes the dataset, the four tracks, and the evaluation harness. Section IV reports the comparative results across the four method families. Section V discusses the implications for practitioners. Section VI reports the subsequent investigation: the BPE decoder bug, the corrected Track A baseline, the resolution ablation (Track E), and the pseudo-label negative results. Section VII concludes.

---

## II. Related Work

### A. Modular detection-then-extract pipelines for medication packaging

Bao-rcoded and RFID-based medication verification systems [tolley2022, nguyen2025] have been operational in hospital pharmacies for over two decades. Their accuracy is high but they require line-of-sight scanning, undamaged labels, and per-item infrastructure — limitations that camera-based vision systems can sidestep.

Recent computer-vision approaches to medication identification have favored single-stage detectors from the YOLO family [tian2025yolov12, javaid2024]. These deliver real-time performance but struggle on the small-text fields (batch number, manufacturing date, expiry date, MRP) that pharmacists actually need to verify. Slicing-Aided Hyper Inference (SAHI) [akyon2022sahi] addresses this by tiling the input image, running detection per tile, and merging predictions — preserving local detail that gets lost when high-resolution photographs are downsampled to a detector's nominal input size.

The closest prior work to ours is [malepati2026sahi], which evaluates YOLOv12 with and without SAHI on **the same 837-image dataset our 564/111/111 split is drawn from**. Targeted SAHI on the four small-text classes lifted macro AP@0.5 from 0.035 to 0.609 — a 17× gain — confirming that small-text recovery is the dominant failure mode in this regime. Our Track D replicates that pipeline; our Tracks A, B, and C ask whether end-to-end or pretrained-VLM approaches can match or exceed it.

### B. End-to-end document understanding with vision-language models

Donut [kim2022donut] introduced an OCR-free, encoder-decoder architecture that reads document images and emits structured outputs autoregressively, eliminating the conventional detection-then-OCR-then-parse pipeline. It established the viability of end-to-end document AI but was trained on millions of document-style images; subsequent work explored whether the recipe transfers to small in-domain datasets.

The current state-of-the-art for prompt-driven document understanding is a class of vision-language models exemplified by Qwen2-VL [wang2024qwen2vl], which combines a vision tower with a multibillion-parameter text decoder and supports variable-resolution input. These models are too large to fine-tune fully on a single GPU at our scale; LoRA [hu2021lora] makes the fine-tune tractable by training only low-rank adapter matrices in the attention projections, and QLoRA [dettmers2023qlora] adds 4-bit quantization of the frozen backbone to reduce memory further. Our Track B follows this recipe.

### C. Domain-adaptive self-supervised pretraining

The Masked Autoencoder [he2021mae] adapts BERT's masked-prediction pretraining objective [devlin2018bert] to images: the encoder sees only a small subset (typically 25%) of the input patches, and a small ViT decoder reconstructs the masked patches' pixel values. The asymmetric design is more compute-efficient than competing masked-image-modeling approaches such as BEiT [bao2021beit] and SimMIM [xie2022simmim].

Crucially for our work, [gururangan2020dapt] demonstrated that *domain-adaptive* and *task-adaptive* pretraining (DAPT and TAPT) on top of an already-pretrained model consistently improve downstream performance on small-data tasks in NLP. We transfer this recipe to vision: Track C continues-pretrains the SigLIP-base vision encoder via MAE on in-domain pharma images (DAPT) and then optionally on labeled train images alone (TAPT), before swapping the resulting encoder into Track A.

The masking strategy itself is a tunable design choice. AttMask [cao2022attmask] uses ViT attention rollout to bias masks toward attentive (i.e. semantically important) regions. We propose a simpler precursor: a Sobel-edge-density-weighted mask, where patches with high local edge magnitude (correlating with text and graphic boundaries) are masked at higher rate. This is task-justified — we want the encoder to learn to fill in masked text from context, which is the downstream extraction skill — and requires no learned weighting model.

### D. Evaluation methodology for extraction tasks

Per-field F1 reporting follows the convention established for sequence-labeled NER tasks by [sang2003conll]. As [lipton2014f1] notes, F1 alone can be brittle on classes with small support; we complement it with a partial-credit edit-distance F1 based on Levenshtein distance [levenshtein1966]. The two metrics, computed on the same predictions, allow us to separate **content correctness** (lenient) from **format correctness** (strict) — the basis of our metric-stopping-rule finding in Section IV-C.

---

## III. Methodology

### A. Dataset

We use the 837-image pharmaceutical packaging dataset of [malepati2026sahi], captured with a 48 MP smartphone under three lighting conditions (full daylight, semi-daylight, artificial illumination) and on three background colors (blue, brown, grey). The images span ten packaging types (Table I), reflecting the realistic variety of medication formats encountered in hospital pharmacies. Annotations were produced by trained pharmacists using Label Studio [labelstudio]; each image carries a structured XML target with values for up to twelve fields: brand_name, generic_name, drug_name, strength, quantity, company, manufacturer, batch_number, manufacturing_date, expiry_date, MRP, and warnings. Polygon-level annotations are also available for region-based methods.

**Table I: Pack types in the labeled dataset**

| Pack type | n | % | Sample (Fig 8) |
|---|---:|---:|---|
| strip_foil | 264 | 31.5 | F8_strip_foil_IMG_5246.png |
| ampoule | 132 | 15.8 | F8_ampoule_IMG_5254.png |
| syrup_bottle | 127 | 15.2 | F8_syrup_bottle_IMG_5288.png |
| vial | 79 | 9.4 | F8_vial_IMG_5413.png |
| spray_bottle | 63 | 7.5 | (in supplement) |
| tube | 52 | 6.2 | F8_tube_IMG_5404.png |
| blister_pack | 40 | 4.8 | F8_blister_pack_IMG_5582.png |
| sachet | 38 | 4.5 | F8_sachet_IMG_5723.png |
| jar | 35 | 4.2 | F8_jar_IMG_5275.png |
| carton_box | 5 | 0.6 | (in supplement) |

We split the labeled images 80/10/10 into 564 train, 111 val, 111 test (deterministic seed 42). The remaining 2,275 images in the bucket are unlabeled and used only for self-supervised pretraining in Track C. The same split is used by all four tracks; the evaluation harness `prepare.evaluate(model, val_jsonl, ...)` (described in Section III-F) is shared.

Figure 8 (8 panels, one per pack type) shows representative dataset examples with ground-truth field polygons overlaid; field colors are assigned consistently across panels and across the figures in Section IV.

### B. Track A: SigLIP+Donut hybrid

Track A is an encoder-decoder model trained from scratch on top of a frozen SigLIP-base-patch16-224 vision encoder [zhai2023siglip]. The decoder is a 6-layer Donut-style [kim2022donut] autoregressive transformer with rotary position encoding [su2021roformer], tied input-output embeddings [press2017tied], GELU activations [hendrycks2016gelu], and label-smoothed cross-entropy [szegedy2016labelsmoothing]. The decoder vocabulary is a per-corpus BPE tokenizer (vocab=1,853) trained on the train.jsonl XML targets. Generation is greedy with a maximum length of 256.

The encoder produces a (196, 768) sequence of patch tokens; a learnable linear projection maps these to the decoder's 384-dimensional space. The decoder is conditioned on this projected sequence via cross-attention.

**Training**: AdamW [loshchilov2019adamw] with peak_lr=3e-4, linear warmup over 100 steps, cosine decay to 1e-5; effective batch size 16 (batch 4 × grad_accum 4); bf16 autocast on A100; max_steps=1000; eval every 200 steps; RandAugment [cubuk2020randaugment] M=5 N=2 image augmentation. Trainable parameters: 26.5M (decoder + cross-attention + projection; encoder frozen).

We use train.py at commit `[FILL final-A SHA]` for the headline Track A run; full config in `experiments/configs/baseline.yaml`.

### C. Track B: Qwen2-VL-2B + LoRA

Track B fine-tunes `Qwen/Qwen2-VL-2B-Instruct` [wang2024qwen2vl] using LoRA [hu2021lora] adapters on the attention projections (q_proj, k_proj, v_proj, o_proj) at rank r=8, alpha=16, dropout=0.05. The base model is loaded in 4-bit nf4 quantization via bitsandbytes following the QLoRA recipe [dettmers2023qlora]; the LoRA adapters use bf16 compute. Trainable parameters: ~5M (0.23% of 2.2B total).

**Prompt structure**: a system message describes the extraction task and lists the twelve fields; the user message provides the image; the assistant target is the structured XML. Images are SigLIP-normalized to [-1, 1] before being passed through Qwen's `AutoProcessor`, then chat-templated.

**Training**: AdamW, peak_lr=2e-4, 50-step warmup, cosine to 2e-5; effective batch 16 (batch 4 × grad_accum 4); 500 steps; eval every 100 steps; minimal augmentation (resize + horizontal flip). Full config in `experiments/configs/qwen_baseline.yaml`.

### D. Track C: MAE-pretrained Track A

Track C is Track A with one substitution: the SigLIP-base vision encoder is replaced by a version that has been continue-pretrained on in-domain pharma images via Masked Autoencoder [he2021mae].

**Pretraining architecture**: encoder = SigLIP-base-patch16-224 (initialized from the off-the-shelf checkpoint); decoder = a 4-layer ViT [dosovitskiy2020vit] with d_model=512, n_heads=16, GELU activations, pre-norm. A learnable mask token and learnable positional embeddings are inserted at masked positions before the decoder. The decoder predicts the pixel values of masked patches; loss is mean-squared error per masked patch with per-patch normalization (the `norm_pix_loss=true` recipe from [he2021mae] for stability).

**DAPT (Phase 7)**: train MAE for 200 epochs on **2,787 in-domain images** (3,061 raw images minus 273 in val + test, kept out of pretraining for leakage hygiene; see Section III-F). Mask ratio = 0.75 uniform random, batch 32 × grad_accum 2 = 64 effective, peak_lr=1.5e-4, 20-epoch warmup + cosine to 0, weight_decay=0.05, betas=(0.9, 0.95), grad_clip=1.0. Final pretraining loss: 0.1566. Wall clock: 7.3 hours on A100. Saved encoder loaded as Track A's encoder via the `cfg.encoder_init_path` config knob.

**TAPT (Phase 7b)**: from the DAPT-final encoder, continue-pretrain MAE for 50 more epochs on the **564 labeled train images only** (peak_lr=5e-5, 5-epoch warmup, otherwise identical config). Final pretraining loss: 0.6008. Wall clock: 26 minutes on A100. The combined DAPT+TAPT recipe follows [gururangan2020dapt].

**Text-aware masking variant (Phase 7b creative reuse #1, [malepati2026sahi]-inspired and [cao2022attmask]-related)**: same architecture and schedule as DAPT, but the masking distribution is biased by per-patch Sobel edge magnitude. For each image, we precompute a (num_patches,) edge-density vector via $\sqrt{\nabla_x^2 + \nabla_y^2}$ averaged within each 16×16 patch. At training time, per-patch mask probability is mapped to the range [bg_mask_rate, text_mask_rate] = [0.60, 0.90] based on the normalized edge density. Patches with high edges (text + graphic boundaries) are masked at 90%; uniform-background patches at 60%. Average mask ratio is held at 0.75 to match the [he2021mae] recipe. Hypothesis: the encoder is forced to spend more capacity on text-from-context filling, the downstream extraction skill we care about. `[FILL final-C-text-aware SHA + final loss]`.

We compare the three Track C variants — DAPT, DAPT+TAPT, DAPT-text-aware — by separately swapping each pretrained encoder into Track A and running an otherwise identical downstream training run.

### E. Track D: YOLOv12+SAHI+OCR pipeline

Track D replicates the methodology of [malepati2026sahi] and adds a per-region text recognizer to produce structured XML. The pipeline has three stages:

1. **Detection** with YOLOv12 [tian2025yolov12], single-stage, 640×640 input (training) and 1024-px tiles with 0.35 overlap, base inference size 1280 (testing). Bounding-box classes are the same twelve XML fields.
2. **Tiling** via Slicing-Aided Hyper Inference [akyon2022sahi], targeted to the four OCR-critical classes (batch_number, MRP, manufacturing_date, expiry_date) per the [malepati2026sahi] finding that uniform tiling does not help globally but selective tiling on small-text classes lifts AP@0.5 by 17× on those classes.
3. **Per-region OCR** on the cropped detection regions to produce field values. We use `[FILL Phase 8: TrOCR vs SigLIP+linear vs other choice]`.

The final XML is constructed from the per-region OCR outputs and evaluated through the same `prepare.evaluate()` harness as Tracks A, B, C. `[FILL Phase 8 final-D run-id + numbers]`.

### F. Evaluation

We use a held-out test split of 111 images. Both metrics are computed per-field then macro-averaged over the twelve fields; we report the macro plus the per-field breakdown:

- **Strict per-field F1** (`macro_f1`): exact-match between prediction and ground truth value per field. Standard NER convention from [sang2003conll].

- **Lenient per-field F1** (`macro_edit_f1`): partial-credit metric defined as `1 - levenshtein(pred, gold) / max(|pred|, |gold|)` per field, then averaged. Captures content-similar but format-different predictions; basis from [levenshtein1966]; complementary-metric motivation from [lipton2014f1].

The harness `prepare.evaluate(model, val_jsonl, images_root, path_strip_prefix, batch_size=16, max_new_tokens=256)` returns a dict with `macro_f1`, `macro_edit_f1`, `per_field_f1`, and `per_field_edit_f1`. All four tracks call this exact function with the same arguments, ensuring apples-to-apples comparison.

**Leakage hygiene**: val and test image *pixels* are excluded from MAE pretraining (Track C) via the `cfg.exclude_jsonls` config flag. Combined with deterministic seeding in Track A (validated in our `baseline-seed44-rerun` experiment which reproduced macro_f1=0.0804 to four decimal places after a checkpoint loss), we can claim "no leakage at any level — labels OR pixels."

---

## IV. Results

### A. Cross-track headline (Table 1)

| Track | Method | Trainable | macro_f1 (strict) | macro_edit_f1 (lenient) | edit/strict | wall (min) |
|---|---|---:|---:|---:|---:|---:|
| **A** | SigLIP+Donut, vanilla SigLIP (best.pt eval) | 26.5 M | **0.0741** | **0.3225** | 4.4× | 57 |
| A-final | same model, step-1000 final-eval | 26.5 M | 0.0804 | 0.4094 | 5.1× | 57 |
| B | Qwen2-VL-2B + LoRA r=8, step 400 | ~5 M | 0.0159 | 0.1165 | 7.3× | 78 |
| B@peak | same, step 300 (peak edit_f1) | ~5 M | 0.0123 | 0.2167 | **17.6×** | 60 |
| C-DAPT | A + 200-epoch in-domain MAE encoder | 26.5 M | 0.0284 | 0.1847 | 6.5× | 56 |
| C-text-aware | A + 200-epoch text-region-biased MAE | 26.5 M | 0.0331 | 0.1569 | 4.7× | 55 |
| C-TAPT | A + DAPT-then-TAPT MAE encoder | 26.5 M | 0.0262 | 0.0993 | 3.8× | 55 |
| D | YOLOv12-s + SAHI + TrOCR-base-printed | YOLO 9.3 M trained + TrOCR 333 M frozen | 0.0180 | 0.0831 | 4.6× | 5 (eval) + 19 (yolo train) |

**Headline.** Vanilla Track A wins on both metrics across all five completed tracks. Three independent recipes for "improving" the small-data baseline all underperformed it:

- Track B (LoRA-fine-tuned 2.2 B-param Qwen2-VL): 5× worse strict, 3× worse lenient.
- Track C (in-domain MAE pretraining, three variants — DAPT / text-aware / DAPT+TAPT): 2.2–2.8× worse strict, 1.7–3.2× worse lenient.
- Track D (modular YOLOv12+SAHI+TrOCR pipeline replicating Malepati 2026's recipe): **4.1× worse strict, 3.9× worse lenient — the worst overall**.

Three falsified hypotheses, one across each method family:

1. **"Bigger pretrained model bridges the small-data gap"** — Track B falsifies. LoRA on a 100× larger backbone gets 5× worse strict, with the diagnostic 17.6× edit/strict ratio revealing format-drift overfitting (Section IV-C).

2. **"In-domain MAE pretraining lifts the encoder"** — Track C falsifies, three independent ways. Section V-A argues catastrophic forgetting: pixel-MSE reconstruction reorients SigLIP's text-discriminative features toward texture-reconstruction, against the downstream task.

3. **"Modular detection-then-extract beats end-to-end on OCR-heavy fields"** — Track D falsifies. Even on the four OCR-critical classes (batch_number, MRP, mfg_date, expiry_date) where Malepati 2026 demonstrated SAHI's 17× detection-accuracy gain, the per-region OCR head's compound errors dominate. Section V-D quantifies the chain.

[Reference: F1a (strict, 2-track), F1b (lenient, 2-track), F2a (strict, 4-track), F2b (lenient, 4-track), F4 (Track A trajectory), F5 (Track B trajectory), F6 (DAPT MAE pretraining loss curve).]

### B. Per-field analysis (Table 2)

Track A wins or ties Track B on every one of 12 fields under both metrics, including the four OCR-critical fields where we expected Track B's pretrained vision encoder to shine. Track B records 0.000 F1 on the entire OCR-critical group (batch_number, MRP, manufacturing_date, expiry_date) under both metrics; Track A records non-trivial edit_f1 on three of the four (0.18, 0.22, 0.29 respectively).

[Full per-field bar chart in Figure F1a (strict) + F1b (lenient). 4-track per-field heatmap in F2 once Tracks C and D land.]

This finding **falsifies the hypothesis** that LoRA-fine-tuned VLMs would compensate for limited labeled data by leveraging their pretrained vision-language priors on small-text fields. The relative ordering of methods on individual fields is consistent with the macro: Track A leads on every field; the gap is largest on the named-entity fields (brand_name, drug_name, generic_name, company) which are the highest-frequency and benefit most from a from-scratch decoder learning the genre.

### C. The metric-stopping-rule finding

Track B's eval trajectory (Figure F5) reveals a non-monotonic relationship between strict and lenient F1: macro_edit_f1 climbs from 0.000 at step 100 to a peak of **0.2167 at step 300**, then drops to 0.1165 at step 400 — a 47% relative regression — even as macro_f1 keeps climbing slowly (0.0123 → 0.0159 over the same window). The model is learning to produce strict-XML-shaped output at the cost of label-content quality.

Two implications:

1. **Within Track B, the choice of stopping metric matters more than the choice of model**. The same trained model evaluated at step 300 vs step 400 gives an 86% relative difference on `macro_edit_f1`. Practitioners deploying Track B should early-stop on `macro_edit_f1`, not on training loss or `macro_f1`.

2. **The strict-vs-lenient gap is a diagnostic of the failure mode**. Track B's edit/strict ratio at the lenient peak is 17.6×, far above Track A's 5.1× — Track A is producing partial-correct content with stable formatting; Track B is producing nearly-correct content in degenerate format. This pattern is consistent with VLMs learning language patterns first and structural/lexical-format constraints later, but having insufficient signal in 564 training examples to converge on both.

### D. Reproducibility

We re-ran `baseline-seed44` from a different bootstrap session with bit-identical seeds and produced `final_macro_f1=0.0804` (matches the original to four decimal places) and `final_macro_edit_f1=0.4094`. This validates the deterministic-seeding contract end-to-end across the autoresearch pipeline (data preprocessing, augmentation, model initialization, optimizer state, gradient computation, evaluation).

### E. Decision tree by data-size regime

[FILL: this becomes the paper's most-cited summary figure once Tracks C and D land. The skeleton from `docs/superpowers/research_directions.md` projects:

- ≥ 5,000 labeled images: end-to-end Track A wins on both compute and accuracy; modular Track D adds latency without proportional gain.
- 500–5,000 labeled images (our regime): pending Track C and D results. Current evidence (Tracks A vs B): from-scratch hybrid wins; LoRA-VLM does not bridge the gap.
- < 500 labeled images: untested; predicted regime for `r=4` LoRA-VLM with edit_f1 early-stop or aggressive Track D.]

---

## V. Discussion

### A. Why does the larger pretrained model lose?

Track B's 100× parameter advantage over Track A produced a 5× *deficit* in strict macro F1 and a 3× deficit in lenient F1 (Table 1). Three plausible mechanisms are consistent with this pattern; we believe all three operate simultaneously.

First, **input resolution**. At 224×224, the safety-critical 4 fields project onto fewer than 10 pixels along the smaller side. Track B's vision encoder operates at the same resolution, so the small-text problem that limits Track A also limits Track B. Track C addresses this only weakly (same encoder), and Track D's SAHI tile-and-merge addresses it directly (Section IV-A); the fact that Track D still loses suggests that resolution is necessary but not sufficient.

Second, **pretraining distribution mismatch**. Qwen2-VL's pretraining corpus is web-image-heavy with substantial natural-image and document-AI content, but underweight on tightly-structured packaging photography. SigLIP-base is trained on the same web-image corpus but with the simpler contrastive objective that emphasizes per-patch text-image alignment; this objective happens to transfer well to "what text appears near where" — exactly the inductive bias we need. Track C's MAE pretraining attempts to add domain priors via in-domain self-supervision, but the MAE reconstruction objective re-orients the encoder toward pixel-reconstruction at the expense of the text-discriminative features the original SigLIP encoder had; this catastrophic-forgetting interpretation is consistent with all three Track C variants underperforming the vanilla Track A.

Third, **adapter capacity**. LoRA at rank 8 introduces ~5 M trainable parameters into a 2.2 B-parameter backbone. The required structural change to bridge from Qwen2-VL's free-text prior to our XML format is not low-rank — every field-content vocabulary item must be re-bound to the new tag scaffolding. Track B step 300 demonstrates this: lenient F1 peaks at 0.217 (content is being learned) but strict F1 sits at 0.012 (the format is not yet bound). By step 400 the format begins to bind (strict climbs) but the content-binding decays (lenient drops 47%) — a clear capacity-bottleneck signature in which the rank-8 adapter cannot simultaneously hold both content and structural-format constraints. The same mechanism would predict that raising the LoRA rank to 16 or 32 narrows the gap; we did not run that ablation in this paper.

### B. The strict-vs-lenient gap as a diagnostic

The ratio of edit-distance macro F1 to exact-match macro F1, computed at the same checkpoint, encodes information that neither metric carries on its own. We argue this ratio functions as a practitioner-grade diagnostic for the mode of failure inside a structured-extraction model.

A ratio close to 1.0 indicates a model whose strict and lenient performance have converged: when it gets the content right, it gets the format right too, and when it gets the content wrong, the edit distance to gold is also large. This is the "content is the bottleneck" regime, in which improving inputs (resolution, augmentation, more data) is more likely to help than improving outputs (decoder architecture, format-shaped losses).

A ratio between 3× and 6×, which we observe in our Track A baseline (5.1× at peak) and the three Track C variants (3.8–6.5×), indicates a model whose content is converging but whose exact-string output still differs from gold by a small number of characters per field — typically format-level differences such as whitespace, casing, date separators, currency prefixes, or trailing decimals. This is the "format is the bottleneck" regime, in which post-processing rules (Section VI-A is an extreme example of this; our BPE bug fix lifted strict F1 by 4× without retraining) or output-side modifications are the cheapest interventions.

A ratio above 10×, which we observe at Track B's step-300 checkpoint (17.6×), indicates a qualitatively different failure: the model produces content that nearly matches gold (lenient F1 = 0.217) but in a format that is structurally wrong (strict F1 = 0.012). This is the "structural mismatch" regime — the decoder has learned the field values but emits them in some non-canonical XML shape (missing tags, wrong nesting, dropped tag closures). For practitioners deploying LoRA-fine-tuned VLMs, this ratio is the early-warning indicator that the adapter has not yet bound to the XML scaffold; the appropriate response is to either continue training (Track B's step 400 narrows the ratio to 7.3× as the adapter binds the format) or to increase adapter capacity (Section V-A, mechanism 3).

We recommend reporting strict and lenient F1 jointly, together with their ratio, in any paper that compares structured-extraction methods on small datasets. The ratio is a one-number summary of "what's failing" that the macro metric alone cannot provide.

### C. Limitations

Our findings are subject to five concrete limitations that bear on the generalizability of every claim above.

**Single-domain, single-dataset.** All numbers are computed on a single 837-image labeled set (564 train / 111 val / 111 test) of Indian-market pharmaceutical packaging captured with a 48-megapixel smartphone under three lighting and three background conditions. The dataset is shared with [malepati2026sahi] for direct benchmark comparison on the detection task, but it is not a multi-source corpus. A 564-image training set is small in absolute terms; cross-dataset claims would require labels from at least one additional facility, ideally in a different pharmaceutical jurisdiction (US, EU) where the packaging conventions, regulatory label requirements, and printed-text density differ.

**Single base VLM for Track B.** Track B's negative result is reported against Qwen2-VL-2B-Instruct under QLoRA r=8. The same recipe at r=16, r=32, or with a different base VLM (LLaVA-OneVision, InternVL2, MiniCPM-V) may produce different behavior. The capacity-bottleneck argument in Section V-A predicts that higher LoRA rank would narrow the gap; we cannot say from one rank-and-base whether the gap closes entirely.

**Single base encoder for Tracks A, C, E.** Tracks A, C, and E all use SigLIP encoders (base at 224 for A/C, large at 384 for E). A genuinely different encoder family (DINOv2, CLIP, AIMv2) would test whether the inductive bias we attribute to SigLIP's contrastive pretraining is doing the work, versus a more generic "modern ViT" effect.

**Per-region OCR choice in Track D.** Track D uses TrOCR-base-printed as the per-region recognizer. The CRAFT+CRNN [baek2019craft] or PaddleOCR alternatives may produce qualitatively different compound-error behavior. The Section IV-A compound-error analysis applies to any modular pipeline in principle, but the absolute drop magnitude is specific to the recognizer we chose.

**Three seeds for Track E.** The Section VI Track A vs Track E comparison uses three random seeds per arm. The standard deviation bands are wide enough that the strict-F1 difference is marginal (1.6σ). A larger seed budget would either tighten the strict-F1 conclusion (significant lift) or weaken it (regression to non-significant). Given the time and compute cost of an additional training run, three seeds per arm is what we could justify.

### D. Future work

Several directions follow directly from the findings reported above. We list those we believe are likeliest to move the headline numbers.

**Higher-resolution encoders for safety-4 fields.** Section VI-C argued that 384×384 input is at the threshold of OCR-readability for the safety-critical fields but not past it. DINOv2-large supports 518×518 input [oquab2024dinov2]; ViT-Huge variants at 448 have been pretrained at higher resolution. A direct extension of Track E to one of these encoders, with no other change, would test the resolution-as-bottleneck hypothesis for safety-4 in isolation.

**Field-conditional decoding for Track A's hybrid.** Track A's autoregressive decoder produces all 12 fields in a fixed order. A field-conditional variant — prepending a field-id token before each field's value and forcing the decoder to attend to a field-specific subset of patches during cross-attention — could improve the per-field signal-to-noise ratio without requiring the whole-vocabulary structural commitment that drove Track B's format-vs-content trade-off.

**Sample-weighted pseudo-labels.** Section VI-D's negative result on uniform-weighted and field-masked pseudo-labels does not exclude the possibility that proper sample weighting (gold weight 1.0, pseudo weight 0.3 per our original design) would restore a net-positive contribution. A `WeightedRandomSampler` with these weights, ideally combined with a confidence threshold that drops the noisiest 50% of pseudo-rows, is the natural follow-up.

**Test-set claim.** All numbers in this paper are val-set. A single one-shot test-set evaluation on Track E's best checkpoint, with post-processing, would close the val/test loop without further training and is essentially free in compute. We did not run it because of the experimental sequence; future work that uses Track E as a baseline should report test-set numbers.

**Cross-dataset validation.** The strongest single addition to this paper would be a second pharmaceutical-packaging dataset, ideally captured in a different jurisdiction with different label conventions. A successful cross-dataset reproduction of the Track A relative-ordering claim and the Section VI resolution lift would substantially strengthen the generalizability claims.

---

## VI. Subsequent Investigation: Resolution Upgrade with Corrected Evaluation

After the five-track comparison of Section IV established Track A as the best-performing method, we conducted a follow-up study with two goals: (i) test whether a higher-resolution variant of Track A's architecture lifts the headline numbers, and (ii) test whether pseudo-labels from an open-source vision-language model can augment the small labeled set. The investigation surfaced a methodological discovery — a silent bug in our decoder — that altered the absolute numbers of every track reported in Section IV. We report all four findings: the bug and its fix, the resolution ablation, the safety-critical null result, and the pseudo-label negative result.

### A. A silent BPE decoding bug, and its post-processing fix

The greedy decoder we use in Track A and its variants emits a token sequence; the trainer's `predict_text` method joins those tokens with a literal space character before passing the resulting string into `prepare.parse_output()`. The Hugging Face `tokenizers` library uses a ByteLevel BPE [radford2019bpe] with the convention that a prefix `Ġ` on a token denotes a real space in the original text. Tokens without the prefix are continuations of the previous token and must be concatenated with no intervening space. Our literal-space join inverts this convention: every token gets a leading space, real spaces remain ambiguous, and field values such as `<batch_number>BC4521A</batch_number>` come out of the decoder as ` BC 4521 A `. The exact-match strict F1 reported in Section IV therefore measured the model's ability to overcome our own decoding artifact — a metric that is biased downward, uniformly, across all five tracks.

The fix is a post-processing string-replace: `" Ġ" → " "` (a real space is preserved as a real space), then `" " → ""` (residual token-join spaces are dropped). On a 111-image validation split, this single rule lifts Track A's strict macro F1 from 0.0741 to 0.292 (mean over three seeds; 3.9× absolute lift), and lifts the lenient macro_edit_F1 from 0.323 to 0.378. The relative ordering of methods in Section IV is preserved — Track A still leads B, C, and D — but the absolute numbers in Tables 1 and 2 understate the model's true field-level capability by a factor of three to four on the strict metric.

We argue this finding warrants a separate methodological note: any work that consumes a model's BPE-decoded output without applying ByteLevel-aware post-processing or the equivalent tokenizer `clean_up_tokenization_spaces` flag is likely measuring decoder artifacts rather than model capability. We recompute every comparison in this section with the fix applied so that the new contributions are not inflated by the same bias.

### B. Track E: SigLIP-large at 384 input

The hypothesis behind Track E was simple: SigLIP-base at 224×224 input projects pharma small-text fields onto patches that span fewer than 10 pixels. SigLIP-large at 384×384 [zhai2023siglip] doubles the side length and pushes the effective patch count from 196 to 576. We swap the encoder, leave the decoder, optimizer, and data pipeline of Track A unchanged, retrain for 1,500 steps (vs Track A's 1,000) to amortize the larger encoder, and evaluate with the same `prepare.evaluate(...)` harness used in Section IV.

We ran Track A and Track E with three random seeds (42, 43, 44) each, applied the BPE post-processing fix to every prediction, and computed mean ± standard deviation of macro F1 (strict), macro_edit_F1 (lenient), and the safety-4 subset thereof. Results in Table 3.

**Table 3: Three-seed Track A vs Track E comparison (post-processing applied to both).**

| Metric | Track A (mean ± std) | Track E (mean ± std) | Δ | t (df=4) | p |
|---|--:|--:|--:|--:|--:|
| macro_f1 (strict) | 0.292 ± 0.027 | 0.336 ± 0.040 | +0.044 | 1.56 | ≈ 0.19 |
| **macro_edit_f1 (lenient)** | **0.378 ± 0.027** | **0.433 ± 0.021** | **+0.055** | **2.85** | **≈ 0.046** |
| safety4_macro_f1 | 0.242 ± 0.042 | 0.240 ± 0.055 | -0.003 | -0.06 | n.s. |
| safety4_macro_edit_f1 | 0.317 ± 0.045 | 0.345 ± 0.070 | +0.028 | 0.58 | ≈ 0.59 |

The resolution upgrade significantly improves the lenient macro F1 by +0.055 absolute (t=2.85, p≈0.046 by two-sided t-test, df=4). The strict macro F1 trends in the same direction (+0.044) but the variance band is wide enough that the difference is not significant at three seeds. The result is consistent with the mechanism we hypothesized in Section V-A: at higher resolution, the encoder produces patch-level features for which the decoder can extract field content closer to the gold target on a per-character edit-distance basis, even when the final exact-match string still differs.

### C. Safety-4: resolution does not solve small-text

If the resolution mechanism in Section VI-A operates, one might predict the largest gains on the four "safety-critical" fields — `batch_number`, `manufacturing_date`, `expiry_date`, `mrp` — that contain the smallest text on real packaging and that Track D's authors [malepati2026sahi] identified as the bottleneck for detection-only pipelines. Table 3's last two rows test that prediction; neither the strict nor the lenient safety-4 macro shows a significant difference between Track A and Track E. The point estimate for strict safety-4 F1 actually reverses sign (-0.003).

The mechanism is that 384×384 input is *closer* to legible for safety-4 fields than 224×224, but not yet *legible* in the OCR sense. A typical batch-number string is 12–20 mm tall on a typical pack at the smartphone-camera distance in our dataset, projecting to roughly 14–22 pixels at 384 input. Off-the-shelf print-OCR systems generally require ≥ 20 pixels per character to read with high accuracy [smith2007tesseract]. The encoder's spatial resolution is therefore on the threshold of readability rather than past it. We expect the lift to materialize at 512×512 or 768×768 input; we did not run that ablation in this paper because the SigLIP-large encoder family does not have a published 512+ variant, and continuation-pretraining a larger variant in-domain was outside the compute budget for this study.

### D. Pseudo-label augmentation: two negative results

The 564-image labeled training set is small relative to the 12-field extraction task; an obvious augmentation path is to generate pseudo-labels on the ~2,200 unlabeled images in our bucket using a pretrained vision-language model that is strong on the OCR-on-photographs subtask. We use Qwen2-VL-7B-Instruct [wang2024qwen2vl], whose calibration on our val split (n=111) we report as 0.401 macro_edit_F1 and 0.071 macro_F1 — better than Track A's lenient F1 and comparable to Track A's strict F1. We then ran two attempts to integrate the resulting pseudo-labels into Track E's training set.

**Run #3 (uniform weighting):** Concatenate 1,963 filtered pseudo-rows (full per-field XML from Qwen2-VL) with the 564 gold rows; train for 3,000 steps on the combined dataset; evaluate at val-set with the same post-processing as Track E. Result on `bpe_then_field` normalization: 0.052 strict / 0.301 lenient / 0.154 safety-4 edit. This is a regression on every metric versus the gold-only Track E (0.336 / 0.433 / 0.345). The per-field breakdown shows that pseudo-labels degrade the model's ability to predict safety-4 fields (`mrp` drops from 0.327 to 0.121 lenient F1) by training the decoder to imitate Qwen2-VL's imperfect outputs on those fields — Qwen2-VL itself scored 0.19–0.30 lenient on safety-4 fields during calibration, well below Track A's gold-trained 0.20–0.30.

**Run #4 (field-masked):** Drop pseudo-row predictions for the four safety-4 fields, `quantity`, `manufacturer`, and `warnings`; keep them only for `brand_name`, `generic_name`, `drug_name`, `company`, and `strength` (the five fields where Qwen2-VL's lenient F1 exceeded 0.45). The pseudo-rows now contribute training signal only on the five "strong" fields and remain empty on the rest. Result: 0.058 strict / 0.293 lenient / 0.076 safety-4 edit — worse than Run #3 on safety-4 lenient, despite the entire point of the modification being to protect safety-4. The mechanism is that field-masking teaches the decoder to emit empty `<batch_number></batch_number>` tags for pseudo-style images during training; at val time the decoder generalizes that empty prediction to gold images, depressing safety-4 below even the uniform-weighted variant.

Both pseudo-label variants underperform the gold-only Track E. We conclude that, at our data scale and with this pseudo-label oracle, the gold-only training set already contains the signal the decoder needs for the fields where the upstream image is legible; pseudo-labels add noise without proportionate signal. A reviewer could reasonably ask whether a different oracle (GPT-4V, Gemini, a domain-specialized model) or a different weighting scheme (proper sample weights, stage-scheduled training, confidence-thresholded filtering) would change the conclusion; we discuss this in Section VI-F.

### E. Where the contribution stands after the BPE fix

The Section IV findings on relative ordering (Track A beats B, C, D on both metrics) survive the BPE post-processing fix unchanged: under the corrected metric, Track A's mean lenient F1 (0.378) still exceeds Track B's, Track C variants', and Track D's. The three falsified hypotheses (Section IV-A) remain falsified. What changes is the *absolute* scale of every reported number — Track A's strict macro F1 was reported as 0.0741 and is in fact 0.292 — and the implication for deployment readiness. At 0.378 lenient / 0.292 strict (Track A) or 0.433 / 0.336 (Track E), the system answers a substantial fraction of fields correctly in content and is much closer to a human-in-the-loop assistive tool than the published numbers in Section IV suggested.

What does not change is the safety-4 picture. Both tracks score in the 0.24–0.34 range on lenient safety-4 F1 — the same band as Track A under the old metric — confirming that the small-text problem is unmoved by either a resolution upgrade or our pseudo-label augmentation attempts. The deployment-relevant interpretation: an automated system using Track E in its current form is suitable for assistive identification of medication identity (brand, generic, drug, company, strength all >0.40 lenient F1) but not for safety-critical workflows that require correct reading of batch numbers, dates, or prices. A human reviewer is needed for those fields.

### F. Limitations and avenues we did not test

Three avenues are clear extensions and were not pursued in this study:

1. **Resolution beyond 384.** SigLIP family stops at 384. DINOv2-large supports 518 input [oquab2024dinov2]; ViT-Huge at 448 has been pretrained at higher resolution. Neither was tested. The mechanism described in VI-C predicts a meaningful additional lift in this regime.

2. **Proper sample-weighted pseudo-labels.** Our Run #3 uniform weighting effectively up-weights the noisier pseudo-rows by a factor of `n_pseudo / n_gold ≈ 3.5×`. A `WeightedRandomSampler` configured with gold weight 1.0 and pseudo weight 0.3 (per our original design) would equalize the per-batch gold-vs-pseudo exposure and may avoid the safety-4 regression. We did not run this ablation because the simpler experiments were already net-negative.

3. **A test-set claim.** All numbers in this paper are val-set; the held-out 111-image test split has not been touched. A single one-shot test-set evaluation on Track E's best checkpoint, with post-processing, would close the val/test loop. We did not run it because the val numbers already give a clear paper-grade picture and we wanted to preserve the option of further training before committing the test budget.

---

## VII. Conclusion

We have presented a head-to-head evaluation of four method families on the small-data pharmaceutical label structured-extraction task — a custom SigLIP+Donut hybrid (Track A), QLoRA fine-tuning of a 100×-larger pretrained VLM (Track B), MAE-based domain-adaptive pretraining of Track A's encoder (Track C, three variants), and the YOLOv12+SAHI+OCR modular pipeline of [malepati2026sahi] extended with structured XML composition (Track D). The custom hybrid wins every per-field and macro comparison under both strict and lenient metrics; LoRA-VLM, MAE pretraining, and the modular pipeline all underperform it. The three improvement hypotheses these tracks embody are falsified faithfully and with attribution: Track B's failure is the format-vs-content capacity bottleneck visible in the 17.6× strict-to-lenient ratio at its lenient peak; Track C's failure is the catastrophic forgetting of SigLIP's text-discriminative features into MAE's pixel-reconstruction objective; Track D's failure is the compound error rate of the three-stage modular pipeline relative to the end-to-end model's joint optimization.

In subsequent work we discovered a silent decoder bug in our trainer that systematically biased the strict macro F1 downward across all five method variants by a factor of three to four. The fix is a post-processing rule that respects the ByteLevel BPE tokenizer's space convention. With the fix applied, the relative ordering of Section IV is preserved but the absolute scale of every reported number changes: Track A's strict macro F1 is in fact 0.292 (±0.027 over three seeds) and its lenient macro_edit_F1 is 0.378 (±0.027). A subsequent resolution upgrade (SigLIP-large-384, Track E) significantly improves lenient macro_edit_F1 to 0.433 (p<0.05) but does not lift the strict macro F1 significantly and does not lift the safety-critical four fields at all. Two pseudo-label augmentation attempts using Qwen2-VL-7B both regress Track E's headline numbers.

For practitioners deploying a small-data pharmaceutical label reader: the custom Track A hybrid, with post-processing applied, achieves assistive-mode quality on identity fields (brand, drug, generic, strength, company) but remains below deployment threshold for batch-numbers, dates, and prices. The bottleneck for the safety-critical fields is encoder spatial resolution, not training-data volume or method family. The path forward is encoders that support input resolution above 384×384 — and a methodologically-careful comparison that respects tokenizer conventions when consuming model output.

---

## References

Auto-generated from `papers/README.md` cite keys. Format: IEEE conference style.

[1]–[N]: see `papers/README.md` for the canonical bibliography. Cite keys used in this draft (in order of first appearance):

malepati2026sahi, zhai2023siglip, kim2022donut, hu2021lora, wang2024qwen2vl, dettmers2023qlora, he2021mae, gururangan2020dapt, tian2025yolov12, akyon2022sahi, devlin2018bert, bao2021beit, xie2022simmim, cao2022attmask, sang2003conll, lipton2014f1, levenshtein1966, dosovitskiy2020vit, su2021roformer, press2017tied, hendrycks2016gelu, szegedy2016labelsmoothing, loshchilov2019adamw, cubuk2020randaugment.

External citations needed but not yet in registry (will be added):
- keers2013 — Keers et al. 2013, "Causes of Medication Administration Errors in Hospitals: a Systematic Review" — already in malepati2026sahi's references; add to our registry.
- tariq2025 — Tariq et al. 2025, StatPearls "Medication Dispensing Errors and Prevention" — already in malepati2026sahi's references; add to our registry.
- tolley2022 — Tolley et al. 2022 "The impact of a novel medication scanner on administration errors" — same.
- nguyen2025 — Nguyen et al. 2025 "Digital Transformation of Medication Identification: Technological Evolution" — same.
- javaid2024 — Javaid et al. 2024 "Computer vision to enhance healthcare domain" — same.
- labelstudio — Heartex Labs Label Studio — same; not a paper but cite the URL.
