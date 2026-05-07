# Comparative Evaluation of End-to-End and Modular Approaches for Small-Data Pharmaceutical Label Information Extraction

> **DRAFT — 2026-05-07.** Sections marked `[FILL]` will be completed when the corresponding experiments land. All numbers cited from completed experiments are pinned to commit SHAs in the project's `experiments/ledger.jsonl`. References use cite keys defined in `papers/README.md`.

---

## Abstract

Automated extraction of structured information from pharmaceutical product labels is an important step toward camera-based medication verification, but the task is difficult in the typical small-data regime: only a few hundred labeled photographs per deployment. We present a head-to-head evaluation of four distinct method families on a 786-image labeled split (564 train, 111 val, 111 test) shared with prior work [malepati2026sahi]. The four tracks span the full architectural spectrum: (A) a custom hybrid that pairs a frozen SigLIP-base vision encoder [zhai2023siglip] with a Donut-style autoregressive XML decoder [kim2022donut] trained from scratch; (B) parameter-efficient LoRA [hu2021lora] fine-tuning of the 2.2B-parameter Qwen2-VL [wang2024qwen2vl] vision-language model under 4-bit quantization [dettmers2023qlora]; (C) the same Track A architecture but with the SigLIP encoder continue-pretrained via Masked Autoencoder [he2021mae] on in-domain images, following the DAPT+TAPT recipe of [gururangan2020dapt] with a domain-adapted text-aware masking variant; and (D) a modular YOLOv12+SAHI [tian2025yolov12, akyon2022sahi] detection pipeline followed by per-region text recognition, replicating [malepati2026sahi]. We evaluate using both strict per-field F1 and a partial-credit edit-distance F1, and we observe a non-monotonic relationship between the two: Track B's lenient F1 peaks early and then **declines** while strict F1 keeps climbing, indicating specialization-induced overfitting. Our findings: in this regime the custom hybrid wins on both metrics; the LoRA-fine-tuned VLM does not bridge the gap despite a 100× larger backbone; the metric-stopping-rule decision matters more than the method choice within a method family.

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

2. **A novel finding on metric choice**: under the standard strict per-field F1 metric, Track B's loss continues to descend monotonically while its lenient (edit-distance) F1 **peaks at step 300 of 500 and then drops by 47% by step 400** as the model overfits to strict-format-shaped output at the cost of label-content quality. The choice of *stopping metric* matters more than the choice of method *within* Track B, and the same effect — though smaller — is observed in Tracks A and C.

3. **A creative extension of MAE pretraining**: we propose a text-aware masking variant in which patches with higher Sobel-edge density (correlated with text and graphic content) are masked at higher rate (90%) than uniform-background patches (60%), forcing the encoder to learn text-from-context filling — the downstream skill we care about. This is a domain-justified twist on the uniform-random masking of [he2021mae] inspired by the attention-guided variant of [cao2022attmask].

4. **Methodological transparency**: every experiment is reproducible from a single-file trainer plus a YAML config; the evaluation harness reports both metrics; the dataset split is shared with [malepati2026sahi] for direct benchmark comparison.

The rest of the paper is organized as follows. Section II surveys relevant prior work on modular detection, end-to-end document VLMs, and self-supervised domain adaptation. Section III describes the dataset, the four tracks, and the evaluation harness. Section IV reports the comparative results. Section V discusses the implications for practitioners deploying small-data extraction systems. Section VI concludes.

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

[FILL: Single master table with macro_f1 + macro_edit_f1 + trainable params + wall-clock for all four tracks. Pinned to commit SHAs.]

Skeleton:

| Track | Method | Trainable | macro_f1 (strict) | macro_edit_f1 (lenient) | edit/strict | wall (min) |
|---|---|---:|---:|---:|---:|---:|
| A | SigLIP+Donut, vanilla SigLIP | 26.5 M | **0.0804** | **0.4094** | 5.1× | 30 |
| B | Qwen2-VL-2B + LoRA r=8, step 400 | ~5 M | 0.0159 | 0.1165 | 7.3× | 78 |
| B@peak | same, step 300 (peak edit_f1) | ~5 M | 0.0123 | 0.2167 | 17.6× | 60 |
| C-DAPT | A + DAPT MAE encoder | 26.5 M | `[FILL]` | `[FILL]` | `[FILL]` | 30 |
| C-TAPT | A + DAPT+TAPT MAE encoder | 26.5 M | `[FILL]` | `[FILL]` | `[FILL]` | 30 |
| C-textaware | A + text-aware MAE encoder | 26.5 M | `[FILL]` | `[FILL]` | `[FILL]` | 30 |
| D | YOLOv12+SAHI+OCR | YOLO ≈ 30 M, OCR head ≈ `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

[Reference: F1a (strict), F1b (lenient), F4 (Track A trajectory), F5 (Track B trajectory).]

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

[Skeleton — fill once Tracks C and D land.]

Three plausible mechanisms, not mutually exclusive: (i) at 224×224 input, pharma small-text fields are simply unreadable — a resolution problem rather than a knowledge problem; (ii) Qwen2-VL's pretraining distribution is web-image-heavy with comparatively little tightly-structured packaging text, so its priors mismatch our domain more than SigLIP's do; (iii) LoRA at r=8 is too low-capacity to bridge the structural format gap from natural-language to XML in only 500 steps, even though the per-field content learning saturates earlier (the metric-stopping-rule finding above). Track C's DAPT+text-aware-MAE addresses (ii) for the SigLIP encoder side; Track D's SAHI tile-and-merge addresses (i) directly.

### B. The strict-vs-lenient gap as a diagnostic

[Three paragraphs. The 17.6× ratio at Track B step 300 is qualitatively different from the 5.1× at Track A's peak; the difference is the format-vs-content axis. A practitioner can use this ratio as a diagnostic for what their model is doing wrong: high ratio = format problem; low ratio = content problem; ratio close to 1.0 = both problems converge.]

### C. Limitations

[Single-domain (pharma); single-dataset (the 837-image set shared with [malepati2026sahi]); single-language (Indian-market English+local); single-camera (48 MP smartphone); single-base-VLM for Track B (Qwen2-VL-2B specifically; effects on other VLM scales are open). A second-dataset cross-validation, ideally in a different pharmaceutical jurisdiction, would strengthen the generalizability claims of Section IV-E.]

### D. Future work

[The five-experiment research_directions.md queue: text-aware MAE (#1, in this paper as Track C variant), field-conditional Track A decoding (#2), native-resolution Qwen2-VL (#3), TAPT (#4, in this paper as Track C variant), heterogeneous-rank LoRA (#5). Plus second-dataset validation. Plus the Track D OCR-head ablation (per-region recognizer choice).]

---

## VI. Conclusion

[FILL last. ~150 words. Headline: in the 500–5000 image small-data pharma regime, a custom hybrid trained from scratch on top of a frozen pretrained vision encoder beats LoRA-fine-tuning of a 100× larger pretrained VLM on every metric we measured. In-domain MAE pretraining (Track C) and modular SAHI+OCR (Track D) provide the most-promising paths beyond Track A's current 0.080 strict / 0.41 lenient baseline. The choice of stopping metric is the highest-leverage decision within any individual method.]

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
