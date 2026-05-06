# Papers — Citation Registry

This is the canonical citation registry for the AutoLearnMeds project. Every paper cited anywhere in the design, training script, ledger, or final paper MUST appear here. CI fails if a `notes.md` cites a key not in this table.

## How to add a new paper

1. Drop the PDF (or markdown summary) into this folder. Filename: `<lastname>_<year>_<short_title>.pdf`.
2. Add a row to the table below.
3. Cite key format: `<lastname><year><lowercaseshorttitle>` (no spaces, no punctuation).
4. Run `python scripts/build_bibtex.py` to regenerate `paper/refs.bib`.

## Citation table

| Cite key | Authors | Year | Title | File | arXiv | Used for |
|---|---|---|---|---|---|---|
| lipton2014f1 | Lipton, Elkan, Naryshkin | 2014 | Optimal Thresholding of Classifiers to Maximize F1 Measure | `lipton_2014_f1.pdf` | [1402.1892](https://arxiv.org/abs/1402.1892) | §1.5 — justifies macro-F1 as primary metric |
| kim2022donut | Kim, Hong, Yim, Nam, Park, Yim, Hwang, Yun, Han, Park | 2022 | OCR-free Document Understanding Transformer (Donut) | `kim_2022_donut.pdf` | [2111.15664](https://arxiv.org/abs/2111.15664) | §3.4 (output format), §4.2 (decoder pattern) |
| cubuk2020randaugment | Cubuk, Zoph, Shlens, Le | 2020 | RandAugment: Practical Automated Data Augmentation | `cubuk_2020_randaugment.pdf` | [1909.13719](https://arxiv.org/abs/1909.13719) | §3.5 — allowed augmentation example |
| hendrycks2020augmix | Hendrycks, Mu, Cubuk, Zoph, Gilmer, Lakshminarayanan | 2020 | AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty | `hendrycks_2020_augmix.pdf` | [1912.02781](https://arxiv.org/abs/1912.02781) | §3.5 — allowed augmentation example |
| zhai2023siglip | Zhai, Mustafa, Kolesnikov, Beyer | 2023 | Sigmoid Loss for Language Image Pre-Training (SigLIP) | `zhai_2023_siglip.pdf` | [2303.15343](https://arxiv.org/abs/2303.15343) | §4.1 — frozen vision encoder |
| vaswani2017attention | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin | 2017 | Attention Is All You Need | `vaswani_2017_attention.pdf` | [1706.03762](https://arxiv.org/abs/1706.03762) | §4.2 — transformer decoder defaults |
| hendrycks2016gelu | Hendrycks, Gimpel | 2016 | Gaussian Error Linear Units (GELUs) | `hendrycks_2016_gelu.pdf` | [1606.08415](https://arxiv.org/abs/1606.08415) | §4.2 — activation |
| su2021roformer | Su, Lu, Pan, Murtadha, Wen, Liu | 2021 | RoFormer: Enhanced Transformer with Rotary Position Embedding | `su_2021_roformer.pdf` | [2104.09864](https://arxiv.org/abs/2104.09864) | §4.2 — RoPE positional encoding |
| press2017tied | Press, Wolf | 2017 | Using the Output Embedding to Improve Language Models | `press_2017_tied_embeddings.pdf` | [1608.05859](https://arxiv.org/abs/1608.05859) | §4.2 — tied input/output embeddings |
| szegedy2016labelsmoothing | Szegedy, Vanhoucke, Ioffe, Shlens, Wojna | 2016 | Rethinking the Inception Architecture for Computer Vision | `szegedy_2016_label_smoothing.pdf` | [1512.00567](https://arxiv.org/abs/1512.00567) | §4.4 — label smoothing |
| loshchilov2019adamw | Loshchilov, Hutter | 2019 | Decoupled Weight Decay Regularization (AdamW) | `loshchilov_2019_adamw.pdf` | [1711.05101](https://arxiv.org/abs/1711.05101) | §4.5 — optimizer |
| press2022alibi | Press, Smith, Lewis | 2022 | Train Short, Test Long: Attention with Linear Biases (ALiBi) | `press_2022_alibi.pdf` | [2108.12409](https://arxiv.org/abs/2108.12409) | §5.4 — example agent variation |
| karpathy2026autoresearch | Karpathy | 2026 | autoresearch (GitHub repo + tweets) | `karpathy_2026_autoresearch.md` | n/a (repo) | §5 — autoresearch methodology |
| mitchell2019modelcards | Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji, Gebru | 2019 | Model Cards for Model Reporting | `mitchell_2019_model_cards.pdf` | [1810.03993](https://arxiv.org/abs/1810.03993) | §8.6 — model card template |
| gebru2018datasheets | Gebru, Morgenstern, Vecchione, Vaughan, Wallach, Daumé III, Crawford | 2018 | Datasheets for Datasets | `gebru_2018_datasheets.pdf` | [1803.09010](https://arxiv.org/abs/1803.09010) | §3.7, §8.6 — data card template |
| hunter2007matplotlib | Hunter | 2007 | Matplotlib: A 2D Graphics Environment | n/a (paywalled DOI) | n/a (DOI: 10.1109/MCSE.2007.55) | §8.3 — figure tooling citation |
| malepati2026sahi | Malepati, Nandamury, Manjunath, Rajan, Prabhune | 2026 | Comparative Evaluation of YOLOv12 and SAHI for Medication Identification in Hospital Pharmacies | `Comparative_Evaluation_of_YOLOv12_and_SAHI_for_Medication_Identification_in_Hospital_Pharmacies.pdf` | n/a (DOI: 10.1109/IITCEE67948.2026.11394638) | §2 (related work, same dataset), §6 (Track D = SAHI+OCR pipeline as competitor track) |
| akyon2022sahi | Akyon, Altinuc, Temizel | 2022 | Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection | n/a (paywalled DOI) | n/a (DOI: 10.1109/ICIP46576.2022.9897990) | §2 (related work, SAHI primary reference), §6 (Track D inference recipe) |
| tian2025yolov12 | Tian, Ye, Doermann | 2025 | YOLOv12: Attention-Centric Real-Time Object Detectors | n/a | [2502.12524](https://arxiv.org/abs/2502.12524) | §6 (Track D detector backbone, Phase 8) |
| dosovitskiy2020vit | Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby | 2020 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | n/a | [2010.11929](https://arxiv.org/abs/2010.11929) | §4 (Track A frozen vision encoder ViT architecture), §7 (MAE encoder + decoder ViT architecture) |
| devlin2018bert | Devlin, Chang, Lee, Toutanova | 2018 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | n/a | [1810.04805](https://arxiv.org/abs/1810.04805) | §7 (origin of masked-prediction self-supervision; MAE adapts the idea to images) |
| he2021mae | He, Chen, Xie, Li, Dollár, Girshick | 2021 | Masked Autoencoders Are Scalable Vision Learners | n/a | [2111.06377](https://arxiv.org/abs/2111.06377) | §7 (Phase 7 MAE pretraining — primary method; mask ratio 0.75, asymmetric encoder-decoder, pixel-MSE per masked patch, per-patch normalization), §7b (text-aware masking — creative twist on uniform-random) |
| gururangan2020dapt | Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith | 2020 | Don't Stop Pretraining: Adapt Language Models to Domains and Tasks | n/a | [2004.10964](https://arxiv.org/abs/2004.10964) | §7 (DAPT recipe — load-bearing for our entire Track C strategy of in-domain continue-pretraining), §7b (TAPT — creative reuse #4) |
| bao2021beit | Bao, Dong, Piao, Wei | 2021 | BEiT: BERT Pre-Training of Image Transformers | n/a | [2106.08254](https://arxiv.org/abs/2106.08254) | §7 (alternative MIM method, considered and not chosen — discrete VQ-VAE token reconstruction instead of pixels; lower-leverage queue #6) |
| xie2022simmim | Xie, Zhang, Cao, Lin, Bao, Yao, Dai, Hu | 2022 | SimMIM: A Simple Framework for Masked Image Modeling | n/a | [2111.09886](https://arxiv.org/abs/2111.09886) | §7 (alternative MIM method, considered and not chosen — simpler linear-projection decoder; lower-leverage queue #7) |
| cao2022attmask | Cao, Xu, Clifton | 2022 | AttMask: Attention-Guided Masked Image Modeling | n/a | [2203.12719](https://arxiv.org/abs/2203.12719) | §7b (related work to text-aware masking — uses attention-rollout for mask weighting; our edge-density approach is a simpler precursor) |
| wang2024qwen2vl | Wang, Bai, Tan, Wang, Liu, Zhou, Lin, Yang, Hou, Lin, et al. | 2024 | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution | n/a | [2409.12191](https://arxiv.org/abs/2409.12191) | §5 (Track B base architecture; Naive Dynamic Resolution discussed in research_directions.md #3) |
| hu2021lora | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen | 2021 | LoRA: Low-Rank Adaptation of Large Language Models | n/a | [2106.09685](https://arxiv.org/abs/2106.09685) | §5 (Track B fine-tuning method, r=8 on q/k/v/o_proj) |
| dettmers2023qlora | Dettmers, Pagnoni, Holtzman, Zettlemoyer | 2023 | QLoRA: Efficient Finetuning of Quantized LLMs | n/a | [2305.14314](https://arxiv.org/abs/2305.14314) | §5 (Track B 4-bit nf4 + LoRA recipe; bnb_4bit_compute_dtype=bfloat16, double-quant) |
| sang2003conll | Sang, De Meulder | 2003 | Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition | n/a | n/a (CoNLL-2003 proceedings) | §1.5 (canonical per-entity-type F1 reporting convention from NER literature), §6 (per-field comparison methodology) |
| levenshtein1966 | Levenshtein | 1966 | Binary codes capable of correcting deletions, insertions, and reversals | n/a | n/a (Soviet Physics Doklady, vol. 10, no. 8) | §1.5 (basis of partial-credit `macro_edit_f1` lenient metric) |

## Notes

- All PDFs except `karpathy_2026_autoresearch.md` (repo, not paper) and `hunter2007matplotlib` (IEEE DOI, paywalled — cited from BibTeX only) are stored in this folder.
- File integrity: each PDF was verified as a valid PDF document at download time (`file -b` returned `PDF document, version 1.x`).
- All sizes between 135KB and 6.8MB — no error-page substitutes.
- The agent will append rows to this table whenever it cites a new paper during autoresearch.
- Reproducibility: the download script lives at `scripts/download_papers.sh` (will be created in Phase 0).
