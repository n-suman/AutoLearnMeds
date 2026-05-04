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

## Notes

- All PDFs except `karpathy_2026_autoresearch.md` (repo, not paper) and `hunter2007matplotlib` (IEEE DOI, paywalled — cited from BibTeX only) are stored in this folder.
- File integrity: each PDF was verified as a valid PDF document at download time (`file -b` returned `PDF document, version 1.x`).
- All sizes between 135KB and 6.8MB — no error-page substitutes.
- The agent will append rows to this table whenever it cites a new paper during autoresearch.
- Reproducibility: the download script lives at `scripts/download_papers.sh` (will be created in Phase 0).
