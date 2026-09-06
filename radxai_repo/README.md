# RadXAI: Probing and Mitigating Hallucination in Radiology VLMs

**Status:** Submitted to AAAI 2027 — currently under double-blind review.
This repository is anonymized in accordance with the review process. Author
names, affiliations, and personally identifying dataset/account references
have been removed or replaced with placeholders. A de-anonymized version
with full attribution will be published after the review outcome.

## Overview

RadXAI grounds medical Vision-Language Model (VLM) outputs in retrieved
visual evidence to reduce hallucination in radiology report generation and
visual question answering (VQA). We reproduce and extend Visual
Retrieval-Augmented Generation (V-RAG) on the CheXpert Plus corpus with
LLaVA-1.5-7B as the backbone, and measure hallucination directly through a
controlled entity-probing benchmark (`test_vqa`) rather than text-similarity
metrics alone.

Key results:
- V-RAG finetuning + retrieval raises finding-level precision from 0.549 to
  0.844 and MCC from +0.159 to +0.472 on CheXpert Plus, corresponding to a
  hallucination-rate reduction from 45.1% to 15.6%.
- The ordering **no-retrieval < random-retrieval < V-RAG-retrieval** holds
  consistently across CheXpert Plus, MIMIC-CXR, and IU X-Ray.
- A GRPO-based reinforcement learning stage explores whether the model can
  learn *when* retrieval is worth its cost; see the "GRPO Selective
  Retrieval" findings in the paper and `VRAG_GRPO.py`.

See [`paper.pdf`](./paper.pdf) for the full writeup, including methodology,
ablations, and related work.

## Repository Structure

```
.
├── paper.pdf                          # Full paper (AAAI 2027 submission)
├── 01-chexpert-cache-builder.ipynb    # Build local image cache from CheXpert Plus
├── 02-encoder-comparison.ipynb        # CheXzero vs. alternative retrieval encoder
├── 04-llava-s-finetuning.ipynb        # Stage 2: domain-adapted LLaVA (llava_s) via LoRA SFT
├── 05-vrag-trainingf.ipynb            # Stage 3: V-RAG finetuning (llava_vrag) via LoRA
├── 06-vrag-evaluation-notebook.ipynb  # test_vqa benchmark evaluation across 3 retrieval modes
├── VRAG_GRPO.py                       # Stage 4: GRPO training loop for selective retrieval
├── GRPO_STAGE_README.md               # Detailed setup/run instructions for the GRPO stage
├── environment.yml                    # Conda environment spec
└── .gitignore
```

## Pipeline Stages

1. **Data preprocessing** (`01-chexpert-cache-builder.ipynb`) — build a local
   image cache and structured metadata from CheXpert Plus, then extract and
   normalize clinical entities (Qwen2.5-14B-Instruct) to construct balanced
   VQA pairs.
2. **Encoder selection** (`02-encoder-comparison.ipynb`) — compare CheXzero
   against an alternative CXR encoder for retrieval quality.
3. **Domain adaptation** (`04-llava-s-finetuning.ipynb`) — LoRA fine-tune
   LLaVA-1.5-7B on CheXpert Plus (image, report) pairs → `llava_s`.
4. **V-RAG fine-tuning** (`05-vrag-trainingf.ipynb`) — further LoRA fine-tune
   `llava_s` on retrieval-oriented multi-image tasks → `llava_vrag`.
5. **Evaluation** (`06-vrag-evaluation-notebook.ipynb`) — run all three models
   (`llava`, `llava_s`, `llava_vrag`) under three retrieval conditions (off,
   V-RAG, random) on the `test_vqa` benchmark, reporting Accuracy, Precision,
   Recall, F1, and MCC.
6. **GRPO selective retrieval** (`VRAG_GRPO.py`, see `GRPO_STAGE_README.md`)
   — train a policy to decide *when* to retrieve, using Group Relative Policy
   Optimization with a correctness-minus-retrieval-penalty reward.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate radxai
```

This installs PyTorch (CUDA 11.8), `transformers`, `peft`, `accelerate`,
`bitsandbytes`, FAISS, and supporting libraries. See `GRPO_STAGE_README.md`
for details on swapping `faiss-gpu` for `faiss-cpu` on non-GPU machines.

## Data Availability

The datasets used (CheXpert Plus, MIMIC-CXR, IU X-Ray) are third-party
clinical imaging datasets with their own access/credentialing requirements
and are **not redistributed in this repository**. Notebooks reference
placeholder paths — configure them to point at your own local copies once
you have the appropriate dataset access. See `GRPO_STAGE_README.md` for the
expected local folder layout for the GRPO stage specifically.

## Citation

This work is currently under review at AAAI 2027. A citation entry will be
added here once the paper is accepted/published.

## License

See [`LICENSE`](./LICENSE).
