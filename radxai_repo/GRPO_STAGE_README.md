# RadXAI — Stage 7: GRPO Selective Retrieval

This stage trains a GRPO policy that learns *when* to invoke retrieval for chest
X-ray VQA, starting from the frozen LLaVA-Vrag adapter. It was originally
prototyped on Kaggle and later moved to a GPU cluster for full training runs.
This README covers setup on a fresh machine (cluster / local workstation).

## 1. Data and model artifacts

All inputs were originally hosted as Kaggle datasets and downloaded via the
Kaggle API. The four required artifacts are:

| Artifact | Kaggle source | Local folder |
|---|---|---|
| CheXpert image cache | `<anon>/cached-data-chexpert` | `chexpert_cache/` |
| V-RAG retrieval database (FAISS index + embeddings + parquet) | `<anon>/vrag-dataset-and-vrag-database` | `vrag_database/` |
| `test_vqa` benchmark (patient blacklist source) | `<anon>/1000-test-img-id-vqa-` | `vqa/test_vqa.jsonl` |
| V-RAG LoRA adapter | `<anon>/llava-vrag-adapter` | `llava_vrag_adapter/` |

### 1a. Kaggle API setup (on the new machine)

1. Generate a Kaggle API token: Kaggle account settings → **Create New Token**.
   This downloads a `kaggle.json` file.
2. Place it at `~/.kaggle/kaggle.json` and lock down permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/downloaded/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. The Kaggle CLI is already included in `environment.yml` (see Section 3), so
   no separate install step is needed once the conda environment is created.

### 1b. Download the datasets

Run from the project root (adjust `RL/data` to wherever you want the data to live):

```bash
mkdir -p RL/data && cd RL/data

kaggle datasets download -d <anon>/cached-data-chexpert -p . --unzip
kaggle datasets download -d <anon>/vrag-dataset-and-vrag-database -p . --unzip
kaggle datasets download -d <anon>/1000-test-img-id-vqa- -p . --unzip
kaggle datasets download -d <anon>/llava-vrag-adapter -p . --unzip
```

After unzipping, confirm the folder layout matches:

```
RL/data/
├── chexpert_cache/
│   └── images_part*/           # PNG shards or zip shards
├── vrag_database/
│   └── vrag_database/
│       ├── database.parquet
│       ├── embeddings.npy
│       └── index.faiss
├── vqa/
│   └── test_vqa.jsonl
└── llava_vrag_adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```


## 2. Configure paths for local / cluster execution

The notebook's config cell originally pointed at Kaggle's read-only mount paths
(`/kaggle/input/datasets/...`). When running outside Kaggle, replace these with
paths into your own `RL/data` folder:

```python
# --- Paths (edit to match your machine) ---
CACHE_HINT    = "your_own_path/RL/data/chexpert_cache"
DB_ZIP        = "your_own_path/RL/data/vrag_database/vrag_database"
TEST_VQA_PATH = "your_own_path/RL/data/vqa/test_vqa.jsonl"
VRAG_ADAPTER  = "your_own_path/RL/data/llava_vrag_adapter"
BASE_MODEL    = "llava-hf/llava-1.5-7b-hf"
OUT_ROOT = os.environ.get("OUT_ROOT", "your_own_path/RL/output")
```

Replace `your_own_path` with the absolute path to wherever `RL/` lives on the
new machine. `BASE_MODEL` stays as the HuggingFace Hub identifier and is
downloaded automatically on first run — no manual step needed for it.

`OUT_ROOT` can also be overridden per-run without touching the notebook, via:

```bash
export OUT_ROOT=/path/to/output_dir
```

## 3. Environment

The full environment — Python, PyTorch/CUDA, FAISS, and the pip-only packages
(`transformers`, `peft`, `accelerate`, `bitsandbytes`, `kaggle`) — is specified
in `environment.yml`. Create and activate it with:

```bash
conda env create -f environment.yml
conda activate radxai
```

This installs, at minimum:

- `torch` (CUDA 11.8 build)
- `transformers`, `peft`, `accelerate`, `bitsandbytes`
- `faiss-gpu` (swap to `faiss-cpu` in `environment.yml` if the cluster lacks a
  GPU-enabled FAISS build)
- `pandas`, `pyarrow`, `numpy`, `pillow`, `tqdm`
- `kaggle` (for the dataset download step in Section 1)



## 4. Run

Once paths are set (Section 2) and the environment is active (Section 3):

```bash
python VRAG_GRPO.py
```

or, if using SLURM, submit via your cluster's batch script pointing at this
script with the GPU partition requested, making sure the batch script activates
the `radxai` conda environment before launching Python, e.g.:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
source ~/miniconda3/etc/profile.d/conda.sh
conda activate radxai
python VRAG_GRPO.py
```



## 5. Sweeping over λ (retrieval penalty)

To reproduce the five parallel runs referenced in the paper
($\lambda \in \{0.00, 0.05, 0.10, 0.20, 0.30\}$), launch one process per value
with a distinct `OUT_ROOT`:

```bash
for L in 0.00 0.05 0.10 0.20 0.30; do
  OUT_ROOT="RL/output/L${L}" LAMBDA_RETRIEVE=${L} python VRAG_GRPO.py &
done
wait
```

Each run writes to its own `L<value>/` subfolder, matching the checkpoint path
convention already used in the logs (e.g. `output/L0.00/llava_vrag_grpo_step200`).