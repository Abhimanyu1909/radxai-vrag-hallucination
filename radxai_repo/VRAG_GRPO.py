
import os
from pathlib import Path

SEED = 42

#Paths 
CACHE_HINT    = "your_own_path/RL/data/chexpert_cache"
DB_ZIP        = "your_own_path/RL/data/vrag_database/vrag_database"
TEST_VQA_PATH = "your_own_path/RL/data/vqa/test_vqa.jsonl"
VRAG_ADAPTER  = "your_own_path/RL/data/llava_vrag_adapter"
BASE_MODEL    = "llava-hf/llava-1.5-7b-hf"
OUT_ROOT = os.environ.get("OUT_ROOT", "your_own_path/RL/output")

# --- RL dataset ---
N_QUERIES        = 8000        # unique query images used for training
ENTS_PER_QUERY   = 4           # pos + neg entities per image (balanced)
TOP_K_RETRIEVAL  = 3           # cached top-k neighbours per query

# --- GRPO ---
G_ROLLOUTS       = 3          
EPS_CLIP         = 0.2         # PPO clip range
KL_BETA          = 0.005       # KL weight to reference (base + Vrag frozen)
LAMBDA_RETRIEVE = float(os.environ.get("LAMBDA_RETRIEVE", "0.10"))
LR               = 1e-5
BATCH_PROMPTS    = 1          
NUM_UPDATES      = 3000      
GRAD_ACCUM       = 2
MAX_NEW_DECISION = 4           # tokens sampled for the decision
MAX_NEW_ANSWER   = 6           # tokens sampled for the answer
SAMPLE_TEMP      = 1.0
SAVE_EVERY       = 200

# --- LoRA (the *new* trainable adapter on top of Vrag) ---
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Derived ---
RL_DATASET_PATH = f"{OUT_ROOT}/stage7_rl_dataset.jsonl"
ADAPTER_OUT     = f"{OUT_ROOT}/llava_vrag_grpo"
LOG_PATH        = f"{OUT_ROOT}/stage7_train.log"
Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)


# --- Sanity print ---
print("DB_ZIP =", repr(DB_ZIP))
print("exists:", Path(DB_ZIP).exists(), "is_dir:", Path(DB_ZIP).is_dir())
print("has parquet:", (Path(DB_ZIP) / "database.parquet").exists())

# ## 3. Imports and environment gate
import io, re, json, math, time, random, zipfile, glob, csv, logging, hashlib
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Stage 7 needs a GPU (V100 32GB or A100)."
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0))

# logging
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"),
              logging.StreamHandler()],
    level=logging.INFO,
    force=True,
)
log = logging.getLogger("stage7").info
log(f"config: G={G_ROLLOUTS}, LR={LR}, LAMBDA={LAMBDA_RETRIEVE}, KL_BETA={KL_BETA}")


# ## 4. Locate inputs
# --- Image cache ---
cands = [Path(CACHE_HINT)]
CACHE = next((c for c in cands if c.exists() and list(c.glob("images_part*"))), None)
assert CACHE is not None, f"chexpert_cache not found at {CACHE_HINT}"
log(f"cache: {CACHE}")

key2src = {}
for p in sorted(CACHE.glob("images_part*")):
    if p.is_dir():
        for png in p.rglob("*.png"):
            key2src[str(png.relative_to(p))] = ("file", str(png))
    elif zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            for n in zf.namelist():
                if n.endswith(".png"):
                    key2src[n] = ("zip", str(p))
log(f"images indexed: {len(key2src):,}")
assert key2src, "no PNGs found in the cache shards"

# --- vrag_database ---
p = Path(DB_ZIP)
if p.is_dir() and (p / "database.parquet").exists():
    DB_DIR = p
elif p.is_file() and str(p).endswith(".zip"):
    DB_DIR = Path(OUT_ROOT) / "vrag_database"
    with zipfile.ZipFile(p) as zf: zf.extractall(DB_DIR)
else:
    raise AssertionError(
        f"DB_ZIP is neither an extracted folder nor a .zip file:\n"
        f"  DB_ZIP  = {DB_ZIP}\n"
        f"  is_dir  = {p.is_dir()}\n"
        f"  is_file = {p.is_file()}"
    )
assert (DB_DIR / "database.parquet").exists() and (DB_DIR / "index.faiss").exists()
assert (DB_DIR / "embeddings.npy").exists(), \
    f"embeddings.npy missing in {DB_DIR} — this file is required for precomputed query embeddings"
log(f"database: {DB_DIR}")

# --- benchmark: patient blacklist ---
test_patients = set()
if Path(TEST_VQA_PATH).exists():
    for l in open(TEST_VQA_PATH):
        if not l.strip(): continue
        r = json.loads(l)
        p = r.get("deid_patient_id") or r.get("patient_id") or r.get("patient")
        if not p:
            for field in ("path", "image", "image_path", "png_key", "image_id"):
                v = r.get(field)
                if v:
                    first = str(v).lstrip("/").split("/")[0]
                    if first.startswith("patient"):
                        p = first; break
        if p: test_patients.add(str(p))
    log(f"test patients excluded: {len(test_patients):,}")
    assert len(test_patients) >= 500, (
        f"blacklist parser only found {len(test_patients)} patients."
    )

# --- Vrag adapter ---
assert Path(VRAG_ADAPTER).exists(), f"Vrag adapter missing at {VRAG_ADAPTER}"
assert (Path(VRAG_ADAPTER) / "adapter_config.json").exists(), \
    f"adapter_config.json missing in {VRAG_ADAPTER}"
log(f"Vrag adapter: {VRAG_ADAPTER}")


# ## 5. Image reader
_zip_cache = {}
def read_image(png_key: str) -> Image.Image:
    kind, src = key2src[png_key]
    if kind == "zip":
        zf = _zip_cache.get(src) or zipfile.ZipFile(src)
        _zip_cache[src] = zf
        img = Image.open(io.BytesIO(zf.read(png_key)))
    else:
        img = Image.open(src)
    return img.convert("RGB")


# ## 6. Build the RL dataset
db = pd.read_parquet(DB_DIR / "database.parquet")
db["deid_patient_id"] = db["deid_patient_id"].astype(str)
db = db[~db["deid_patient_id"].isin(test_patients)].reset_index(drop=True)
log(f"database rows (post-blacklist): {len(db):,}")

ent_map = {row.png_key: set(row.entities) for row in db.itertuples()}
vocab = sorted({e for s in ent_map.values() for e in s})
log(f"entity vocabulary size: {len(vocab)}")

SIBLINGS = [
    {"Lung Opacity","Consolidation","Pneumonia","Atelectasis","Edema"},
    {"Pleural Effusion","Pleural Other"},
    {"Cardiomegaly","Enlarged Cardiomediastinum"},
]
def sibling_group(e):
    for g in SIBLINGS:
        if e in g: return g
    return {e}

rng = random.Random(SEED)


# ### 6a. Pre-cache top-K retrievals per query (precomputed embeddings, no CheXzero)
import faiss

log("loading precomputed database embeddings from notebook 03 ...")
db_embeddings = np.load(str(DB_DIR / "embeddings.npy")).astype(np.float32)
CZ_DIM = db_embeddings.shape[1]
log(f"embeddings loaded: shape={db_embeddings.shape}")

# key -> row index lookup for O(1) query embedding retrieval
key_to_row = {k: i for i, k in enumerate(db["png_key"].values)}

index = faiss.read_index(str(DB_DIR / "index.faiss"))
db_keys    = db["png_key"].values
db_reports = db["report"].values
db_pats    = db["deid_patient_id"].values


# --- Select query images ---
query_keys = [k for k in db["png_key"].values if k in key2src]
rng.shuffle(query_keys)
query_keys = query_keys[:N_QUERIES]
log(f"queries to build: {len(query_keys):,}")

# Query embeddings: instant lookup from precomputed cache
from tqdm import tqdm
query_emb = np.stack([db_embeddings[key_to_row[k]] for k in query_keys]).astype(np.float32)
log(f"query embeddings from cache: shape={query_emb.shape}")

K = TOP_K_RETRIEVAL
sims, ids = index.search(query_emb, K + 2)
log(f"searched top-{K+2} for {len(query_keys):,} queries")


# ### 6b. Sample entities and write the RL JSONL
def sample_entities(present, k):
    present = sorted(present)
    forbidden = set().union(*(sibling_group(e) for e in present))
    neg_pool = [e for e in vocab if e not in forbidden]
    n_pos = min(len(present), k // 2)
    n_neg = k - n_pos
    pos = rng.sample(present, n_pos) if n_pos else []
    neg = rng.sample(neg_pool, min(n_neg, len(neg_pool))) if n_neg else []
    return [(e, "yes") for e in pos] + [(e, "no") for e in neg]

n_written = 0
with open(RL_DATASET_PATH, "w") as f:
    for qi, qkey in enumerate(tqdm(query_keys, desc="write rows")):
        present = ent_map.get(qkey, set())
        if not present: continue
        qpat = str(db.loc[db.png_key == qkey, "deid_patient_id"].iloc[0])
        refs = []
        for r in range(K + 2):
            nid = int(ids[qi, r])
            if db_pats[nid] == qpat: continue
            refs.append({"png_key": str(db_keys[nid]), "report": str(db_reports[nid])})
            if len(refs) == K: break
        if len(refs) < K: continue
        for entity, gt in sample_entities(present, ENTS_PER_QUERY):
            f.write(json.dumps({
                "query_key": qkey, "entity": entity, "gt": gt, "cached_refs": refs,
            }) + "\n")
            n_written += 1
log(f"wrote {n_written:,} rows -> {RL_DATASET_PATH}")


# ## 7. Load models — REFERENCE (base + Vrag frozen) + POLICY (base + trainable LoRA from Vrag)
from transformers import (
    LlavaForConditionalGeneration, LlavaProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

processor = LlavaProcessor.from_pretrained(BASE_MODEL)
tokenizer = processor.tokenizer
tokenizer.padding_side = "left"

# --- REFERENCE model: base + Vrag adapter (frozen) — correct KL target ---
log("loading REFERENCE (base + Vrag, frozen) ...")
ref_base = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL, quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map={"": 0},
)
ref_model = PeftModel.from_pretrained(
    ref_base, VRAG_ADAPTER, adapter_name="vrag", is_trainable=False,
)
for p in ref_model.parameters(): p.requires_grad_(False)
ref_model.eval()

# --- POLICY model: base + trainable LoRA initialised from Vrag ---
log("loading POLICY (base + trainable LoRA initialised from Vrag) ...")
policy_base = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL, quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map={"": 0},
)
policy_base = prepare_model_for_kbit_training(policy_base)
model = PeftModel.from_pretrained(
    policy_base, VRAG_ADAPTER, adapter_name="grpo", is_trainable=True,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"trainable parameters: {trainable/1e6:.1f} M")

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR, betas=(0.9, 0.95), weight_decay=0.01,
)


# ## 8. Prompt templates and parsers
ORD = ["1st", "2nd", "3rd", "4th", "5th"]

def decision_prompt(entity: str) -> str:
    return (
        "USER: <image>\n"
        f"Question: Does the patient have {entity}?\n"
        "Before answering, decide whether you need to see similar reference images.\n"
        "Reply with only ONE of: RETRIEVE or ANSWER_NOW.\n"
        "ASSISTANT:"
    )

def answer_prompt_no_refs(entity: str) -> str:
    return (
        "USER: <image>\n"
        "Answer with only the word yes or no. "
        f"Does the patient have {entity}?\n"
        "ASSISTANT:"
    )

def answer_prompt_with_refs(entity: str, refs) -> str:
    parts = ["USER: "]
    for i, r in enumerate(refs):
        parts.append(f"<image>\nThis is the {ORD[i]} similar image and its report "
                     f"for your reference. {r['report']}\n")
    parts.append("<image>\nAnswer the question with only the word yes or no. "
                 "Do not provide explanations. According to the last query image and the "
                 f"reference images and reports, does the patient have {entity}?\n"
                 "ASSISTANT:")
    return "".join(parts)

def parse_decision(text: str) -> str:
    t = text.strip().upper()
    if t.startswith("RETRIEVE"):   return "RETRIEVE"
    if t.startswith("ANSWER_NOW"): return "ANSWER_NOW"
    return "ANSWER_NOW"

def parse_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    head = t.split()[:3]
    if "yes" in head: return "yes"
    if "no"  in head: return "no"
    return "no"


# ## 9. Rollout — two-stage sampling with forced exploration
def _process(text, pil_imgs):
    return processor(text=text, images=pil_imgs, return_tensors="pt").to(device)

@torch.no_grad()
def sample_tokens(mdl, text, pil_imgs, max_new):
    inp = _process(text, pil_imgs)
    out = mdl.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=SAMPLE_TEMP,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_ids = out[0, inp.input_ids.shape[1]:]
    gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_ids.detach().cpu(), gen_txt, inp


def rollout(row):
    q_img  = read_image(row["query_key"])
    refs   = row["cached_refs"]
    entity = row["entity"]
    gt     = row["gt"]

    # --- Stage A: decision (with forced exploration for cold-start) ---
    FORCED_EXPLORE_STEPS = 400
    _step_counter = globals().get("_step_counter", 0)

    dtext = decision_prompt(entity)
    if _step_counter < FORCED_EXPLORE_STEPS and random.random() < 0.5:
        forced = random.choice(["RETRIEVE", "ANSWER_NOW"])
        dec_ids = tokenizer(forced, return_tensors="pt",
                            add_special_tokens=False).input_ids[0]
        dec_txt = forced
    else:
        dec_ids, dec_txt, _ = sample_tokens(model, dtext, [q_img], MAX_NEW_DECISION)
    globals()["_step_counter"] = _step_counter + 1

    decision  = parse_decision(dec_txt)
    retrieved = int(decision == "RETRIEVE")

    # --- Stage B: answer ---
    if retrieved:
        atext = answer_prompt_with_refs(entity, refs)
        pil_imgs = [read_image(r["png_key"]) for r in refs] + [q_img]
    else:
        atext = answer_prompt_no_refs(entity)
        pil_imgs = [q_img]
    ans_ids, ans_txt, _ = sample_tokens(model, atext, pil_imgs, MAX_NEW_ANSWER)
    answer  = parse_yes_no(ans_txt)
    correct = int(answer == gt)

    return {
        "decision_prompt":  dtext,
        "decision_pils":    [q_img],
        "decision_gen_ids": dec_ids,
        "answer_prompt":    atext,
        "answer_pils":      pil_imgs,
        "answer_gen_ids":   ans_ids,
        "retrieved":        retrieved,
        "answer":           answer,
        "correct":          correct,
    }


# ## 10. Reward
def compute_reward(rollout_out) -> float:
    return float(rollout_out["correct"]) - LAMBDA_RETRIEVE * float(rollout_out["retrieved"])


# ## 11. Log-probability and GRPO loss (KL via ref_model)
def logprob_of_gen(mdl, prompt_text, pil_imgs, gen_ids):
    full_text = prompt_text + tokenizer.decode(gen_ids, skip_special_tokens=True)
    inp = _process(full_text, pil_imgs)
    out = mdl(**inp)
    logits = out.logits[0]
    N = gen_ids.shape[0]
    pred = logits[-N-1:-1].float()
    logp = F.log_softmax(pred, dim=-1)
    return logp.gather(1, gen_ids.to(logits.device).unsqueeze(1)).squeeze(1)


def grpo_loss(group):
    rewards = torch.tensor([compute_reward(r) for r in group], device=device)
    A = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

    total_pg, total_kl, n_tokens = 0.0, 0.0, 0

    for i, r in enumerate(group):
        pieces = [
            (r["decision_prompt"], r["decision_pils"], r["decision_gen_ids"]),
            (r["answer_prompt"],   r["answer_pils"],   r["answer_gen_ids"]),
        ]
        for prompt_text, pils, gen_ids in pieces:
            if gen_ids.numel() == 0: continue

            new_logp = logprob_of_gen(model, prompt_text, pils, gen_ids)

            with torch.no_grad():
                old_logp = new_logp.detach()
                ref_logp = logprob_of_gen(ref_model, prompt_text, pils, gen_ids)

            ratio = torch.exp(new_logp - old_logp)
            clip  = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP)
            pg    = -torch.min(ratio * A[i], clip * A[i]).sum()
            kl    = (new_logp - ref_logp).sum()

            total_pg  = total_pg + pg
            total_kl  = total_kl + kl
            n_tokens += gen_ids.numel()

    torch.cuda.empty_cache()

    if n_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (total_pg + KL_BETA * total_kl) / n_tokens


# ## 12. Training loop
class RLDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

ds = RLDataset(RL_DATASET_PATH)
sampler = torch.utils.data.RandomSampler(ds, replacement=True,
                                         num_samples=NUM_UPDATES * BATCH_PROMPTS)
loader = DataLoader(ds, sampler=sampler, batch_size=BATCH_PROMPTS, collate_fn=lambda x: x)

running = {"reward": 0.0, "corr": 0.0, "ret": 0.0, "n": 0}
step = 0
t0 = time.time()
model.train()

for batch in loader:
    optimizer.zero_grad()
    step_loss = 0.0

    for row in batch:
        group = [rollout(row) for _ in range(G_ROLLOUTS)]
        loss  = grpo_loss(group)
        loss.backward()
        step_loss += float(loss.detach())

        for r in group:
            running["reward"] += compute_reward(r)
            running["corr"]   += r["correct"]
            running["ret"]    += r["retrieved"]
            running["n"]      += 1

    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
    if (step + 1) % GRAD_ACCUM == 0:
        optimizer.step()
        optimizer.zero_grad()

    step += 1
    if step % 20 == 0:
        n = max(running["n"], 1)
        log(f"step {step:5d} | loss {step_loss:+.3f} | "
            f"reward {running['reward']/n:+.3f} | acc {running['corr']/n:.3f} | "
            f"retrieve {running['ret']/n:.3f} | "
            f"speed {step/(time.time()-t0):.2f} step/s")
        running = {"reward": 0.0, "corr": 0.0, "ret": 0.0, "n": 0}

    if step % SAVE_EVERY == 0:
        ckpt = f"{ADAPTER_OUT}_step{step}"
        model.save_pretrained(ckpt, selected_adapters=["grpo"])
        log(f"checkpoint -> {ckpt}")

    if step >= NUM_UPDATES: break


# ## 13. Save the final adapter
model.save_pretrained(ADAPTER_OUT, selected_adapters=["grpo"])
tokenizer.save_pretrained(ADAPTER_OUT)
log(f"final GRPO adapter -> {ADAPTER_OUT}")
for f in sorted(Path(ADAPTER_OUT).iterdir()):
    log(f"  {f.name:40s} {f.stat().st_size/1e6:.2f} MB")


# ## 14. Quick sanity evaluation
model.eval()
holdout = ds.rows[-200:]
tally = {"corr":0, "ret":0, "corr_r":0, "n_r":0, "corr_nr":0, "n_nr":0}
for row in tqdm(holdout, desc="sanity eval"):
    r = rollout(row)
    tally["corr"] += r["correct"]; tally["ret"] += r["retrieved"]
    if r["retrieved"]:
        tally["corr_r"] += r["correct"]; tally["n_r"] += 1
    else:
        tally["corr_nr"] += r["correct"]; tally["n_nr"] += 1

n = len(holdout)
log(f"sanity: acc={tally['corr']/n:.3f}  retrieval_rate={tally['ret']/n:.3f}")
log(f"  when RETRIEVE   (n={tally['n_r']}): acc={tally['corr_r']/max(tally['n_r'],1):.3f}")
log(f"  when ANSWER_NOW (n={tally['n_nr']}): acc={tally['corr_nr']/max(tally['n_nr'],1):.3f}")