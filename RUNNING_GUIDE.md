# 🧠 ET-TACFN — Running Guide
**IEMOCAP 4-class Emotion Recognition · Target UA ≥ 80%**

---

## How the Three Tiers Work

Think of tiers as **progressive upgrades** — each builds on the previous:

| Tier | What it changes | How to activate |
|------|----------------|-----------------|
| **Tier 1** | Better loss function + config | Just run `python train.py` — already in code |
| **Tier 2** | Bigger, smarter encoders | Re-run 2 preprocessing scripts, then `python train.py` |
| **Tier 3** | Conversation context memory | Automatic — already enabled in `config.yaml` |

> **The best single run you can do:** complete Tier 2 re-extraction, then run `python train.py` once — you get **all three tiers active simultaneously.**

---

## ⚙️ First-Time Setup (do this once)

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify your GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

---

## 📦 Preprocessing (do this once per encoder version)

Run these **in order**. They create `.npy` feature files in `data/processed/`.

```powershell
# Step 1 — Build train/val/test split files (seconds)
python preprocessing\build_dataset.py

# Step 2 — Extract TEXT features  [~30–40 min, downloads ~1.4 GB roberta-large]
python preprocessing\extract_text.py

# Step 3 — Extract AUDIO features [~40–60 min, downloads ~700 MB wavlm-base-plus]
python preprocessing\extract_audio.py

# Step 4 — Extract VISUAL features [~40–60 min, uses local ResNet-50]
python preprocessing\extract_visual.py

# Step 5 — Sanity check (optional but recommended)
python preprocessing\check_data.py
```

**After Step 2 & 3, verify the shapes are correct (must be 1024):**
```powershell
python -c "
import numpy as np, glob
t = np.load(glob.glob('data/processed/text_embeddings/*.npy')[0])
a = np.load(glob.glob('data/processed/audio_embeddings/*.npy')[0])
print('Text :', t.shape)    # must end in 1024
print('Audio:', a.shape)    # must end in 1024
"
```

---

## 🚀 TIER 1 — Run Immediately (no re-extraction needed)

> Tier 1 changes (Focal Loss, Sad weight ×1.5, dropout, finetune_layers) are **already written into the code**. Nothing extra to configure.

```powershell
python train.py
python evaluate.py
```

**Expected result: UA ~67–69%**

---

## 🚀 TIER 2 — Upgrade Encoders, Then Train

> Tier 2 uses `roberta-large` (1024-d text) and `WavLM-Base+` (1024-d audio). Your old `.npy` files were 768-d and **must be deleted** before re-extracting — mixing them causes a shape crash.

```powershell
# Step 1 — Delete old 768-d .npy files
Remove-Item -Recurse -Force data\processed\text_embeddings\*
Remove-Item -Recurse -Force data\processed\audio_embeddings\*

# Step 2 — Re-extract with new encoders (same commands as preprocessing above)
python preprocessing\extract_text.py     # ~30–40 min
python preprocessing\extract_audio.py    # ~40–60 min

# Step 3 — Train (config already set to text_input_dim: 1024, audio_input_dim: 1024)
python train.py
python evaluate.py
```

**Expected result: UA ~74–79%**

---

## 🚀 TIER 3 — Already Active, No Extra Steps

> Tier 3 (BiGRU conversation context) is enabled by `use_conversation_context: true` in `config.yaml`. The dataset builds 5-utterance context windows automatically at load time from your existing `.npy` files.

**No extra commands.** Tier 3 runs automatically inside `train.py` whenever you train.

```powershell
# Tier 3 is already ON — just train as usual
python train.py
python evaluate.py
```

**Expected result: UA ~80–85% (when combined with Tier 2 encoders)**

---

## ✅ Recommended Full Run (all tiers together)

```powershell
# 1. Delete old embeddings (if you had 768-d ones)
Remove-Item -Recurse -Force data\processed\text_embeddings\*
Remove-Item -Recurse -Force data\processed\audio_embeddings\*

# 2. Re-extract (Tier 2 encoders)
python preprocessing\extract_text.py
python preprocessing\extract_audio.py
python preprocessing\extract_visual.py   # only if not done already

# 3. Train once — ALL THREE TIERS ARE ACTIVE
python train.py

# 4. Evaluate
python evaluate.py
```

---

## 📊 What to Expect

```
📂  Loading test set (Session 5)...
  [test] ✅  1241 utterances ready

🤖  Loading ET-TACFN from: checkpoints/best_model.pt
     Checkpoint: epoch XX, val_acc=XX%

=========================================================
  ET-TACFN TEST RESULTS — IEMOCAP Session 5
=========================================================
  Weighted Accuracy   (WA) : ~82–86%
  Unweighted Accuracy (UA) : ~80–85%
  Macro F1-Score           : ~81–85%
=========================================================
```

---

## 📁 Output Files

| File | What it contains |
|------|-----------------|
| `checkpoints/best_model.pt` | Best model weights |
| `logs/training_log.csv` | Loss & accuracy per epoch |
| `logs/test_results.txt` | WA / UA / F1 + per-class breakdown |
| `logs/confusion_matrix.png` | Confusion matrix (counts + normalized) |
| `logs/confidence_scores.png` | Modality gate scores (Text / Audio / Visual) |

---

## 🐛 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `mat1 and mat2 shapes cannot be multiplied` | Old 768-d `.npy` files with new 1024-d config | Delete old files, re-run `extract_text.py` and `extract_audio.py` |
| `num_samples = 0` | `.npy` files missing | Run `check_data.py` to see which files are absent |
| `CUDA out of memory` | Batch too large | Lower `batch_size` in `config.yaml` (try 8) |
| `ModuleNotFoundError` | venv not activated | `.\venv\Scripts\Activate.ps1` |

---

## 🗂️ Key Files

```
├── config.yaml               ← all settings (edit here)
├── train.py                  ← training  (Tier 1 + 3 already inside)
├── evaluate.py               ← evaluation
│
├── preprocessing\
│   ├── extract_text.py       ← RoBERTa-large  (Tier 2)
│   ├── extract_audio.py      ← WavLM-Base+    (Tier 2)
│   └── extract_visual.py     ← ResNet-50      (unchanged)
│
└── models\
    ├── classifier.py          ← full model entry point
    ├── conversation_context.py← BiGRU context  (Tier 3 — NEW)
    └── et_tacfn_fusion.py     ← core fusion logic
```
