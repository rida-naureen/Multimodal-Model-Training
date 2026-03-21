# 🧠 ET-TACFN — Running Guide
**IEMOCAP 4-class Emotion Recognition (Happy / Sad / Angry / Neutral)**

---

## ⚙️ First-Time Setup

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 📦 Step 1 — Preprocessing (run once)

> These scripts create `.npy` feature files in `data/processed/`.  
> Run **in order**. Each step must complete before the next.

```powershell
# Build train / val / test split files  (a few seconds)
python preprocessing\build_dataset.py

# Extract TEXT embeddings  — RoBERTa-base  (~20–30 min)
python preprocessing\extract_text.py

# Extract AUDIO embeddings — wav2vec2-base (~30–40 min)
python preprocessing\extract_audio.py

# Extract VISUAL embeddings — ResNet-50    (~30–40 min)
python preprocessing\extract_visual.py

# Optional: sanity check all files are present
python preprocessing\check_data.py
```

**Verify shapes after extraction:**
```powershell
python -c "
import numpy as np, glob
t = np.load(glob.glob('data/processed/text_embeddings/*.npy')[0])
a = np.load(glob.glob('data/processed/audio_embeddings/*.npy')[0])
v = np.load(glob.glob('data/processed/visual_embeddings/*.npy')[0])
print('Text  :', t.shape)    # expect (..., 1024)
print('Audio :', a.shape)    # expect (..., 1024)
print('Visual:', v.shape)    # expect (30, 256)
"
```

---

## 🚀 Step 2 — Train

```powershell
python train.py
```

**What to expect during training:**
```
🖥️   Device   : cuda
     GPU      : NVIDIA ...
     VRAM     : X.X GB

📂  Loading datasets...
  [train] ✅  XXXX utterances ready
  [val]   ✅  XXX  utterances ready

🤖  Building ET-TACFN model...
     Trainable parameters: X,XXX,XXX

🚀  Training ET-TACFN for up to 40 epochs

Epoch 01/40  |  Train  loss=X.XXXX  acc=XX.X%  |  Val  loss=X.XXXX  acc=XX.X%
  ✅  Best model saved  (val_acc=XX.XX%)
...
```

The best model is saved to `checkpoints/best_model.pt` automatically.

---

## 📊 Step 3 — Evaluate

```powershell
python evaluate.py
```

**Expected output:**
```
=========================================================
  ET-TACFN TEST RESULTS — IEMOCAP Session 5
=========================================================
  Weighted Accuracy   (WA) : XX.XX%
  Unweighted Accuracy (UA) : XX.XX%
  Macro F1-Score           : XX.XX%
=========================================================

  Per-class breakdown:
              precision    recall  f1-score   support
       Happy     X.XXXX    X.XXXX    X.XXXX      XXX
         Sad     X.XXXX    X.XXXX    X.XXXX      XXX
       Angry     X.XXXX    X.XXXX    X.XXXX      XXX
     Neutral     X.XXXX    X.XXXX    X.XXXX      XXX
```

---

## 📁 Output Files

| File | Contents |
|------|----------|
| `checkpoints/best_model.pt` | Best model weights |
| `logs/training_log.csv` | Loss & accuracy per epoch |
| `logs/test_results.txt` | WA / UA / F1 + per-class report |
| `logs/confusion_matrix.png` | Confusion matrix (counts + normalized) |
| `logs/confidence_scores.png` | Modality confidence gates (Text/Audio/Visual) |

---

## ⚙️ Key Settings (`config.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `text_input_dim` | `1024` | Must match your text `.npy` files (roberta-large output) |
| `audio_input_dim` | `1024` | Must match your audio `.npy` files (wavlm-base-plus output) |
| `visual_input_dim` | `256` | ResNet-50 → Linear(2048→256), do not change |
| `d_model` | `512` | Shared fusion dimension |
| `batch_size` | `16` | Lower to `8` if CUDA OOM |
| `epochs` | `40` | Max training epochs |
| `patience` | `7` | Early stopping patience |
| `lr` | `0.0001` | Learning rate |
| `modality_dropout` | `0.15` | Fraction of batches with a random modality dropped |

---

## 🗂️ Project Structure

```
ET-TACFN/
├── config.yaml                 ← All settings (edit here)
├── train.py                    ← Training script
├── evaluate.py                 ← Evaluation script
│
├── preprocessing/
│   ├── build_dataset.py        ← Creates split ID files
│   ├── extract_text.py         ← RoBERTa-base text features
│   ├── extract_audio.py        ← wav2vec2-base audio features
│   ├── extract_visual.py       ← ResNet-50 visual features
│   └── check_data.py           ← Verifies all .npy files exist
│
├── models/
│   ├── classifier.py           ← Top-level model (import this)
│   ├── et_tacfn_fusion.py      ← Core fusion module
│   ├── intra_modal_attention.py
│   ├── hierarchical_fusion.py
│   ├── confidence_gate.py
│   ├── missing_modality.py
│   └── cross_modal_attention.py
│
├── dataset/
│   └── iemocap_dataset.py      ← PyTorch Dataset + collate_fn
│
├── data/
│   ├── raw/                    ← Original IEMOCAP audio/video/text files
│   ├── splits/                 ← train_ids.txt, val_ids.txt, test_ids.txt, labels.txt
│   └── processed/
│       ├── text_embeddings/    ← .npy files per utterance
│       ├── audio_embeddings/
│       └── visual_embeddings/
│
├── checkpoints/                ← Saved model .pt files
└── logs/                       ← Training CSV + evaluation plots
```

---

## 🐛 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `mat1 and mat2 shapes cannot be multiplied` | `audio_input_dim` in config doesn't match actual `.npy` dim | Check shape with verify command above, update `config.yaml` |
| `FileNotFoundError: best_model.pt` | Haven't trained yet | Run `python train.py` first |
| `Training set is empty` | `.npy` files missing | Re-run preprocessing steps |
| `CUDA out of memory` | Batch too large | Set `batch_size: 8` in `config.yaml` |
| `ModuleNotFoundError` | venv not activated | Run `.\venv\Scripts\Activate.ps1` |
| `[split] ⚠️ Skipped N utterances` | Some `.npy` files missing | Run `python preprocessing\check_data.py` |
