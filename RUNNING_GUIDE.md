# 🧠 ET-TACFN — Complete Running Guide
**Enhanced Trimodal Adaptive Cross-Modal Fusion Network for Emotion Recognition on IEMOCAP**

---

## ✅ What Was Fixed (Before You Run)

| # | File | Fix |
|---|------|-----|
| 1 | `preprocessing/extract_audio.py` | **Recreated** — old file was the visual extractor; now correctly extracts audio with Wav2Vec2-base |
| 2 | `models/et_tacfn_fusion.py` | Confidence gate now uses speech-enriched pooled features (not stale pre-fusion) |
| 3 | `models/hierarchical_fusion.py` | Added `.contiguous()` to `speech_seq.expand()` to fix memory layout bug |
| 4 | `models/missing_modality.py` | Modality dropout now checks actual model training state, not sub-module state |
| 5 | `models/classifier.py` | Classifier dropout raised to 0.35 + added `LayerNorm` for IEMOCAP's small dataset |
| 6 | `train.py` | Replaced `OneCycleLR` (incompatible with early stopping) with `ReduceLROnPlateau` |
| 7 | `evaluate.py` | Fixed `torch.load` PyTorch 2.x warning |
| 8 | `requirements.txt` | Removed unused `xgboost`, added `accelerate` |

---

## 🖥️ GPU Requirements

| Tier | GPU | VRAM | Training Time |
|------|-----|------|--------------|
| **Minimum** | GTX 1080 Ti | 11 GB | ~6–8 hrs |
| **Recommended** | RTX 3090 / 4080 | 24 GB | ~2–3 hrs |
| **Ideal** | A100 / H100 | 40–80 GB | ~45–90 min |
| **Cloud** | Colab Pro+ / Kaggle | 16–40 GB | ~2–4 hrs |

> **Minimum memory needed: ~6–8 GB VRAM** at `batch_size=16` with partial encoder fine-tuning.

---

## 🗂️ Expected Dataset Folder Structure

```
data/
└── raw/
    ├── Session1/
    │   ├── dialog/
    │   │   ├── EmoEvaluation/   ← emotion labels + timestamps (.txt)
    │   │   ├── transcriptions/  ← text transcripts (.txt)
    │   │   └── avi/DivX/        ← dialog-level video (.avi)
    │   └── sentences/
    │       └── wav/             ← utterance-level audio (.wav)
    ├── Session2/  ... (same structure)
    ├── Session3/
    ├── Session4/
    └── Session5/
```

> ⚠️ IEMOCAP is a licensed dataset. Request access at:
> **https://sail.usc.edu/iemocap/**

---

## ⚙️ Environment Setup

### Step 1 — Create Python Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ For CUDA support, install the right PyTorch for your GPU driver:
> ```bash
> # CUDA 11.8
> pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
>
> # CUDA 12.1
> pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
> ```
> Then install the rest: `pip install -r requirements.txt`

### Step 3 — Verify GPU is Detected

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## 📦 Preprocessing (Run Once Per Dataset)

Run these in order. Each step saves `.npy` feature files to `data/processed/`.

### Step 4A — Build Dataset Splits

```bash
python preprocessing\build_dataset.py
```

Creates:
- `data/splits/train_ids.txt`
- `data/splits/val_ids.txt`
- `data/splits/test_ids.txt`
- `data/splits/labels.txt`

> ⏱️ Fast — seconds only.

---

### Step 4B — Extract Text Features (RoBERTa-base)

```bash
python preprocessing\extract_text.py
```

- Downloads RoBERTa-base model (~500 MB, first time only)
- Reads transcripts from `dialog/transcriptions/`
- Saves: `data/processed/text_embeddings/<utt_id>.npy` → shape `[128, 768]`

> ⏱️ ~20–30 min on GPU

---

### Step 4C — Extract Audio Features (Wav2Vec2-base)

```bash
python preprocessing\extract_audio.py
```

- Downloads Wav2Vec2-base model (~360 MB, first time only)
- Reads `.wav` files from `sentences/wav/`
- Saves: `data/processed/audio_embeddings/<utt_id>.npy` → shape `[T_a, 768]`

> ⏱️ ~30–50 min on GPU

---

### Step 4D — Extract Visual Features (ResNet-50)

```bash
python preprocessing\extract_visual.py
```

- Uses pretrained ResNet-50 (torchvision, no download needed)
- Reads dialog AVI from `dialog/avi/DivX/`, clips per utterance timestamps
- Saves: `data/processed/visual_embeddings/<utt_id>.npy` → shape `[30, 256]`

> ⏱️ ~40–60 min on GPU

---

### Step 4E — Verify Data (Optional but Recommended)

```bash
python preprocessing\check_data.py
```

Reports how many utterances have all 3 modalities present per split.

---

## 🚀 Training

```bash
python train.py
```

**What happens:**
- Loads `config.yaml` settings
- Computes class weights to handle IEMOCAP imbalance (Happy/Angry are rare)
- Trains ET-TACFN for up to 40 epochs with early stopping (patience=7)
- `ReduceLROnPlateau` halves LR every 3 epochs of no val improvement
- Best checkpoint saved to: `checkpoints/best_model.pt`
- Training log saved to: `logs/training_log.csv`

**Expected output (rough targets):**
```
Epoch 05/40 | Train loss=1.10  acc=55.2% | Val loss=1.08  acc=57.1% | lr=0.000100
Epoch 15/40 | Train loss=0.82  acc=70.8% | Val loss=0.88  acc=68.3% | lr=0.000050
Epoch 30/40 | Train loss=0.61  acc=80.5% | Val loss=0.73  acc=76.9% | lr=0.000025
```

> 💡 **To adjust settings**, edit `config.yaml`:
> - `batch_size: 16` → increase to 32 if you have >16 GB VRAM
> - `finetune_layers: 2` → increase to 4–6 for more LM capacity (needs more RAM)
> - `epochs: 40` → increase to 60 for thorough convergence

---

## 📊 Evaluation

```bash
python evaluate.py
```

**Outputs to `logs/`:**
- `logs/test_results.txt` — WA, UA, Macro-F1, per-class breakdown
- `logs/confusion_matrix.png` — raw counts + normalized confusion matrix
- `logs/confidence_scores.png` — average modality trust per emotion

**Expected accuracy range (Session 5 test):**

| Metric | Target Range |
|--------|-------------|
| **WA (Weighted Accuracy)** | 78–85% |
| **UA (Unweighted Accuracy)** | 74–82% |
| **Macro F1** | 74–82% |

---

## 📈 Plot Training Curves

```bash
python plot_training.py
```

Reads `logs/training_log.csv` and generates a loss/accuracy plot.

---

## 🔧 Config Tuning for Higher Accuracy

Edit `config.yaml`:

```yaml
training:
  batch_size:       32       # ↑ if you have ≥16 GB VRAM
  epochs:           60       # ↑ for more convergence time
  lr:               0.0001
  encoder_lr:       0.000005
  patience:         10       # ↑ give more time before stopping
  finetune_layers:  4        # ↑ unfreeze more encoder layers

model:
  d_model:          512      # already large; don't go higher without >24GB VRAM
  dropout:          0.15     # can try 0.1–0.2
```

**Biggest accuracy boosts (in order of impact):**
1. Switch to `WavLM-Large` instead of `Wav2Vec2-base` in `extract_audio.py` → change model name to `"microsoft/wavlm-large"` (audio_input_dim → 1024 in config)
2. More fine-tuned encoder layers (`finetune_layers: 4–6`)
3. Larger batch size (if VRAM allows)

---

## 🗺️ File Map

```
Multimodal-Model-Training/
├── config.yaml                    ← all hyperparameters
├── train.py                       ← main training script
├── evaluate.py                    ← test set evaluation
├── plot_training.py               ← loss/acc curves
├── requirements.txt               ← pip dependencies
│
├── dataset/
│   └── iemocap_dataset.py         ← PyTorch Dataset class
│
├── models/
│   ├── classifier.py              ← full model wrapper (import this)
│   ├── et_tacfn_fusion.py         ← main fusion module
│   ├── hierarchical_fusion.py     ← Stage A+B hierarchical fusion
│   ├── cross_modal_attention.py   ← CMA block
│   ├── intra_modal_attention.py   ← intra-modal self-attention
│   ├── confidence_gate.py         ← confidence gating
│   ├── missing_modality.py        ← missing modality handling
│   └── adaptive_fusion.py         ← adaptive modality weighting
│
├── preprocessing/
│   ├── build_dataset.py           ← create splits + labels
│   ├── extract_text.py            ← RoBERTa-base text features
│   ├── extract_audio.py           ← Wav2Vec2-base audio features
│   ├── extract_visual.py          ← ResNet-50 visual features
│   ├── create_splits.py           ← session-based split logic
│   └── check_data.py              ← verify all .npy files exist
│
├── checkpoints/
│   └── best_model.pt              ← saved after training
└── logs/
    ├── training_log.csv
    ├── test_results.txt
    ├── confusion_matrix.png
    └── confidence_scores.png
```
