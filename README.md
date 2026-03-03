# IEMOCAP Cross-Modal Attention — Complete Guide

## What This Project Does

Detects emotions (Happy / Sad / Angry / Neutral) from:
- **Text** — what was said (RoBERTa)
- **Audio** — how it was said (Wav2Vec2)
- **Video** — facial expressions (ResNet-50)

Using **Cross-Modal Attention**: each modality learns from the others
to make a richer combined decision.

---

## Complete File Map

```
iemocap_cross_modal/
│
├── config.yaml                     ← ALL settings (edit this, not .py files)
├── requirements.txt                ← Python libraries to install
│
├── data/
│   ├── raw/                        ← PUT IEMOCAP SESSIONS HERE
│   │   ├── Session1/
│   │   ├── Session2/
│   │   ├── Session3/
│   │   ├── Session4/
│   │   └── Session5/
│   ├── processed/
│   │   ├── text_embeddings/        ← .npy files (auto-created)
│   │   ├── audio_embeddings/       ← .npy files (auto-created)
│   │   └── visual_embeddings/      ← .npy files (auto-created)
│   └── splits/
│       ├── train_ids.txt           ← (auto-created)
│       ├── val_ids.txt             ← (auto-created)
│       ├── test_ids.txt            ← (auto-created)
│       └── labels.txt             ← (auto-created)
│
├── preprocessing/
│   ├── check_data.py              ← STEP 1: verify dataset
│   ├── create_splits.py           ← STEP 2: make splits
│   ├── extract_text.py            ← STEP 3A: RoBERTa features
│   ├── extract_audio.py           ← STEP 3B: Wav2Vec2 features
│   ├── extract_visual.py          ← STEP 3C: ResNet features
│   └── build_dataset.py           ← runs all 5 steps above at once
│
├── models/
│   ├── cross_modal_attention.py   ← Core Q/K/V attention module
│   ├── adaptive_fusion.py         ← Combines all 3 modalities
│   └── classifier.py              ← Full model (fusion + MLP)
│
├── dataset/
│   └── iemocap_dataset.py         ← PyTorch data loader
│
├── train.py                       ← STEP 4: train the model
├── evaluate.py                    ← STEP 5: test on Session 5
├── plot_training.py               ← visualize training curves
│
├── checkpoints/                   ← best_model.pt saved here
└── logs/                          ← training_log.csv, confusion_matrix.png
```

---

## Step-by-Step Instructions

### Setup (one time)
```cmd
cd Desktop\iemocap_cross_modal
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Place your data
```
Copy IEMOCAP folders into:   data\raw\Session1\  ...  data\raw\Session5\
```

### Run preprocessing (when server is accessible)
```cmd
python preprocessing\check_data.py       ← verify dataset OK
python preprocessing\create_splits.py    ← make train/val/test lists
python preprocessing\extract_text.py     ← ~20-30 min
python preprocessing\extract_audio.py    ← ~30-40 min
python preprocessing\extract_visual.py   ← ~40-60 min

# OR run all at once:
python preprocessing\build_dataset.py
```

### Train
```cmd
python train.py
```

### Visualize training
```cmd
python plot_training.py
```

### Evaluate (once, final test)
```cmd
python evaluate.py
```

---

## Data Split Strategy

| Split | Sessions      | Size   | Purpose                    |
|-------|---------------|--------|----------------------------|
| Train | 1, 2, 3, 4 (90%) | ~3953 | Learn model weights        |
| Val   | 1, 2, 3, 4 (10%) | ~439  | Early stopping, tune       |
| Test  | 5 (100%)        | ~1139 | Final evaluation (once!)   |

**LOSO** = Leave-One-Session-Out. Session 5 is always held out.
Never tune based on test results.

---

## How Cross-Modal Attention Works

```
"I'm fine."  ← TEXT says positive
[shaky voice] ← AUDIO says negative
[sad face]    ← VISUAL says negative

Text  asks Audio:  "Is the vocal tone relevant to this word?"
Text  asks Visual: "Does the face match the sentiment?"
Audio asks Text:   "Does the transcript clarify this vocal pattern?"

→ All 3 combined → correctly classified as SAD
```

---

## Expected Results

| Metric | Expected Range |
|--------|----------------|
| WA     | 65 – 72%       |
| UA     | 62 – 70%       |
| F1     | 63 – 70%       |
