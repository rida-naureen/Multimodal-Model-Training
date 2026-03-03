# evaluate.py
# ============================================================
#  Final Evaluation Script
#
#  ⚠️  Run this ONLY ONCE — after training is complete.
#     Never use test results to change your model/settings.
#
#  What it does:
#    • Loads best_model.pt from checkpoints/
#    • Runs on test set (Session 5 only)
#    • Prints:
#        WA  = Weighted Accuracy   (standard metric)
#        UA  = Unweighted Accuracy (mean per-class, handles imbalance)
#        F1  = Macro F1-score
#        Per-class precision/recall/F1
#    • Saves confusion matrix → logs/confusion_matrix.png
#    • Saves results text    → logs/test_results.txt
#
#  Run:  python evaluate.py
# ============================================================

import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score, balanced_accuracy_score)
from tqdm import tqdm

from dataset.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.classifier import MultimodalEmotionModel

# ── Config ────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = os.path.join(cfg["paths"]["checkpoints"], "best_model.pt")
LOG_DIR   = cfg["paths"]["logs"]
os.makedirs(LOG_DIR, exist_ok=True)

EMOTION_NAMES = ["Happy", "Sad", "Angry", "Neutral"]

# ── Load test data ────────────────────────────────────────────
print("\n📂  Loading test set (Session 5)...")
SPLITS_DIR = cfg["data"]["splits_dir"]
TEXT_DIR   = os.path.join(cfg["data"]["processed_dir"], "text_embeddings")
AUDIO_DIR  = os.path.join(cfg["data"]["processed_dir"], "audio_embeddings")
VISUAL_DIR = os.path.join(cfg["data"]["processed_dir"], "visual_embeddings")

test_set = IEMOCAPDataset("test", SPLITS_DIR, TEXT_DIR, AUDIO_DIR, VISUAL_DIR)
test_loader = DataLoader(
    test_set,
    batch_size  = cfg["training"]["batch_size"],
    shuffle     = False,
    collate_fn  = collate_fn,
    num_workers = 0
)

# ── Load model ────────────────────────────────────────────────
print(f"\n🤖  Loading model from: {CKPT_PATH}")
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found at {CKPT_PATH}\n"
        f"Run 'python train.py' first!"
    )

checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model      = MultimodalEmotionModel(cfg).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"    Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(val_acc = {checkpoint['val_acc']*100:.2f}%)")

# ── Run inference on test set ─────────────────────────────────
print(f"\n⚙️   Running inference on {len(test_set)} test utterances...")
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="  Testing"):
        text   = batch["text"].to(DEVICE)
        audio  = batch["audio"].to(DEVICE)
        visual = batch["visual"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        t_mask = batch["text_mask"].to(DEVICE)
        a_mask = batch["audio_mask"].to(DEVICE)
        v_mask = batch["visual_mask"].to(DEVICE)

        logits, _ = model(text, audio, visual, t_mask, a_mask, v_mask)
        preds     = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Compute metrics ───────────────────────────────────────────
wa = accuracy_score(all_labels, all_preds)
ua = balanced_accuracy_score(all_labels, all_preds)   # mean per-class accuracy
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n" + "=" * 57)
print("  TEST RESULTS — IEMOCAP Session 5")
print("=" * 57)
print(f"  Weighted Accuracy   (WA) : {wa*100:.2f}%")
print(f"  Unweighted Accuracy (UA) : {ua*100:.2f}%")
print(f"  Macro F1-Score           : {f1*100:.2f}%")
print("=" * 57)
print("\n  Per-class breakdown:")
print(classification_report(
    all_labels, all_preds,
    target_names=EMOTION_NAMES,
    digits=4
))

# ── Confusion matrix ──────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
            ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
            ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")

plt.suptitle(
    f"IEMOCAP Test — WA: {wa*100:.2f}%   UA: {ua*100:.2f}%   F1: {f1*100:.2f}%",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
print(f"  📊  Confusion matrix saved → {cm_path}")

# ── Save text results ─────────────────────────────────────────
results_path = os.path.join(LOG_DIR, "test_results.txt")
with open(results_path, "w") as f:
    f.write("IEMOCAP Test Results (Session 5)\n")
    f.write("=" * 40 + "\n")
    f.write(f"WA  : {wa*100:.2f}%\n")
    f.write(f"UA  : {ua*100:.2f}%\n")
    f.write(f"F1  : {f1*100:.2f}%\n\n")
    f.write(classification_report(all_labels, all_preds,
                                  target_names=EMOTION_NAMES))
print(f"  📄  Results saved         → {results_path}\n")
