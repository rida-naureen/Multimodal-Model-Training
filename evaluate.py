# evaluate.py
# ============================================================
#  ET-TACFN Final Evaluation
#
#  ⚠️  Run ONCE after training is complete.
#
#  Outputs:
#    • WA, UA, Macro F1
#    • Per-class precision / recall / F1
#    • Confusion matrix PNG
#    • Confidence scores per modality (ET-TACFN analysis)
#    • All saved to logs/
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

test_set = IEMOCAPDataset("test", SPLITS_DIR, TEXT_DIR, AUDIO_DIR, VISUAL_DIR, cfg=cfg)
test_loader = DataLoader(
    test_set, batch_size=cfg["training"]["batch_size"],
    shuffle=False, collate_fn=collate_fn, num_workers=0)

# ── Load model ────────────────────────────────────────────────
print(f"\n🤖  Loading ET-TACFN from: {CKPT_PATH}")
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)  # cfg dict needs full unpickling
model      = MultimodalEmotionModel(cfg).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"     Checkpoint: epoch {checkpoint['epoch']}, "
      f"val_acc={checkpoint['val_acc']*100:.2f}%")

# ── Inference ─────────────────────────────────────────────────
print(f"\n⚙️   Running inference on {len(test_set)} utterances...")
all_preds, all_labels = [], []
all_conf_t, all_conf_a, all_conf_v = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="  Testing"):
        text   = batch["text"].to(DEVICE)
        audio  = batch["audio"].to(DEVICE)
        visual = batch["visual"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        t_mask = batch["text_mask"].to(DEVICE)
        a_mask = batch["audio_mask"].to(DEVICE)
        v_mask = batch["visual_mask"].to(DEVICE)

        # Tier 3: conversation context window (optional)
        ctx_window = batch.get("context_window", None)
        if ctx_window is not None:
            ctx_window = ctx_window.to(DEVICE)

        logits, info = model(text, audio, visual, t_mask, a_mask, v_mask,
                             context_window=ctx_window)
        preds        = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Collect confidence scores (ET-TACFN analysis)
        if "text_conf" in info:
            all_conf_t.extend(info["text_conf"].squeeze(-1).numpy())
            all_conf_a.extend(info["audio_conf"].squeeze(-1).numpy())
            all_conf_v.extend(info["visual_conf"].squeeze(-1).numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ───────────────────────────────────────────────────
wa = accuracy_score(all_labels, all_preds)
ua = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n" + "=" * 57)
print("  ET-TACFN TEST RESULTS — IEMOCAP Session 5")
print("=" * 57)
print(f"  Weighted Accuracy   (WA) : {wa*100:.2f}%")
print(f"  Unweighted Accuracy (UA) : {ua*100:.2f}%")
print(f"  Macro F1-Score           : {f1*100:.2f}%")
print("=" * 57)
print("\n  Per-class breakdown:")
print(classification_report(all_labels, all_preds,
                             target_names=EMOTION_NAMES, digits=4))

# ── Confidence Analysis (ET-TACFN specific) ───────────────────
if all_conf_t:
    print("  Average Modality Confidence Scores:")
    print(f"    Text   : {np.mean(all_conf_t):.3f}")
    print(f"    Audio  : {np.mean(all_conf_a):.3f}")
    print(f"    Visual : {np.mean(all_conf_v):.3f}")
    print("  (Higher = model trusted this modality more on average)")

# ── Confusion Matrix ──────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_ylabel("True")
axes[0].set_xlabel("Predicted")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_ylabel("True")
axes[1].set_xlabel("Predicted")

plt.suptitle(
    f"ET-TACFN — WA: {wa*100:.2f}%   UA: {ua*100:.2f}%   F1: {f1*100:.2f}%",
    fontsize=13, fontweight="bold")
plt.tight_layout()

cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
print(f"\n  📊  Confusion matrix → {cm_path}")

# ── Modality Confidence Plot ──────────────────────────────────
if all_conf_t:
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Text", "Audio", "Visual"],
           [np.mean(all_conf_t), np.mean(all_conf_a), np.mean(all_conf_v)],
           color=["steelblue", "coral", "seagreen"])
    ax.set_title("ET-TACFN: Average Confidence Gate per Modality")
    ax.set_ylabel("Confidence Score (0–1)")
    ax.set_ylim(0, 1)
    conf_path = os.path.join(LOG_DIR, "confidence_scores.png")
    plt.tight_layout()
    plt.savefig(conf_path, dpi=150, bbox_inches="tight")
    print(f"  📊  Confidence plot   → {conf_path}")

# ── Save results text ─────────────────────────────────────────
results_path = os.path.join(LOG_DIR, "test_results.txt")
with open(results_path, "w") as f:
    f.write("ET-TACFN Test Results (IEMOCAP Session 5)\n")
    f.write("=" * 45 + "\n")
    f.write(f"WA  : {wa*100:.2f}%\n")
    f.write(f"UA  : {ua*100:.2f}%\n")
    f.write(f"F1  : {f1*100:.2f}%\n\n")
    f.write(classification_report(all_labels, all_preds,
                                  target_names=EMOTION_NAMES))
print(f"  📄  Results text      → {results_path}\n")
