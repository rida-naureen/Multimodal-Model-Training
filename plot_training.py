# plot_training.py
# ============================================================
#  Visualize training curves after training is done.
#  Run:  python plot_training.py
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "logs/training_log.csv"

if not os.path.exists(LOG_PATH):
    print(f"❌  {LOG_PATH} not found. Run train.py first.")
    exit()

df = pd.read_csv(LOG_PATH)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Loss ────────────────────────────────────────
axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss",
             color="steelblue", linewidth=2)
axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",
             color="coral",     linewidth=2)
axes[0].set_title("Loss per Epoch", fontsize=13)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Accuracy ────────────────────────────────────
best_epoch = df.loc[df["val_acc"].idxmax(), "epoch"]
best_acc   = df["val_acc"].max() * 100

axes[1].plot(df["epoch"], df["train_acc"] * 100, label="Train Acc",
             color="steelblue", linewidth=2)
axes[1].plot(df["epoch"], df["val_acc"]   * 100, label="Val Acc",
             color="coral",     linewidth=2)
axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best: {best_acc:.1f}% @ epoch {best_epoch}")
axes[1].set_title("Accuracy per Epoch", fontsize=13)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("IEMOCAP Cross-Modal Attention — Training Curves",
             fontsize=14, fontweight="bold")
plt.tight_layout()

out_path = "logs/training_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✅  Saved → {out_path}")
plt.show()
