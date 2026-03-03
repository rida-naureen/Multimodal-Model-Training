# train.py
# ============================================================
#  Training Script
#
#  What it does:
#    1. Loads train + val datasets from data/processed/
#    2. Builds the MultimodalEmotionModel
#    3. Trains for up to 30 epochs
#    4. Saves the BEST model to checkpoints/best_model.pt
#    5. Logs results to logs/training_log.csv
#    6. Stops early if val accuracy doesn't improve (patience=5)
#
#  Run:  python train.py
#
#  After training:
#    python plot_training.py  ← see loss/accuracy curves
#    python evaluate.py       ← test on Session 5
# ============================================================

import os
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.classifier import MultimodalEmotionModel

# ── Load config ───────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Device: use GPU if available, else CPU ────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️   Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"     GPU: {torch.cuda.get_device_name(0)}")

# ── Paths ─────────────────────────────────────────────────────
SPLITS_DIR = cfg["data"]["splits_dir"]
TEXT_DIR   = os.path.join(cfg["data"]["processed_dir"], "text_embeddings")
AUDIO_DIR  = os.path.join(cfg["data"]["processed_dir"], "audio_embeddings")
VISUAL_DIR = os.path.join(cfg["data"]["processed_dir"], "visual_embeddings")
CKPT_DIR   = cfg["paths"]["checkpoints"]
LOG_DIR    = cfg["paths"]["logs"]
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── Datasets ──────────────────────────────────────────────────
print("\n📂  Loading datasets...")
train_set = IEMOCAPDataset("train", SPLITS_DIR, TEXT_DIR, AUDIO_DIR, VISUAL_DIR)
val_set   = IEMOCAPDataset("val",   SPLITS_DIR, TEXT_DIR, AUDIO_DIR, VISUAL_DIR)

train_loader = DataLoader(
    train_set,
    batch_size  = cfg["training"]["batch_size"],
    shuffle     = True,    # shuffle train every epoch
    collate_fn  = collate_fn,
    num_workers = 0        # Windows: keep at 0
)
val_loader = DataLoader(
    val_set,
    batch_size  = cfg["training"]["batch_size"],
    shuffle     = False,   # don't shuffle val/test
    collate_fn  = collate_fn,
    num_workers = 0
)

# ── Model ─────────────────────────────────────────────────────
print("\n🤖  Building model...")
model = MultimodalEmotionModel(cfg).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Trainable parameters: {n_params:,}")

# ── Loss function ─────────────────────────────────────────────
# CrossEntropyLoss = measures how wrong the predictions are
criterion = nn.CrossEntropyLoss()

# ── Optimizer: AdamW (Adam + weight decay) ────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = cfg["training"]["lr"],
    weight_decay = cfg["training"]["weight_decay"]
)

# ── Scheduler: gradually reduce LR during training ───────────
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = cfg["training"]["epochs"]
)

# ── One epoch of training or validation ───────────────────────
def run_epoch(loader, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    # torch.no_grad() skips gradient calculation during validation
    # (faster + less memory)
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader,
                          desc="  Train" if is_train else "  Val  ",
                          leave=False):

            # Move data to GPU/CPU
            text   = batch["text"].to(DEVICE)
            audio  = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            t_mask = batch["text_mask"].to(DEVICE)
            a_mask = batch["audio_mask"].to(DEVICE)
            v_mask = batch["visual_mask"].to(DEVICE)

            # Forward pass: compute predictions
            logits, _ = model(text, audio, visual, t_mask, a_mask, v_mask)

            # Compute loss
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()     # clear old gradients
                loss.backward()           # compute new gradients
                # Clip gradients: prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()          # update weights

            # Track metrics
            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ── Main training loop ────────────────────────────────────────
EPOCHS   = cfg["training"]["epochs"]
PATIENCE = cfg["training"]["patience"]
CKPT_PATH = os.path.join(CKPT_DIR, "best_model.pt")

print(f"\n🚀  Training for up to {EPOCHS} epochs "
      f"(early stop patience = {PATIENCE})\n")

best_val_acc    = 0.0
patience_counter = 0
log_rows         = []

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = run_epoch(train_loader, is_train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   is_train=False)
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train  loss={train_loss:.4f}  acc={train_acc*100:.1f}%  |  "
          f"Val    loss={val_loss:.4f}  acc={val_acc*100:.1f}%  |  "
          f"lr={lr:.6f}")

    log_rows.append({
        "epoch"     : epoch,
        "train_loss": round(train_loss, 5),
        "train_acc" : round(train_acc,  5),
        "val_loss"  : round(val_loss,   5),
        "val_acc"   : round(val_acc,    5),
        "lr"        : round(lr,         8)
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        patience_counter = 0
        torch.save({
            "epoch"             : epoch,
            "model_state_dict"  : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc"           : val_acc,
            "cfg"               : cfg
        }, CKPT_PATH)
        print(f"  ✅  Best model saved  (val_acc = {val_acc*100:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️   Early stopping: no improvement for {PATIENCE} epochs")
            break

# ── Save training log ─────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "training_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\n{'='*55}")
print(f"  ✅  Training complete!")
print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
print(f"  Model saved to    : {CKPT_PATH}")
print(f"  Log saved to      : {log_path}")
print(f"\n  Next steps:")
print(f"    python plot_training.py   ← visualize training curves")
print(f"    python evaluate.py        ← test on Session 5")
print(f"{'='*55}\n")
