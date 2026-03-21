# train.py
# ============================================================
#  ET-TACFN Training Script
#
#  Features:
#    • Weighted cross-entropy loss  (handles class imbalance)
#    • Label smoothing 0.1
#    • ReduceLROnPlateau scheduler
#    • Gradient clipping
#    • Mixed precision (AMP) + gradient accumulation
#    • Early stopping
#    • Saves best checkpoint + training log CSV
#
#  Run:  python train.py
# ============================================================

import os
import csv
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from dataset.iemocap_dataset import IEMOCAPDataset, collate_fn, EMOTION_TO_IDX
from models.classifier import MultimodalEmotionModel

# ── Config ────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"\n🖥️   Device   : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"     GPU      : {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"     VRAM     : {total_mem:.1f} GB")
    print(f"     AMP      : enabled")
else:
    print(f"     AMP      : disabled (CPU mode)")

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
train_set = IEMOCAPDataset("train", SPLITS_DIR, TEXT_DIR, AUDIO_DIR, VISUAL_DIR, cfg=cfg)
val_set   = IEMOCAPDataset("val",   SPLITS_DIR, TEXT_DIR, AUDIO_DIR, VISUAL_DIR, cfg=cfg)

if len(train_set) == 0:
    raise RuntimeError("❌  Training set is empty — check that text and audio "
                       ".npy files exist in the processed directories.")
if len(val_set) == 0:
    print("⚠️   Validation set is empty — val loss/acc will read 0.0 each epoch.")

ACCUM_STEPS = cfg["training"].get("grad_accum_steps", 4)
print(f"\n🔄  Gradient accumulation : {ACCUM_STEPS} steps "
      f"(effective batch = {cfg['training']['batch_size'] * ACCUM_STEPS})")

train_loader = DataLoader(
    train_set, batch_size=cfg["training"]["batch_size"],
    shuffle=True, collate_fn=collate_fn, num_workers=0,
    pin_memory=(DEVICE.type == "cuda"))
val_loader = DataLoader(
    val_set, batch_size=cfg["training"]["batch_size"],
    shuffle=False, collate_fn=collate_fn, num_workers=0,
    pin_memory=(DEVICE.type == "cuda"))

# ── Class-weighted loss (handles class imbalance) ─────────────
print("\n⚖️   Computing class weights...")
all_labels = [train_set.label_map[uid] for uid in train_set.utt_ids]
label_ints = [EMOTION_TO_IDX[e] for e in all_labels]

class_weights = compute_class_weight(
    class_weight = "balanced",
    classes      = np.array([0, 1, 2, 3]),
    y            = np.array(label_ints)
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"     Class weights: "
      f"{ {['hap','sad','ang','neu'][i]: round(w,3) for i,w in enumerate(class_weights)} }")

criterion = nn.CrossEntropyLoss(
    weight          = weights_tensor,
    label_smoothing = cfg["training"]["label_smoothing"]
)

# ── Model ─────────────────────────────────────────────────────
print("\n🤖  Building ET-TACFN model...")
model = MultimodalEmotionModel(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"     Trainable parameters: {n_params:,}")

# ── Optimizer ─────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = cfg["training"]["lr"],
    weight_decay = cfg["training"]["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
)

scaler = GradScaler(enabled=USE_AMP)
EPOCHS = cfg["training"]["epochs"]

# ── Train / Val loop ──────────────────────────────────────────
def run_epoch(loader, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(
            loader, desc="  Train" if is_train else "  Val  ", leave=False)):

        text   = batch["text"].to(DEVICE,   non_blocking=True)
        audio  = batch["audio"].to(DEVICE,  non_blocking=True)
        visual = batch["visual"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE,  non_blocking=True)
        t_mask = batch["text_mask"].to(DEVICE,   non_blocking=True)
        a_mask = batch["audio_mask"].to(DEVICE,  non_blocking=True)
        v_mask = batch["visual_mask"].to(DEVICE, non_blocking=True)

        with autocast(enabled=USE_AMP):
            with torch.set_grad_enabled(is_train):
                logits, _ = model(text, audio, visual, t_mask, a_mask, v_mask)
                loss      = criterion(logits, labels)
                if is_train:
                    loss = loss / ACCUM_STEPS

        if is_train:
            scaler.scale(loss).backward()
            is_update_step = (step + 1) % ACCUM_STEPS == 0 or \
                             (step + 1) == len(loader)
            if is_update_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        raw_loss = loss.item() * (ACCUM_STEPS if is_train else 1)
        total_loss += raw_loss * labels.size(0)
        preds       = logits.detach().argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        del text, audio, visual, labels, t_mask, a_mask, v_mask, logits

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


# ── Main loop ─────────────────────────────────────────────────
PATIENCE  = cfg["training"]["patience"]
CKPT_PATH = os.path.join(CKPT_DIR, "best_model.pt")

print(f"\n🚀  Training ET-TACFN for up to {EPOCHS} epochs\n")

best_val_acc     = 0.0
patience_counter = 0
log_rows         = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, is_train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   is_train=False)

    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train  loss={train_loss:.4f}  acc={train_acc*100:.1f}%  |  "
          f"Val    loss={val_loss:.4f}  acc={val_acc*100:.1f}%  |  "
          f"lr={lr:.6f}")

    scheduler.step(val_loss)

    log_rows.append({
        "epoch"     : epoch,
        "train_loss": round(train_loss, 5),
        "train_acc" : round(train_acc,  5),
        "val_loss"  : round(val_loss,   5),
        "val_acc"   : round(val_acc,    5),
        "lr"        : round(lr,         8)
    })

    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        patience_counter = 0
        torch.save({
            "epoch"              : epoch,
            "model_state_dict"   : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc"            : val_acc,
            "cfg"                : cfg
        }, CKPT_PATH)
        print(f"  ✅  Best model saved  (val_acc={val_acc*100:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️   Early stopping at epoch {epoch}")
            break

# ── Save log ──────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "training_log.csv")
if log_rows:
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

print(f"\n{'='*55}")
print(f"  ✅  Training complete!")
print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
print(f"  Model saved       : {CKPT_PATH}")
print(f"  Log saved         : {log_path}")
print(f"\n  Next → python evaluate.py")
print(f"{'='*55}\n")
