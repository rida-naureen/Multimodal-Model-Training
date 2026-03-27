# inference.py
# ============================================================
#  ET-TACFN Inference Script
#
#  Supports:
#    1. Single utterance inference (audio + optional video + optional text)
#    2. Batch inference on a folder
#    3. Transfer learning fine-tuning on a new dataset
#
#  Usage:
#    # Single file:
#    python inference.py --audio path/to/audio.wav --video path/to/video.mp4
#
#    # Batch folder inference:
#    python inference.py --batch_dir path/to/folder/
#
#    # Transfer-learn on new dataset:
#    python inference.py --finetune --new_data_dir path/to/new_data/
#
# ============================================================

import os
import re
import argparse
import numpy as np
import torch
import yaml
import torchaudio
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = os.path.join(cfg["paths"]["checkpoints"], "best_model.pt")

EMOTION_LABELS = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral"}
TARGET_SR      = 16000


# ── Load Model ────────────────────────────────────────────────
def load_model(ckpt_path=CKPT_PATH, cfg=cfg):
    from models.classifier import MultimodalEmotionModel
    model = MultimodalEmotionModel(cfg).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  ✅  Loaded checkpoint: epoch {ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']*100:.2f}%")
    return model


# ── Feature Extraction ────────────────────────────────────────

def extract_audio_features(wav_path):
    """
    Extract WavLM audio features from a .wav file.
    Returns numpy array of shape [T_a, 768].
    Uses the same encoders.py pipeline as training.
    """
    from preprocessing.encoders import get_audio_encoder
    encoder = get_audio_encoder()

    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)
    feats = encoder.encode(audio_np)   # returns (768,) mean-pooled

    # Expand to [1, 768] so it has a time dimension for the attention layers
    if feats.ndim == 1:
        feats = feats[np.newaxis, :]
    return feats


def extract_text_features(wav_path_or_text):
    """
    Extract RoBERTa text features.
    If a .wav path is given, Whisper transcribes it first.
    If a string is given, it encodes it directly.
    Returns numpy array of shape [1, 768].
    """
    from preprocessing.encoders import get_text_encoder
    pipeline = get_text_encoder()

    if wav_path_or_text.endswith(".wav") and os.path.exists(wav_path_or_text):
        result = pipeline.process(wav_path_or_text)
        feats = np.array(result["text_features"], dtype=np.float32)
    else:
        feats = pipeline.encode_text(wav_path_or_text)

    if feats.ndim == 1:
        feats = feats[np.newaxis, :]
    return feats


def extract_visual_features(video_path, utt_id="unknown"):
    """
    Extract ResNet-50 visual features with MTCNN face cropping.
    video_path : path to .mp4 or .avi file
    start_sec / end_sec : optional time window (if None, uses full video)
    Returns numpy array of shape [T_v, 2048].
    """
    from preprocessing.extract_visual import sample_and_crop_faces, embed_frames
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    end_sec = total_frames / fps
    cap.release()

    faces = sample_and_crop_faces(video_path, 0.0, end_sec, utt_id)
    if faces is None or len(faces) == 0:
        return np.zeros((1, 2048), dtype=np.float32)
    return embed_frames(faces)   # [T_v, 2048]


# ── Inference ────────────────────────────────────────────────

def predict(model, audio_feats, text_feats=None, visual_feats=None):
    """
    Run one utterance through the model.

    Args:
        model        : loaded ET-TACFN model
        audio_feats  : np.array [T_a, audio_dim]
        text_feats   : np.array [T_t, text_dim]   or None
        visual_feats : np.array [T_v, visual_dim] or None

    Returns:
        pred_label   : string (e.g. "Happy")
        confidence   : float (0–1)
        all_probs    : dict of {emotion: probability}
    """
    def to_tensor_3d(arr):
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, T, D]
        return t, torch.zeros(1, t.shape[1], dtype=torch.bool).to(DEVICE)

    audio,  a_mask = to_tensor_3d(audio_feats)
    text,   t_mask = to_tensor_3d(text_feats)   if text_feats   is not None \
                     else (None, None)
    visual, v_mask = to_tensor_3d(visual_feats) if visual_feats is not None \
                     else (None, None)

    with torch.no_grad():
        logits, info = model(text, audio, visual, t_mask, a_mask, v_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred_idx    = int(probs.argmax())
    pred_label  = EMOTION_LABELS[pred_idx]
    confidence  = float(probs[pred_idx])
    all_probs   = {EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)}

    return pred_label, confidence, all_probs


def run_single(args, model):
    """Run inference on a single audio/video/text input."""
    print(f"\n  Extracting features...")

    audio_feats  = extract_audio_features(args.audio)
    text_feats   = extract_text_features(args.audio) if not args.text \
                   else extract_text_features(args.text)
    visual_feats = extract_visual_features(args.video) if args.video else None

    label, conf, probs = predict(model, audio_feats, text_feats, visual_feats)

    print(f"\n  ── Prediction ──────────────────────────")
    print(f"  Emotion     : {label}")
    print(f"  Confidence  : {conf*100:.1f}%")
    print(f"  All probs   : { {k: f'{v*100:.1f}%' for k, v in probs.items()} }")
    print(f"  ────────────────────────────────────────")


def run_batch(args, model):
    """Run inference on all .wav files in a directory."""
    wav_paths = list(Path(args.batch_dir).glob("**/*.wav"))
    print(f"\n  Found {len(wav_paths)} .wav files in {args.batch_dir}")

    results = []
    for wav_path in wav_paths:
        audio_feats  = extract_audio_features(str(wav_path))
        text_feats   = extract_text_features(str(wav_path))
        visual_feats = None

        # Try to find a matching video file
        video_candidates = list(wav_path.parent.glob(f"{wav_path.stem}.*"))
        video_candidates = [v for v in video_candidates
                            if v.suffix in (".mp4", ".avi", ".mov")]
        if video_candidates:
            visual_feats = extract_visual_features(str(video_candidates[0]))

        label, conf, probs = predict(model, audio_feats, text_feats, visual_feats)
        results.append({
            "file"      : wav_path.name,
            "emotion"   : label,
            "confidence": f"{conf*100:.1f}%",
        })
        print(f"  {wav_path.name:<40} → {label:<8} ({conf*100:.1f}%)")

    # Save CSV
    import csv
    out_csv = os.path.join(args.batch_dir, "predictions.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "emotion", "confidence"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  ✅  Saved predictions to: {out_csv}")


# ── Transfer Learning ─────────────────────────────────────────

def finetune(args):
    """
    Fine-tune the pre-trained ET-TACFN on a new dataset.

    Your new dataset directory should have this structure:
        new_data_dir/
            train_ids.txt      ← list of utterance IDs
            val_ids.txt
            labels.txt         ← <utt_id>\\t<emotion_label>
            text_embeddings/   ← <utt_id>.npy
            audio_embeddings/  ← <utt_id>.npy
            visual_embeddings/ ← <utt_id>.npy

    Emotion labels must be one of: hap, sad, ang, neu (or exc → hap)

    Strategy:
        - Freeze the fusion encoders for the first N epochs (feature extraction)
        - Then unfreeze all and train end-to-end with a lower LR
    """
    from dataset.iemocap_dataset import IEMOCAPDataset, collate_fn
    from torch.utils.data import DataLoader
    import torch.nn as nn

    print("\n  📦  Loading new dataset for fine-tuning...")

    # Assumes your new dataset has the same split/embedding structure
    train_set = IEMOCAPDataset(
        "train",
        splits_dir  = args.new_data_dir,
        text_dir    = os.path.join(args.new_data_dir, "text_embeddings"),
        audio_dir   = os.path.join(args.new_data_dir, "audio_embeddings"),
        visual_dir  = os.path.join(args.new_data_dir, "visual_embeddings"),
        cfg         = cfg
    )
    val_set = IEMOCAPDataset(
        "val",
        splits_dir  = args.new_data_dir,
        text_dir    = os.path.join(args.new_data_dir, "text_embeddings"),
        audio_dir   = os.path.join(args.new_data_dir, "audio_embeddings"),
        visual_dir  = os.path.join(args.new_data_dir, "visual_embeddings"),
        cfg         = cfg
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Load the pre-trained model
    model = load_model()

    # ── Phase 1: Freeze fusion, train classifier only (5 epochs) ──
    # This avoids destroying pre-trained representations immediately.
    print("\n  Phase 1: Freezing fusion layers, training classifier only...")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        model.train()
        for batch in train_loader:
            text   = batch["text"].to(DEVICE)
            audio  = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            t_mask = batch["text_mask"].to(DEVICE)
            a_mask = batch["audio_mask"].to(DEVICE)
            v_mask = batch["visual_mask"].to(DEVICE)
            logits, _ = model(text, audio, visual, t_mask, a_mask, v_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  Phase 1 — Epoch {epoch}/5  loss={loss.item():.4f}")

    # ── Phase 2: Unfreeze all, fine-tune end-to-end with low LR ──
    print("\n  Phase 2: Unfreezing all layers, end-to-end fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        for batch in train_loader:
            text   = batch["text"].to(DEVICE)
            audio  = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            t_mask = batch["text_mask"].to(DEVICE)
            a_mask = batch["audio_mask"].to(DEVICE)
            v_mask = batch["visual_mask"].to(DEVICE)
            logits, _ = model(text, audio, visual, t_mask, a_mask, v_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                text   = batch["text"].to(DEVICE)
                audio  = batch["audio"].to(DEVICE)
                visual = batch["visual"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                t_mask = batch["text_mask"].to(DEVICE)
                a_mask = batch["audio_mask"].to(DEVICE)
                v_mask = batch["visual_mask"].to(DEVICE)
                logits, _ = model(text, audio, visual, t_mask, a_mask, v_mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)
        print(f"  Phase 2 — Epoch {epoch}/{args.finetune_epochs}  "
              f"val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ft_ckpt = os.path.join(cfg["paths"]["checkpoints"], "finetuned_model.pt")
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "val_acc"          : val_acc,
                "cfg"              : cfg
            }, ft_ckpt)
            print(f"  ✅  Saved finetuned model (val_acc={val_acc*100:.2f}%)")

    print(f"\n  ✅  Fine-tuning complete! Best val_acc: {best_val_acc*100:.2f}%")


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET-TACFN Inference & Fine-tuning")

    # Single utterance inference
    parser.add_argument("--audio",     type=str, help="Path to .wav file")
    parser.add_argument("--video",     type=str, help="Path to .mp4/.avi file (optional)")
    parser.add_argument("--text",      type=str, help="Text transcript (optional, else Whisper transcribes)")

    # Batch inference
    parser.add_argument("--batch_dir", type=str, help="Folder of .wav files for batch inference")

    # Fine-tuning
    parser.add_argument("--finetune",        action="store_true",
                        help="Enable transfer learning mode")
    parser.add_argument("--new_data_dir",    type=str,
                        help="Path to new dataset directory (must follow IEMOCAP embedding structure)")
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Number of fine-tuning epochs (phase 2)")

    # Checkpoint override
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH,
                        help="Path to checkpoint file")

    args = parser.parse_args()

    if args.finetune:
        finetune(args)
    elif args.batch_dir:
        model = load_model(args.ckpt)
        run_batch(args, model)
    elif args.audio:
        model = load_model(args.ckpt)
        run_single(args, model)
    else:
        parser.print_help()
