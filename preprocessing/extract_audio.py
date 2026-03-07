# preprocessing/extract_visual.py
# ============================================================
#  STEP 3C — Extract VISUAL features using ResNet-50
#
#  ⚠️  IEMOCAP has NO utterance-level video files.
#      Video is only at DIALOG level: dialog/avi/DivX/Ses01F_impro01.avi
#
#  Strategy:
#    1. Read emotion labels to get each utterance's start/end timestamp
#    2. Open the dialog-level AVI
#    3. Seek to the utterance's time window
#    4. Sample 30 frames from that window
#    5. Pass through ResNet-50 → project 2048→256
#    6. Save as utterance-level .npy
#
#  Saves: data/processed/visual_embeddings/Ses01F_impro01_F000.npy
#         shape: [30, 256]
#
#  Time: ~40–60 min for all sessions
#  Run:  python preprocessing\extract_visual.py
# ============================================================

import os
import re
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/visual_embeddings"
MAX_FRAMES = 30

print("\n  Loading ResNet-50...")
backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # remove FC
backbone.eval()

projector = torch.nn.Linear(2048, 256)
projector.eval()

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone  = backbone.to(device)
projector = projector.to(device)
print(f"  Device: {device}")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])


def parse_timestamps(session_num):
    """
    Reads EmoEvaluation/*.txt to get utterance timestamps.
    Returns dict: { "Ses01F_impro01_F000": (start_sec, end_sec), ... }

    Label line format:
      [6.2901 - 8.2357]  Ses01F_impro01_F000  hap  [2.5, 2.5, 2.5]
    """
    timestamps = {}
    emo_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                           "dialog", "EmoEvaluation")
    if not os.path.exists(emo_dir):
        return timestamps

    for fname in os.listdir(emo_dir):
        if not fname.endswith(".txt") or fname.startswith("._"):
            continue
        with open(os.path.join(emo_dir, fname),
                  encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Match: [6.2901 - 8.2357]
                m = re.match(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)', line)
                if m:
                    start  = float(m.group(1))
                    end    = float(m.group(2))
                    utt_id = m.group(3).strip()
                    timestamps[utt_id] = (start, end)
    return timestamps


def extract_frames_from_dialog(avi_path, start_sec, end_sec):
    """
    Opens dialog-level AVI, seeks to [start_sec, end_sec],
    samples MAX_FRAMES evenly spaced frames.
    Returns numpy array [MAX_FRAMES, 256] or None on failure.
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return None

    fps        = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    n_frames    = max(end_frame - start_frame, 1)

    indices = np.linspace(start_frame, end_frame - 1, MAX_FRAMES, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # If seek fails, use a blank frame
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()

    feats = []
    with torch.no_grad():
        for frame in frames:
            t    = transform(frame).unsqueeze(0).to(device)
            feat = backbone(t).squeeze()          # [2048]
            feat = projector(feat)                # [256]
            feats.append(feat.cpu().numpy())

    return np.stack(feats)  # [30, 256]


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = skipped = failed = 0

    for session_num in range(1, 6):
        avi_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                               "dialog", "avi", "DivX")
        if not os.path.exists(avi_dir):
            print(f"  ⚠️  Session{session_num}: dialog/avi/DivX/ not found")
            continue

        # Get all utterance timestamps
        timestamps = parse_timestamps(session_num)
        if not timestamps:
            print(f"  ⚠️  Session{session_num}: no timestamps found")
            continue

        # Group utterances by their dialog (e.g. Ses01F_impro01)
        dialog_utts = {}
        for utt_id, ts in timestamps.items():
            # Extract dialog name: Ses01F_impro01 from Ses01F_impro01_F000
            parts  = utt_id.rsplit("_", 1)
            dialog = parts[0]  # e.g. Ses01F_impro01
            dialog_utts.setdefault(dialog, []).append((utt_id, ts))

        # Get available AVI files
        avis = {os.path.splitext(f)[0]: os.path.join(avi_dir, f)
                for f in os.listdir(avi_dir)
                if f.endswith(".avi") and not f.startswith("._")}

        print(f"\n  Session{session_num} — {len(timestamps)} utterances "
              f"across {len(avis)} dialog AVIs")

        for dialog, utts in tqdm(dialog_utts.items(), desc=f"  Ses0{session_num}"):
            avi_path = avis.get(dialog)
            if not avi_path:
                failed += len(utts)
                continue

            for utt_id, (start, end) in utts:
                save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")
                if os.path.exists(save_path):
                    skipped += 1
                    continue

                try:
                    feats = extract_frames_from_dialog(avi_path, start, end)
                    if feats is not None:
                        np.save(save_path, feats)
                        saved += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"\n  ⚠️  {utt_id}: {e}")
                    failed += 1

    print(f"\n  ✅  Visual done: {saved} saved, {skipped} existed, {failed} failed")
    print(f"      → {OUTPUT_DIR}")
    print("\n  Preprocessing complete! Run next:")
    print("      python train.py\n")


if __name__ == "__main__":
    extract_all()
