# preprocessing/extract_visual.py
# ============================================================
#  STEP 4D — Extract VISUAL features using ResNet-50
#
#  Reads:  data/raw/SessionX/dialog/avi/DivX/<dialog>.avi
#          data/splits/train_ids.txt, val_ids.txt, test_ids.txt
#          (to know utterance timestamps from EmoEvaluation .txt)
#
#  Saves:  data/processed/visual_embeddings/<utt_id>.npy
#          shape: [30, 256]  (30 frames uniformly sampled, 256-d pool5 feat)
#
#  Time:   ~40–60 min for all sessions
#  Run:    python preprocessing\extract_visual.py
# ============================================================

import os
import re
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────
RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/visual_embeddings"
NUM_FRAMES = 30          # frames sampled per utterance
FEAT_DIM   = 256         # ResNet-50 layer4 avg-pool → 2048, then projected

# ── build ResNet-50 feature extractor ────────────────────────
from preprocessing.encoders import get_visual_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ── image pre-processing (ImageNet stats) ─────────────────────
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── helpers ───────────────────────────────────────────────────

def parse_emo_evaluation(emo_file):
    """
    Parse an EmoEvaluation .txt to get utterance start/end timestamps.
    Returns dict: {utt_id: (start_sec, end_sec)}
    """
    timestamps = {}
    with open(emo_file, "r", encoding="latin-1") as f:  # latin-1 safe for IEMOCAP annotation files
        for line in f:
            # Format: [start_time - end_time]	Ses01F_impro01_F001	ang	...
            m = re.match(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+\w+", line)
            if m:
                start = float(m.group(1))
                end   = float(m.group(2))
                utt   = m.group(3)
                timestamps[utt] = (start, end)
    return timestamps


def sample_frames_from_video(video_path, start_sec, end_sec, n_frames=NUM_FRAMES):
    """
    Open video, seek to [start_sec, end_sec], uniformly sample n_frames.
    Returns list of BGR uint8 numpy arrays (H, W, 3), or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    total       = max(end_frame - start_frame, 1)

    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # pad with last valid or black frame
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            frames.append(frame)

    cap.release()
    return frames   # list of n_frames BGR images


def embed_frames(frames):
    """
    BGR frame list → numpy (2048,)
    """
    # Convert BGR frames from OpenCV to RGB for the encoder
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames if f is not None]
    
    encoder = get_visual_encoder()
    emb = encoder.encode(rgb_frames)
    return emb


# ── main extraction loop ──────────────────────────────────────

def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = skipped = errors = 0

    for session_num in range(1, 6):
        session_dir = os.path.join(RAW_DIR, f"Session{session_num}")
        emo_dir     = os.path.join(session_dir, "dialog", "EmoEvaluation")
        avi_dir     = os.path.join(session_dir, "dialog", "avi", "DivX")

        if not os.path.exists(emo_dir):
            print(f"  ⚠️  Session{session_num}: EmoEvaluation/ not found — skipping")
            continue
        if not os.path.exists(avi_dir):
            print(f"  ⚠️  Session{session_num}: dialog/avi/DivX/ not found — skipping")
            continue

        # Gather all EmoEvaluation files → utterance timestamps
        emo_files = sorted(f for f in os.listdir(emo_dir) if f.endswith(".txt"))
        print(f"\n  Session{session_num} — {len(emo_files)} dialog(s)")

        for emo_fname in tqdm(emo_files, desc=f"  Ses0{session_num}"):
            dialog_id = emo_fname.replace(".txt", "")
            emo_path  = os.path.join(emo_dir, emo_fname)
            # Find corresponding AVI  (e.g. Ses01F_impro01.avi)
            avi_path  = os.path.join(avi_dir, f"{dialog_id}.avi")

            if not os.path.exists(avi_path):
                # Sometimes the AVI name ends in _X; try glob
                candidates = [f for f in os.listdir(avi_dir)
                              if f.startswith(dialog_id) and f.endswith(".avi")]
                if candidates:
                    avi_path = os.path.join(avi_dir, sorted(candidates)[0])
                else:
                    errors += 1
                    continue

            timestamps = parse_emo_evaluation(emo_path)

            for utt_id, (start_sec, end_sec) in timestamps.items():
                save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")
                if os.path.exists(save_path):
                    skipped += 1
                    continue

                try:
                    frames = sample_frames_from_video(avi_path, start_sec, end_sec)
                    if frames is None:
                        errors += 1
                        continue
                    emb = embed_frames(frames)           # [30, 256]
                    np.save(save_path, emb)
                    saved += 1
                except Exception as e:
                    print(f"\n  ⚠️  {utt_id}: {e}")
                    errors += 1

    print(f"\n  ✅  Visual done: {saved} saved, {skipped} already existed, {errors} errors")
    print(f"      → {OUTPUT_DIR}")
    print("\n  Run next:")
    print("      python preprocessing\\check_data.py\n")


if __name__ == "__main__":
    extract_all()
