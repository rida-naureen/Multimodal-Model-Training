# preprocessing/extract_visual.py
# ============================================================
#  STEP 3C — Extract VISUAL features using ResNet-50
#
#  What it does:
#    • Reads .avi video files (facial expressions)
#    • Samples 30 evenly-spaced frames per video
#    • Passes each frame through ResNet-50
#    • Projects 2048-dim → 256-dim
#    • Saves embedding as .npy file:
#        data/processed/visual_embeddings/Ses01F_impro01_F000.npy
#        shape: [30, 256]
#
#  Time: ~40–60 minutes for all 5 sessions
#  Run:  python preprocessing/extract_visual.py
# ============================================================

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/visual_embeddings"
MAX_FRAMES = 30   # sample this many frames per video

# ── Load ResNet-50, remove final FC layer ─────────────────────
print("\n  Loading ResNet-50...")
backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
backbone.eval()

# Project 2048 → 256
projector = torch.nn.Linear(2048, 256)
projector.eval()

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone  = backbone.to(device)
projector = projector.to(device)
print(f"  Using device: {device}")

# Image preprocessing pipeline
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])


def extract_for_video(avi_path):
    """
    avi_path → numpy array [MAX_FRAMES, 256]
    """
    cap    = cv2.VideoCapture(avi_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    # Sample MAX_FRAMES evenly spaced frames
    indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
    sampled = [frames[i] for i in indices]

    frame_features = []
    with torch.no_grad():
        for frame in sampled:
            tensor = transform(frame).unsqueeze(0).to(device)  # [1,3,224,224]
            feat   = backbone(tensor).squeeze()                 # [2048]
            feat   = projector(feat)                           # [256]
            frame_features.append(feat.cpu().numpy())

    return np.stack(frame_features)   # [MAX_FRAMES, 256]


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0

    for session_num in range(1, 6):
        session  = f"Session{session_num}"
        avi_root = os.path.join(RAW_DIR, session, "sentences", "avi")

        if not os.path.exists(avi_root):
            print(f"  ⚠️   {session}: sentences/avi/ not found, skipping")
            continue

        avi_files = []
        for subdir in os.listdir(avi_root):
            subdir_path = os.path.join(avi_root, subdir)
            if os.path.isdir(subdir_path):
                for af in os.listdir(subdir_path):
                    if af.endswith(".avi"):
                        avi_files.append(os.path.join(subdir_path, af))

        print(f"\n  {session} — {len(avi_files)} AVI files")

        for avi_path in tqdm(avi_files, desc=f"  {session}"):
            utt_id    = os.path.splitext(os.path.basename(avi_path))[0]
            save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")

            if os.path.exists(save_path):
                continue

            try:
                embedding = extract_for_video(avi_path)
                if embedding is not None:
                    np.save(save_path, embedding)
                    total_saved += 1
            except Exception as e:
                print(f"\n  ⚠️   Skipped {utt_id}: {e}")

    print(f"\n  ✅  Visual extraction complete: {total_saved} files saved")
    print(f"      Location: {OUTPUT_DIR}")
    print("\n  Preprocessing DONE! Run next:")
    print("      python train.py\n")


if __name__ == "__main__":
    extract_all()
