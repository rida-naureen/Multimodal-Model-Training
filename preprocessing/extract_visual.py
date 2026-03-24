# preprocessing/extract_visual.py
# ============================================================
#  STEP 4D — Extract VISUAL features using ResNet-50 + MTCNN
#
#  Reads:  data/raw/SessionX/dialog/avi/DivX/<dialog>.avi
#          data/splits/train_ids.txt, val_ids.txt, test_ids.txt
#
#  Saves:  data/processed/visual_embeddings/<utt_id>.npy
#          shape: [T_v, 256]  (variable length frames, 15 FPS)
# ============================================================

import os
import re
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ── paths ────────────────────────────────────────────────────
RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/visual_embeddings"
TARGET_FPS = 15          # frames sampled per second
FEAT_DIM   = 256         # ResNet-50 layer4 avg-pool → 2048, then projected

# ── Actor side mapping (IEMOCAP conventions) ─────────────────
# Usually M is left in Ses01/02, F is left in Ses03/04/05
ACTOR_POSITIONS = {
    "01": {"M": "Left",  "F": "Right"},
    "02": {"M": "Left",  "F": "Right"},
    "03": {"M": "Right", "F": "Left"},
    "04": {"M": "Right", "F": "Left"},
    "05": {"M": "Right", "F": "Left"}
}

# ── build extractors ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")

print("  Building MTCNN face detector...")
# keep_all=True to detect multiple faces in the frame
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

print("  Building ResNet-50 feature extractor...")
_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(
    *list(_resnet.children())[:-1],           # → [B, 2048, 1, 1]
    torch.nn.Flatten(),                       # → [B, 2048]
    torch.nn.Linear(2048, FEAT_DIM),          # → [B, 256]
    torch.nn.ReLU(),
)
feature_extractor.eval()
feature_extractor = feature_extractor.to(device)

# ── image pre-processing ─────────────────────────────────────
# (MTCNN outputs cropped faces directly, but we still need ResNet norm)
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── helpers ───────────────────────────────────────────────────

def parse_emo_evaluation(emo_file):
    timestamps = {}
    with open(emo_file, "r", encoding="latin-1") as f:
        for line in f:
            m = re.match(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+\w+", line)
            if m:
                start = float(m.group(1))
                end   = float(m.group(2))
                utt   = m.group(3)
                timestamps[utt] = (start, end)
    return timestamps


def get_target_side(utt_id):
    """
    Given 'Ses01F_impro01_F001', figure out if the active speaker is Left or Right.
    """
    m = re.match(r"Ses(0[1-5])[a-zA-Z]_.*_([MF])\d{3}", utt_id)
    if m:
        session_num = m.group(1)
        speaker_gen = m.group(2)
        return ACTOR_POSITIONS.get(session_num, {}).get(speaker_gen, None)
    return None


def sample_and_crop_faces(video_path, start_sec, end_sec, utt_id):
    """
    Open video, seek to [start_sec, end_sec], sample at TARGET_FPS.
    Detect faces using MTCNN, crop the correct actor's face.
    Returns list of RGB cropped faces (H, W, 3).
    """
    target_side = get_target_side(utt_id)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    
    # Calculate how many frames to extract to get TARGET_FPS
    duration_sec = end_sec - start_sec
    if duration_sec <= 0:
        duration_sec = 0.5
    n_frames = max(1, int(duration_sec * TARGET_FPS))
    
    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
    
    cropped_faces = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            if cropped_faces:
                cropped_faces.append(cropped_faces[-1])
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            boxes, probs = mtcnn.detect(rgb_frame)
            
            if boxes is not None and len(boxes) > 0:
                # Find the box that matches the target side
                selected_box = None
                
                if target_side and len(boxes) >= 2:
                    # Sort by X coordinate
                    boxes_sorted = sorted(boxes, key=lambda b: b[0])
                    if target_side == "Left":
                        selected_box = boxes_sorted[0]
                    else:
                        selected_box = boxes_sorted[-1]
                else:
                    # Fallback: just use the bounding box with highest prob
                    best_idx = np.argmax(probs)
                    selected_box = boxes[best_idx]
                
                # Crop the face
                x1, y1, x2, y2 = [int(v) for v in selected_box]
                
                # Add some padding (margin)
                margin = 20
                h, w, _ = rgb_frame.shape
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face_crop = rgb_frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    cropped_faces.append(face_crop)
                elif cropped_faces:
                    cropped_faces.append(cropped_faces[-1])
            else:
                # No face detected
                if cropped_faces:
                    cropped_faces.append(cropped_faces[-1])
                else:
                    # Pad with zero image if nothing found yet
                    cropped_faces.append(np.zeros((224, 224, 3), dtype=np.uint8))
        except Exception as e:
            if cropped_faces:
                cropped_faces.append(cropped_faces[-1])
            else:
                cropped_faces.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return cropped_faces


def embed_frames(frames):
    """
    RGB face crop list → numpy [T_v, FEAT_DIM]
    """
    if not frames:
        return np.zeros((1, FEAT_DIM), dtype=np.float32)
        
    tensors = []
    for rgb in frames:
        # Avoid crashing on Empty frames
        if rgb.size == 0:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        tensors.append(preprocess(rgb))
        
    batch = torch.stack(tensors).to(device)         # [T_v, 3, 224, 224]
    
    # Process in chunks to avoid OOM if video is very long
    chunk_size = 64
    feats_list = []
    
    with torch.no_grad():
        for i in range(0, batch.shape[0], chunk_size):
            chunk = batch[i:i+chunk_size]
            feats = feature_extractor(chunk)             # [chunk, FEAT_DIM]
            feats_list.append(feats)
            
    all_feats = torch.cat(feats_list, dim=0)
    return all_feats.cpu().numpy()


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

        emo_files = sorted(f for f in os.listdir(emo_dir) if f.endswith(".txt"))
        print(f"\n  Session{session_num} — {len(emo_files)} dialog(s)")

        for emo_fname in tqdm(emo_files, desc=f"  Ses0{session_num}"):
            dialog_id = emo_fname.replace(".txt", "")
            emo_path  = os.path.join(emo_dir, emo_fname)
            avi_path  = os.path.join(avi_dir, f"{dialog_id}.avi")

            if not os.path.exists(avi_path):
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
                    faces = sample_and_crop_faces(avi_path, start_sec, end_sec, utt_id)
                    if faces is None:
                        errors += 1
                        continue
                    emb = embed_frames(faces)           # [T_v, 256]
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
