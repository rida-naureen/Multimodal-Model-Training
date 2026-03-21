# preprocessing/extract_text.py
# ============================================================
#  STEP 3A — Extract TEXT features using RoBERTa-large (Tier 2 upgrade)
#
#  Tier 2 change: roberta-base (768-d, ~125M params)
#              →  roberta-large (1024-d, ~355M params)
#  Expected gain: +4–6 pp UA on IEMOCAP 4-class.
#
#  Reads: dialog/transcriptions/Ses01F_impro01.txt
#
#  Transcript file format:
#    Ses01F_impro01_F000 [6.29-8.23]: I had fun.
#    Ses01F_impro01_M001 [9.10-10.5]: Me too.
#
#  Saves: data/processed/text_embeddings/Ses01F_impro01_F000.npy
#         shape: [seq_len, 1024]   (was 768 with roberta-base)
#
#  ⚠️  IMPORTANT: Delete old 768-d .npy files first!
#      Remove-Item -Recurse -Force data\processed\text_embeddings\*
#
#  Downloads: ~1.4 GB on first run (roberta-large weights)
#  Time: ~20–40 min for all sessions on GPU
#  Run:  python preprocessing\extract_text.py
# ============================================================

import os
import re
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/text_embeddings"
MAX_LEN    = 128

print("\n  Loading RoBERTa-large (Tier 2 upgrade, downloads ~1.4 GB first time)...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model     = RobertaModel.from_pretrained("roberta-large")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"  Device: {device}")


def embed_text(text):
    """text → numpy [seq_len, 1024]  (RoBERTa-large hidden size)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=MAX_LEN, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.squeeze(0).cpu().numpy()  # [128, 1024]


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = skipped = 0

    for session_num in range(1, 6):
        trans_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                                 "dialog", "transcriptions")
        if not os.path.exists(trans_dir):
            print(f"  ⚠️  Session{session_num}: transcriptions/ not found")
            continue

        files = [f for f in os.listdir(trans_dir)
                 if f.endswith(".txt") and not f.startswith("._")]
        print(f"\n  Session{session_num} — {len(files)} transcript files")

        for fname in tqdm(files, desc=f"  Ses0{session_num}"):
            fpath = os.path.join(trans_dir, fname)
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue

            for line in lines:
                line = line.strip()
                # Format: SesXXX_dialog_F000 [start-end]: text here
                # Also accept lines without timestamp
                match = re.match(
                    r'^(Ses\d+[FM]_\S+)\s*(?:\[[\d\.\-]+\])?\s*:\s*(.+)$',
                    line)
                if not match:
                    continue

                utt_id = match.group(1).strip()
                text   = match.group(2).strip()
                if not text:
                    continue

                save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")
                if os.path.exists(save_path):
                    skipped += 1
                    continue

                try:
                    emb = embed_text(text)
                    np.save(save_path, emb)
                    saved += 1
                except Exception as e:
                    print(f"\n  ⚠️  {utt_id}: {e}")

    print(f"\n  ✅  Text done: {saved} saved, {skipped} already existed")
    print(f"      → {OUTPUT_DIR}")
    print("\n  Run next:")
    print("      python preprocessing\\extract_audio.py\n")


if __name__ == "__main__":
    extract_all()
