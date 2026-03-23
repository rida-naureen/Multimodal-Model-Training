# preprocessing/extract_text.py
# ============================================================
#  Extract TEXT features using RoBERTa-large
#
#  Encoder : roberta-large (~355M params)
#  Output  : [seq_len, 1024]  per utterance
#  Saves to: data/processed/text_embeddings/<utt_id>.npy
#
#  Reads transcript files:
#    data/raw/SessionX/dialog/transcriptions/*.txt
#  Format:
#    Ses01F_impro01_F000 [6.29-8.23]: I had fun.
#
#  Downloads: ~1.4 GB on first run
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

from preprocessing.encoders import get_text_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")


def embed_text_from_wav(wav_path):
    """wav_path → (transcript, features_numpy_768)"""
    pipeline = get_text_encoder()
    result = pipeline.process(wav_path)
    return result["transcript"], np.array(result["text_features"], dtype=np.float32)


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
            dialog_id = fname.replace(".txt", "")
            
            # Find wav directory for this dialog
            wav_dir = os.path.join(RAW_DIR, f"Session{session_num}", 
                                   "sentences", "wav", dialog_id)
            if not os.path.exists(wav_dir):
                continue
                
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue

            for line in lines:
                line = line.strip()
                match = re.match(
                    r'^(Ses\d+[FM]_\S+)\s*(?:\[[\d\.\-]+\])?\s*:\s*(.+)$',
                    line)
                if not match:
                    continue

                utt_id = match.group(1).strip()
                save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")
                if os.path.exists(save_path):
                    skipped += 1
                    continue

                # Locate the wav file for this utterance
                wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
                if not os.path.exists(wav_path):
                    continue

                try:
                    # Use Whisper + RoBERTa pipeline
                    transcript, emb = embed_text_from_wav(wav_path)
                    if emb is not None:
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
