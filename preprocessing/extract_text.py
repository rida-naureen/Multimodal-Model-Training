# preprocessing/extract_text.py
# ============================================================
#  STEP 3A — Extract TEXT features using RoBERTa
#
#  What it does:
#    • Reads transcription .txt files (what was said)
#    • Passes each utterance through RoBERTa-base
#    • Saves embedding as .npy file:
#        data/processed/text_embeddings/Ses01F_impro01_F000.npy
#        shape: [seq_len, 768]
#
#  Time: ~20–30 minutes for all 5 sessions
#  Run:  python preprocessing/extract_text.py
# ============================================================

import os
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/text_embeddings"
MAX_LEN    = 128   # max token length per utterance

# ── Load RoBERTa (downloads ~500MB on first run) ──────────────
print("\n  Loading RoBERTa-base...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model     = RobertaModel.from_pretrained("roberta-base")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"  Using device: {device}")


def extract_for_utterance(text):
    """
    text (str) → embedding numpy array [seq_len, 768]
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: [1, seq_len, 768] → [seq_len, 768]
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0

    for session_num in range(1, 6):
        session    = f"Session{session_num}"
        trans_dir  = os.path.join(RAW_DIR, session, "dialog", "transcriptions")

        if not os.path.exists(trans_dir):
            print(f"  ⚠️   {session}: transcriptions/ not found, skipping")
            continue

        files = [f for f in os.listdir(trans_dir) if f.endswith(".txt")]
        print(f"\n  {session} — {len(files)} transcription files")

        for fname in tqdm(files, desc=f"  {session}"):
            fpath = os.path.join(trans_dir, fname)
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue

                    # Line format:
                    # Ses01F_impro01_F000 [0.00-1.50]: Hello there
                    try:
                        header, text = line.split(":", 1)
                        utt_id = header.split(" ")[0].strip()
                        text   = text.strip()
                    except ValueError:
                        continue

                    if not text or not utt_id:
                        continue

                    save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")
                    if os.path.exists(save_path):
                        continue   # skip already extracted

                    embedding = extract_for_utterance(text)
                    np.save(save_path, embedding)
                    total_saved += 1

    print(f"\n  ✅  Text extraction complete: {total_saved} files saved")
    print(f"      Location: {OUTPUT_DIR}")
    print("\n  Run next:")
    print("      python preprocessing/extract_audio.py\n")


if __name__ == "__main__":
    extract_all()
