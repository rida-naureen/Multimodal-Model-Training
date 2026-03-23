# preprocessing/extract_audio.py
# ============================================================
#  Extract AUDIO features using WavLM-Base+
#
#  Encoder : microsoft/wavlm-base-plus
#  Output  : [T_a, 1024]  per utterance  (variable T_a)
#  Saves to: data/processed/audio_embeddings/<utt_id>.npy
#
#  Reads .wav files:
#    data/raw/SessionX/sentences/wav/<dialog>/<utt_id>.wav
#  Resampled to 16 kHz before encoding.
#
#  Downloads: ~700 MB on first run
#  Time: ~30–60 min for all sessions on GPU
#  Run:  python preprocessing\extract_audio.py
# ============================================================

import os
import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

RAW_DIR    = "data/raw"
OUTPUT_DIR = "data/processed/audio_embeddings"
TARGET_SR  = 16000   # WavLM requires 16 kHz

from preprocessing.encoders import get_audio_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")


def load_audio(wav_path):
    """
    Load a .wav file and resample to TARGET_SR.
    Returns 1-D float32 tensor at 16 kHz.
    """
    waveform, sr = torchaudio.load(wav_path)
    # Convert stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0)   # [T]


def embed_audio(wav_path):
    """
    wav_path → numpy array of shape (768,).
    Returns None on error.
    """
    try:
        waveform = load_audio(wav_path)   # [T]
    except Exception as e:
        print(f"\n  ⚠️  Load error {wav_path}: {e}")
        return None

    # Use singleton encoder
    encoder = get_audio_encoder()
    emb = encoder.encode(waveform.numpy())
    return emb


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = skipped = failed = 0

    for session_num in range(1, 6):
        wav_root = os.path.join(RAW_DIR, f"Session{session_num}",
                                "sentences", "wav")
        if not os.path.exists(wav_root):
            print(f"  ⚠️  Session{session_num}: sentences/wav/ not found — skipping")
            continue

        # Collect all .wav paths
        wav_paths = []
        for dialog_dir in os.listdir(wav_root):
            dialog_path = os.path.join(wav_root, dialog_dir)
            if not os.path.isdir(dialog_path):
                continue
            for fname in os.listdir(dialog_path):
                if fname.endswith(".wav") and not fname.startswith("._"):
                    utt_id = os.path.splitext(fname)[0]   # e.g. Ses01F_impro01_F000
                    wav_paths.append((utt_id, os.path.join(dialog_path, fname)))

        print(f"\n  Session{session_num} — {len(wav_paths)} utterances")

        for utt_id, wav_path in tqdm(wav_paths, desc=f"  Ses0{session_num}"):
            save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")
            if os.path.exists(save_path):
                skipped += 1
                continue

            emb = embed_audio(wav_path)
            if emb is not None:
                np.save(save_path, emb)
                saved += 1
            else:
                failed += 1

    print(f"\n  ✅  Audio done: {saved} saved, {skipped} existed, {failed} failed")
    print(f"      → {OUTPUT_DIR}")
    print("\n  Run next:")
    print("      python preprocessing\\extract_visual.py\n")


if __name__ == "__main__":
    extract_all()
