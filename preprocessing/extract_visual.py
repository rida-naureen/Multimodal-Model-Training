# preprocessing/extract_audio.py
# ============================================================
#  STEP 3B — Extract AUDIO features using Wav2Vec2-base-960h
#
#  Reads: sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
#
#  Saves: data/processed/audio_embeddings/Ses01F_impro01_F000.npy
#         shape: [time_frames, 768]
#
#  Time: ~30–40 min for all sessions
#  Run:  python preprocessing\extract_audio.py
# ============================================================

import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

RAW_DIR           = "data/raw"
OUTPUT_DIR        = "data/processed/audio_embeddings"
TARGET_SR         = 16000

print("\n  Loading Wav2Vec2-base-960h (downloads ~360MB first time)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"  Device: {device}")


def embed_wav(wav_path):
    """wav path → numpy [time_frames, 768]"""
    waveform, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    waveform = waveform.mean(dim=0)  # stereo → mono

    inputs = processor(waveform.numpy(), sampling_rate=TARGET_SR,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 768]


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = skipped = 0

    for session_num in range(1, 6):
        # Utterance WAVs are in sentences/wav/SesXXX_dialog/SesXXX_F000.wav
        wav_root = os.path.join(RAW_DIR, f"Session{session_num}",
                                "sentences", "wav")
        if not os.path.exists(wav_root):
            print(f"  ⚠️  Session{session_num}: sentences/wav/ not found")
            continue

        # Collect all .wav files across all subdirectories
        wav_files = []
        for subdir in sorted(os.listdir(wav_root)):
            subpath = os.path.join(wav_root, subdir)
            if os.path.isdir(subpath):
                for wf in sorted(os.listdir(subpath)):
                    if wf.endswith(".wav") and not wf.startswith("._"):
                        wav_files.append(os.path.join(subpath, wf))

        print(f"\n  Session{session_num} — {len(wav_files)} utterance WAV files")

        for wav_path in tqdm(wav_files, desc=f"  Ses0{session_num}"):
            utt_id    = os.path.splitext(os.path.basename(wav_path))[0]
            save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")

            if os.path.exists(save_path):
                skipped += 1
                continue

            try:
                emb = embed_wav(wav_path)
                np.save(save_path, emb)
                saved += 1
            except Exception as e:
                print(f"\n  ⚠️  {utt_id}: {e}")

    print(f"\n  ✅  Audio done: {saved} saved, {skipped} already existed")
    print(f"      → {OUTPUT_DIR}")
    print("\n  Run next:")
    print("      python preprocessing\\extract_visual.py\n")


if __name__ == "__main__":
    extract_all()
