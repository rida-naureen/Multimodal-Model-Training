# preprocessing/extract_audio.py
# ============================================================
#  STEP 3B — Extract AUDIO features using Wav2Vec2
#
#  What it does:
#    • Reads .wav files (how it was said — tone, pitch, rhythm)
#    • Resamples to 16kHz (required by Wav2Vec2)
#    • Passes through Wav2Vec2-base-960h
#    • Saves embedding as .npy file:
#        data/processed/audio_embeddings/Ses01F_impro01_F000.npy
#        shape: [time_frames, 768]
#
#  Time: ~30–40 minutes for all 5 sessions
#  Run:  python preprocessing/extract_audio.py
# ============================================================

import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

RAW_DIR           = "data/raw"
OUTPUT_DIR        = "data/processed/audio_embeddings"
TARGET_SAMPLERATE = 16000   # Wav2Vec2 requires 16kHz

# ── Load Wav2Vec2 (downloads ~360MB on first run) ─────────────
print("\n  Loading Wav2Vec2-base-960h...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"  Using device: {device}")


def extract_for_wav(wav_path):
    """
    wav_path → embedding numpy array [time_frames, 768]
    """
    waveform, sample_rate = torchaudio.load(wav_path)

    # Resample if needed
    if sample_rate != TARGET_SAMPLERATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLERATE)
        waveform  = resampler(waveform)

    # Convert stereo → mono
    waveform = waveform.mean(dim=0)   # [samples]

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLERATE,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: [1, T, 768] → [T, 768]
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()


def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0

    for session_num in range(1, 6):
        session  = f"Session{session_num}"
        wav_root = os.path.join(RAW_DIR, session, "sentences", "wav")

        if not os.path.exists(wav_root):
            print(f"  ⚠️   {session}: sentences/wav/ not found, skipping")
            continue

        # Collect all .wav files under subdirectories
        wav_files = []
        for subdir in os.listdir(wav_root):
            subdir_path = os.path.join(wav_root, subdir)
            if os.path.isdir(subdir_path):
                for wf in os.listdir(subdir_path):
                    if wf.endswith(".wav"):
                        wav_files.append(os.path.join(subdir_path, wf))

        print(f"\n  {session} — {len(wav_files)} WAV files")

        for wav_path in tqdm(wav_files, desc=f"  {session}"):
            utt_id    = os.path.splitext(os.path.basename(wav_path))[0]
            save_path = os.path.join(OUTPUT_DIR, f"{utt_id}.npy")

            if os.path.exists(save_path):
                continue   # skip already extracted

            try:
                embedding = extract_for_wav(wav_path)
                np.save(save_path, embedding)
                total_saved += 1
            except Exception as e:
                print(f"\n  ⚠️   Skipped {utt_id}: {e}")

    print(f"\n  ✅  Audio extraction complete: {total_saved} files saved")
    print(f"      Location: {OUTPUT_DIR}")
    print("\n  Run next:")
    print("      python preprocessing/extract_visual.py\n")


if __name__ == "__main__":
    extract_all()
