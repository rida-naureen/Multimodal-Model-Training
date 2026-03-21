# preprocessing/extract_audio.py
# ============================================================
#  STEP 3B — Extract AUDIO features using WavLM-Base+ (Tier 2 upgrade)
#
#  Tier 2 change: facebook/wav2vec2-base-960h (768-d)
#              →  microsoft/wavlm-base-plus   (1024-d)
#  WavLM adds masked speech + denoising pretraining for richer prosody.
#  Expected gain: +3–5 pp UA on IEMOCAP 4-class.
#
#  ⚠️  IMPORTANT: Delete old 768-d .npy files first!
#      Remove-Item -Recurse -Force data\processed\audio_embeddings\*
#
#  IEMOCAP stores audio per-utterance as .wav files:
#      Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
#
#  Strategy:
#    1. Walk all Session*/sentences/wav/ directories
#    2. Load each .wav with torchaudio
#    3. Resample to 16000 Hz (WavLM requirement)
#    4. Pass through WavLM-Base+ → extract last hidden states
#    5. Save as utterance-level .npy  shape: [T_a, 1024]
#
#  Saves: data/processed/audio_embeddings/Ses01F_impro01_F000.npy
#         shape: [T_a, 1024]   (was [T_a, 768] with wav2vec2-base)
#
#  Downloads: ~700 MB on first run (WavLM-Base+ weights)
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

print("\n  Loading WavLM-Base+ (Tier 2 upgrade, downloads ~700 MB first time)...")
processor = AutoProcessor.from_pretrained("microsoft/wavlm-base-plus")
wavlm     = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
wavlm.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavlm  = wavlm.to(device)
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
    wav_path → numpy array of shape [T_a, 1024].  (WavLM-Base+ hidden size)
    Returns None on error.
    """
    try:
        waveform = load_audio(wav_path)   # [T]
    except Exception as e:
        print(f"\n  ⚠️  Load error {wav_path}: {e}")
        return None

    # AutoProcessor normalises the raw waveform (same API as Wav2Vec2Processor)
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)   # [1, T]

    with torch.no_grad():
        outputs = wavlm(input_values)
        # last_hidden_state: [1, T_a, 1024]
        hidden = outputs.last_hidden_state.squeeze(0)   # [T_a, 1024]

    return hidden.cpu().numpy()


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
