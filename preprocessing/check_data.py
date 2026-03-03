# preprocessing/check_data.py
# ============================================================
#  STEP 1 — Run this FIRST to verify your IEMOCAP dataset
#  is correctly placed before doing anything else.
#
#  Run:  python preprocessing/check_data.py
# ============================================================

import os

RAW_DIR  = "data/raw"
SESSIONS = [f"Session{i}" for i in range(1, 6)]
VALID_EMOTIONS = {"hap", "exc", "sad", "ang", "neu"}

print("=" * 60)
print("  IEMOCAP Dataset Verification")
print("=" * 60)

total_wav, total_avi, total_labels = 0, 0, 0

for session in SESSIONS:
    session_path = os.path.join(RAW_DIR, session)

    if not os.path.exists(session_path):
        print(f"\n❌  {session} — NOT FOUND at {session_path}")
        continue

    print(f"\n📁  {session}")

    # ── Check WAV files ──────────────────────────────────────
    wav_root  = os.path.join(session_path, "sentences", "wav")
    wav_count = 0
    if os.path.exists(wav_root):
        for sub in os.listdir(wav_root):
            sub_path = os.path.join(wav_root, sub)
            if os.path.isdir(sub_path):
                wav_count += len([f for f in os.listdir(sub_path)
                                   if f.endswith(".wav")])
        print(f"     ✅  WAV files  : {wav_count}")
        total_wav += wav_count
    else:
        print(f"     ❌  sentences/wav/ NOT FOUND")

    # ── Check AVI files ──────────────────────────────────────
    avi_root  = os.path.join(session_path, "sentences", "avi")
    avi_count = 0
    if os.path.exists(avi_root):
        for sub in os.listdir(avi_root):
            sub_path = os.path.join(avi_root, sub)
            if os.path.isdir(sub_path):
                avi_count += len([f for f in os.listdir(sub_path)
                                   if f.endswith(".avi")])
        print(f"     ✅  AVI files  : {avi_count}")
        total_avi += avi_count
    else:
        print(f"     ❌  sentences/avi/ NOT FOUND")

    # ── Check Emotion Labels ─────────────────────────────────
    emo_dir     = os.path.join(session_path, "dialog", "EmoEvaluation")
    label_count = 0
    emo_counts  = {}
    if os.path.exists(emo_dir):
        for fname in os.listdir(emo_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(emo_dir, fname), encoding="utf-8") as f:
                for line in f:
                    if line.startswith("["):
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            emo = parts[2].strip()
                            if emo in VALID_EMOTIONS:
                                label_count += 1
                                emo_counts[emo] = emo_counts.get(emo, 0) + 1
        print(f"     ✅  Labels     : {label_count}  →  {emo_counts}")
        total_labels += label_count
    else:
        print(f"     ❌  dialog/EmoEvaluation/ NOT FOUND")

    # ── Check Transcriptions ─────────────────────────────────
    trans_dir = os.path.join(session_path, "dialog", "transcriptions")
    if os.path.exists(trans_dir):
        n = len([f for f in os.listdir(trans_dir) if f.endswith(".txt")])
        print(f"     ✅  Transcripts: {n} files")
    else:
        print(f"     ❌  dialog/transcriptions/ NOT FOUND")

print("\n" + "=" * 60)
print(f"  TOTAL WAV    : {total_wav}")
print(f"  TOTAL AVI    : {total_avi}")
print(f"  TOTAL Labels : {total_labels}")
print("=" * 60)

if total_wav > 0 and total_labels > 0:
    print("\n  ✅  Dataset looks good! Run next:")
    print("      python preprocessing/create_splits.py\n")
else:
    print("\n  ❌  Issues found — fix folder structure first.\n")
