# preprocessing/check_data.py
# ============================================================
#  STEP 1 — Verifies IEMOCAP_full_release matches expected structure.
#
#  Your actual IEMOCAP structure:
#
#  IEMOCAP_full_release/
#    Session1/
#      dialog/
#        avi/DivX/          ← dialog-level AVI  (Ses01F_impro01.avi)
#        EmoEvaluation/     ← emotion label TXT (Ses01F_impro01.txt)
#        transcriptions/    ← transcript TXT    (Ses01F_impro01.txt)
#        wav/               ← dialog-level WAV  (not used)
#      sentences/
#        wav/
#          Ses01F_impro01/  ← utterance WAVs    (Ses01F_impro01_F000.wav)
#
#  ⚠️  There is NO sentences/avi/ folder.
#      Visual features are extracted from dialog/avi/DivX/ instead.
#
#  Run:  python preprocessing\check_data.py
# ============================================================

import os

RAW_DIR        = "data/raw"
SESSIONS       = [f"Session{i}" for i in range(1, 6)]
VALID_EMOTIONS = {"hap", "exc", "sad", "ang", "neu"}

print("=" * 65)
print("  IEMOCAP Dataset Check  (IEMOCAP_full_release layout)")
print("=" * 65)

total_wav = total_avi = total_labels = total_trans = 0

for session in SESSIONS:
    spath = os.path.join(RAW_DIR, session)

    if not os.path.exists(spath):
        print(f"\n❌  {session}  NOT FOUND at: {os.path.abspath(spath)}")
        print(f"    → Copy your Session folders into data/raw/")
        continue

    print(f"\n📁  {session}")

    # ── Utterance WAVs: sentences/wav/SesXXX_dialog/SesXXX_F000.wav ──
    wav_root  = os.path.join(spath, "sentences", "wav")
    wav_count = 0
    if os.path.exists(wav_root):
        for subdir in os.listdir(wav_root):
            sub = os.path.join(wav_root, subdir)
            if os.path.isdir(sub):
                wav_count += len([f for f in os.listdir(sub)
                                   if f.endswith(".wav") and not f.startswith("._")])
        total_wav += wav_count
        icon = "✅" if wav_count > 0 else "❌"
        print(f"   {icon}  sentences/wav/          : {wav_count} utterance WAV files")
    else:
        print(f"   ❌  sentences/wav/          : NOT FOUND")

    # ── Dialog-level AVI: dialog/avi/DivX/SesXXX_dialog.avi ────────
    avi_root  = os.path.join(spath, "dialog", "avi", "DivX")
    avi_count = 0
    if os.path.exists(avi_root):
        avis      = [f for f in os.listdir(avi_root)
                     if f.endswith(".avi") and not f.startswith("._")]
        avi_count = len(avis)
        total_avi += avi_count
        icon = "✅" if avi_count > 0 else "❌"
        sample = f"  (e.g. {avis[0]})" if avis else ""
        print(f"   {icon}  dialog/avi/DivX/         : {avi_count} dialog AVI files{sample}")
    else:
        print(f"   ❌  dialog/avi/DivX/         : NOT FOUND")

    # ── Emotion Labels: dialog/EmoEvaluation/*.txt ──────────────────
    emo_dir     = os.path.join(spath, "dialog", "EmoEvaluation")
    label_count = 0
    emo_counts  = {}
    if os.path.exists(emo_dir):
        for fname in os.listdir(emo_dir):
            if not fname.endswith(".txt") or fname.startswith("._"):
                continue
            try:
                with open(os.path.join(emo_dir, fname),
                          encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("["):
                            parts = line.strip().split("\t")
                            if len(parts) >= 3:
                                emo = parts[2].strip()
                                if emo in VALID_EMOTIONS:
                                    label_count += 1
                                    emo_counts[emo] = emo_counts.get(emo, 0) + 1
            except Exception:
                continue
        total_labels += label_count
        icon = "✅" if label_count > 0 else "❌"
        print(f"   {icon}  dialog/EmoEvaluation/   : {label_count} labels  {emo_counts}")
    else:
        print(f"   ❌  dialog/EmoEvaluation/   : NOT FOUND")

    # ── Transcriptions: dialog/transcriptions/*.txt ─────────────────
    trans_dir = os.path.join(spath, "dialog", "transcriptions")
    if os.path.exists(trans_dir):
        tfiles = [f for f in os.listdir(trans_dir)
                  if f.endswith(".txt") and not f.startswith("._")]
        total_trans += len(tfiles)
        print(f"   ✅  dialog/transcriptions/  : {len(tfiles)} transcript files")
    else:
        print(f"   ❌  dialog/transcriptions/  : NOT FOUND")

print("\n" + "=" * 65)
print(f"  Utterance WAV files    : {total_wav}")
print(f"  Dialog AVI files       : {total_avi}")
print(f"  Emotion labels         : {total_labels}")
print(f"  Transcript files       : {total_trans}")
print("=" * 65)

if total_wav > 0 and total_labels > 0 and total_trans > 0:
    print("\n  ✅  All good! Run next:")
    print("      python preprocessing\\create_splits.py\n")
else:
    print("\n  ❌  Issues found — check paths above.")
    print("\n  Expected layout inside data/raw/Session1/:")
    print("    sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav")
    print("    dialog/avi/DivX/Ses01F_impro01.avi")
    print("    dialog/EmoEvaluation/Ses01F_impro01.txt")
    print("    dialog/transcriptions/Ses01F_impro01.txt\n")
