# preprocessing/build_dataset.py
# ============================================================
#  OPTIONAL — Runs all preprocessing steps in sequence.
#
#  Instead of running 4 scripts separately, run just this one:
#      python preprocessing/build_dataset.py
#
#  Order:
#    1. check_data.py    — verify dataset
#    2. create_splits.py — make train/val/test lists
#    3. extract_text.py  — RoBERTa embeddings
#    4. extract_audio.py — Wav2Vec2 embeddings
#    5. extract_visual.py— ResNet embeddings
# ============================================================

import subprocess
import sys

STEPS = [
    ("Verifying dataset",          "preprocessing/check_data.py"),
    ("Creating splits",            "preprocessing/create_splits.py"),
    ("Extracting text features",   "preprocessing/extract_text.py"),
    ("Extracting audio features",  "preprocessing/extract_audio.py"),
    ("Extracting visual features", "preprocessing/extract_visual.py"),
]

print("\n" + "=" * 55)
print("  IEMOCAP Full Preprocessing Pipeline")
print("=" * 55)

for i, (desc, script) in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {desc}...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n❌  Step failed: {script}")
        print("    Fix the error above, then re-run this script.")
        sys.exit(1)

print("\n" + "=" * 55)
print("  ✅  All preprocessing complete!")
print("  Run next:  python train.py")
print("=" * 55 + "\n")
