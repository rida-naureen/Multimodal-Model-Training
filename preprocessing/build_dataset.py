# preprocessing/build_dataset.py
# ============================================================
#  Runs all preprocessing steps in one go (SERVER use).
#  Run:  python preprocessing\build_dataset.py
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
print("  ET-TACFN Full Preprocessing Pipeline")
print("=" * 55)

for i, (desc, script) in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {desc}...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n❌ Failed at step {i}: {script}")
        print("   Fix the error above, then re-run this script.")
        sys.exit(1)

print("\n" + "=" * 55)
print("  ✅ All preprocessing complete!")
print("  Run: python train.py")
print("=" * 55 + "\n")
