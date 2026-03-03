# preprocessing/create_splits.py
# ============================================================
#  STEP 2 — Creates train / val / test split files.
#
#  What it does:
#    • Reads all emotion labels from EmoEvaluation .txt files
#    • Session 5  → test set   (speaker-independent evaluation)
#    • Sessions 1-4 → 90% train / 10% val
#    • Merges "exc" (excited) into "hap" (happy)  ← standard practice
#    • Saves:
#        data/splits/train_ids.txt
#        data/splits/val_ids.txt
#        data/splits/test_ids.txt
#        data/splits/labels.txt     ← maps every utt_id → emotion
#
#  Run:  python preprocessing/create_splits.py
# ============================================================

import os
import random

RAW_DIR        = "data/raw"
SPLITS_DIR     = "data/splits"
VALID_EMOTIONS = {"hap", "exc", "sad", "ang", "neu"}
TEST_SESSION   = 5
VAL_RATIO      = 0.1
SEED           = 42


def parse_all_labels():
    """
    Reads EmoEvaluation .txt files for all 5 sessions.
    Returns dict:  { "Ses01F_impro01_F000": "hap",  ... }
    """
    labels = {}
    for session_num in range(1, 6):
        emo_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                               "dialog", "EmoEvaluation")
        if not os.path.exists(emo_dir):
            print(f"  ❌  Not found: {emo_dir}")
            continue

        for fname in os.listdir(emo_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(emo_dir, fname), encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    utt_id  = parts[1].strip()
                    emotion = parts[2].strip()
                    if emotion in VALID_EMOTIONS:
                        # Merge excited → happy (standard 4-class setup)
                        labels[utt_id] = "hap" if emotion == "exc" else emotion
    return labels


def create_splits():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print("\n  Reading emotion labels from all sessions...")
    labels = parse_all_labels()
    print(f"  Total valid utterances found: {len(labels)}")

    # Count per emotion
    from collections import Counter
    print(f"  Emotion counts: {dict(Counter(labels.values()))}")

    # ── Separate Session 5 as test ────────────────────────────
    # Session 5 utterance IDs contain "Ses05"
    test_ids  = [uid for uid in labels if "Ses05" in uid]
    other_ids = [uid for uid in labels if "Ses05" not in uid]

    # ── Split sessions 1-4 into train / val ──────────────────
    random.seed(SEED)
    random.shuffle(other_ids)
    n_val     = int(len(other_ids) * VAL_RATIO)
    val_ids   = other_ids[:n_val]
    train_ids = other_ids[n_val:]

    # ── Save split files ──────────────────────────────────────
    print()
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = os.path.join(SPLITS_DIR, f"{name}_ids.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))
        print(f"  ✅  {name:5s}: {len(ids):5d} utterances  →  {path}")

    # ── Save label lookup ─────────────────────────────────────
    label_path = os.path.join(SPLITS_DIR, "labels.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        for uid, emo in labels.items():
            f.write(f"{uid}\t{emo}\n")
    print(f"  ✅  labels.txt saved  →  {label_path}")
    print("\n  ✅  Done! Run next:")
    print("      python preprocessing/extract_text.py\n")


if __name__ == "__main__":
    create_splits()
