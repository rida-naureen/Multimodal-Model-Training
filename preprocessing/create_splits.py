# preprocessing/create_splits.py
# ============================================================
#  STEP 2 — Creates train / val / test split files.
#
#  Reads emotion labels from dialog/EmoEvaluation/*.txt
#  Session 5  → test  (speaker-independent LOSO evaluation)
#  Sessions 1–4 → 90% train / 10% val
#  Merges "exc" (excited) → "hap" (happy)  ← standard practice
#
#  Saves:
#    data/splits/train_ids.txt
#    data/splits/val_ids.txt
#    data/splits/test_ids.txt
#    data/splits/labels.txt    ← maps every utt_id → emotion
#
#  Run:  python preprocessing\create_splits.py
# ============================================================

import os
import random
from collections import Counter

RAW_DIR        = "data/raw"
SPLITS_DIR     = "data/splits"
VALID_EMOTIONS = {"hap", "exc", "sad", "ang", "neu"}
TEST_SESSION   = 5
VAL_RATIO      = 0.1
SEED           = 42


def parse_all_labels():
    """
    Reads EmoEvaluation .txt files for all 5 sessions.
    Returns dict: { "Ses01F_impro01_F000": "hap", ... }

    Label file format (one line per utterance):
      [6.2901 - 8.2357]	Ses01F_impro01_F000	hap	[2.5, 2.5, 2.5]
    """
    labels = {}

    for session_num in range(1, 6):
        emo_dir = os.path.join(RAW_DIR, f"Session{session_num}",
                               "dialog", "EmoEvaluation")

        if not os.path.exists(emo_dir):
            print(f"  ⚠️  Not found: {emo_dir}")
            continue

        for fname in os.listdir(emo_dir):
            if not fname.endswith(".txt") or fname.startswith("._"):
                continue
            fpath = os.path.join(emo_dir, fname)
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line.startswith("["):
                            continue
                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue
                        utt_id  = parts[1].strip()
                        emotion = parts[2].strip()
                        if emotion in VALID_EMOTIONS:
                            # Merge excited → happy (standard 4-class IEMOCAP)
                            labels[utt_id] = "hap" if emotion == "exc" else emotion
            except Exception as e:
                print(f"  ⚠️  Skipped {fname}: {e}")

    return labels


def create_splits():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print("\n  Reading labels from all 5 sessions...")
    labels = parse_all_labels()
    print(f"  Total valid utterances : {len(labels)}")
    print(f"  Emotion counts         : {dict(Counter(labels.values()))}")

    # Session 5 → test
    test_ids  = [uid for uid in labels if "Ses05" in uid]
    other_ids = [uid for uid in labels if "Ses05" not in uid]

    # ── Dialogue-level split (prevents data leakage) ──────────────────────────
    # Extract unique dialogue IDs (e.g. "Ses01F_impro01") from utterance IDs.
    # All utterances from the same dialogue must go to the SAME split.
    # Shuffling dialogues, not individual utterances, eliminates cross-dialogue leakage.
    from collections import defaultdict
    dialogue_to_utts = defaultdict(list)
    for uid in other_ids:
        # Dialogue ID is everything before the last underscore+speaker+index
        # e.g. "Ses01F_impro01_F001" → "Ses01F_impro01"
        dialogue_id = "_".join(uid.split("_")[:-1])
        dialogue_to_utts[dialogue_id].append(uid)

    dialogue_ids = sorted(dialogue_to_utts.keys())
    random.seed(SEED)
    random.shuffle(dialogue_ids)

    n_val_dialogues = max(1, int(len(dialogue_ids) * VAL_RATIO))
    val_dialogues   = set(dialogue_ids[:n_val_dialogues])
    train_dialogues = set(dialogue_ids[n_val_dialogues:])

    val_ids   = [uid for d in val_dialogues   for uid in dialogue_to_utts[d]]
    train_ids = [uid for d in train_dialogues for uid in dialogue_to_utts[d]]

    print()
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = os.path.join(SPLITS_DIR, f"{name}_ids.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))
        print(f"  ✅  {name:5s}: {len(ids):5d} utterances  →  {path}")

    # Save label lookup
    label_path = os.path.join(SPLITS_DIR, "labels.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        for uid, emo in labels.items():
            f.write(f"{uid}\t{emo}\n")
    print(f"  ✅  labels.txt          →  {label_path}")
    print("\n  Run next:")
    print("      python preprocessing\\extract_text.py\n")


if __name__ == "__main__":
    create_splits()
