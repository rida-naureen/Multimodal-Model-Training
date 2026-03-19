# dataset/iemocap_dataset.py
# ============================================================
#  PyTorch Dataset — loads pre-extracted .npy files.
#
#  Used by train.py and evaluate.py like this:
#    train_set = IEMOCAPDataset("train", ...)
#    loader    = DataLoader(train_set, batch_size=16,
#                           collate_fn=collate_fn)
# ============================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Map emotion string → integer class index
# "exc" (excited) is merged into "hap" (happy) — standard 4-class IEMOCAP setup
EMOTION_TO_IDX = {
    "hap": 0,   # happy
    "exc": 0,   # excited → treated as happy
    "sad": 1,   # sad
    "ang": 2,   # angry
    "neu": 3    # neutral
}
IDX_TO_EMOTION = {0: "hap", 1: "sad", 2: "ang", 3: "neu"}  # 0=hap (exc merged into hap)


def load_label_map(splits_dir):
    """
    Reads data/splits/labels.txt
    Returns dict: { "Ses01F_impro01_F000": "hap", ... }
    """
    label_map = {}
    path = os.path.join(splits_dir, "labels.txt")
    with open(path, encoding="latin-1") as f:  # latin-1 decodes any byte → no UnicodeDecodeError
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    label_map[parts[0]] = parts[1]
    return label_map


class IEMOCAPDataset(Dataset):
    """
    Loads text + audio + visual embeddings for a given split.

    Args:
        split      : "train", "val", or "test"
        splits_dir : "data/splits"
        text_dir   : "data/processed/text_embeddings"
        audio_dir  : "data/processed/audio_embeddings"
        visual_dir : "data/processed/visual_embeddings"
    """

    def __init__(self, split, splits_dir, text_dir, audio_dir, visual_dir):
        # ── Load utterance IDs for this split ─────────────────
        split_file = os.path.join(splits_dir, f"{split}_ids.txt")
        with open(split_file, encoding="latin-1") as f:  # latin-1 safe for IEMOCAP splits
            all_ids = [l.strip() for l in f if l.strip()]

        self.label_map  = load_label_map(splits_dir)
        self.text_dir   = text_dir
        self.audio_dir  = audio_dir
        self.visual_dir = visual_dir

        # ── Keep only utterances where text + audio .npy files exist ─
        # Visual embeddings are optional: if missing, a zero tensor is used.
        valid, skipped = [], 0
        missing_visual = 0
        for uid in all_ids:
            has_text   = os.path.exists(os.path.join(text_dir,   f"{uid}.npy"))
            has_audio  = os.path.exists(os.path.join(audio_dir,  f"{uid}.npy"))
            has_visual = os.path.exists(os.path.join(visual_dir, f"{uid}.npy"))
            has_label  = uid in self.label_map
            if has_text and has_audio and has_label:
                valid.append(uid)
                if not has_visual:
                    missing_visual += 1
            else:
                skipped += 1

        if skipped > 0:
            print(f"  [{split}] ⚠️  Skipped {skipped} utterances "
                  f"(missing text/audio .npy or label)")
        if missing_visual > 0:
            print(f"  [{split}] ℹ️  {missing_visual} utterances missing visual .npy "
                  f"→ zero tensors will be used")
        print(f"  [{split}] ✅  {len(valid)} utterances ready")
        self.utt_ids = valid

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]

        # Load pre-extracted features
        text   = np.load(os.path.join(self.text_dir,   f"{uid}.npy"))  # [T_t, 768]
        audio  = np.load(os.path.join(self.audio_dir,  f"{uid}.npy"))  # [T_a, 768]
        visual_path = os.path.join(self.visual_dir, f"{uid}.npy")
        if os.path.exists(visual_path):
            visual = np.load(visual_path)                               # [30, 256]
        else:
            visual = np.zeros((30, 256), dtype=np.float32)             # zero fallback

        emotion = self.label_map[uid]
        label   = EMOTION_TO_IDX[emotion]

        return {
            "utt_id": uid,
            "text":   torch.tensor(text,   dtype=torch.float32),
            "audio":  torch.tensor(audio,  dtype=torch.float32),
            "visual": torch.tensor(visual, dtype=torch.float32),
            "label":  torch.tensor(label,  dtype=torch.long)
        }


def collate_fn(batch):
    """
    Called automatically by DataLoader to combine a list of samples
    into one batch.

    Because text and audio have different lengths per utterance,
    we pad them to the longest in the batch.

    Returns tensors of shape:
      text:   [B, T_t_max, 768]
      audio:  [B, T_a_max, 768]
      visual: [B, 30, 256]       (already fixed length)
      label:  [B]
    """
    texts   = [item["text"]   for item in batch]
    audios  = [item["audio"]  for item in batch]
    visuals = [item["visual"] for item in batch]
    labels  = torch.stack([item["label"]  for item in batch])
    utt_ids = [item["utt_id"] for item in batch]

    # Pad variable-length sequences
    texts_padded   = pad_sequence(texts,   batch_first=True)   # [B, T_t, 768]
    audios_padded  = pad_sequence(audios,  batch_first=True)   # [B, T_a, 768]
    visuals_padded = torch.stack(visuals)                      # [B, 30,  256]

    # Padding masks: True = this position is padding (ignore in attention)
    def make_mask(original_seqs, padded_tensor):
        B, T, _ = padded_tensor.shape
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, seq in enumerate(original_seqs):
            mask[i, len(seq):] = True   # positions beyond real length → True
        return mask

    text_mask   = make_mask(texts,  texts_padded)
    audio_mask  = make_mask(audios, audios_padded)
    # visual is fixed length (30 frames) → no padding needed
    visual_mask = torch.zeros(len(batch), visuals_padded.shape[1], dtype=torch.bool)

    return {
        "utt_ids":     utt_ids,
        "text":        texts_padded,
        "audio":       audios_padded,
        "visual":      visuals_padded,
        "text_mask":   text_mask,
        "audio_mask":  audio_mask,
        "visual_mask": visual_mask,
        "label":       labels
    }