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
        cfg        : full config dict (used for Tier 3 context window)
    """

    def __init__(self, split, splits_dir, text_dir, audio_dir, visual_dir, cfg=None):
        # ── Load utterance IDs for this split ───────────────────
        split_file = os.path.join(splits_dir, f"{split}_ids.txt")
        with open(split_file, encoding="latin-1") as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self.label_map  = load_label_map(splits_dir)
        self.text_dir   = text_dir
        self.audio_dir  = audio_dir
        self.visual_dir = visual_dir
        self.cfg        = cfg or {}

        # ── Keep only utterances where text + audio .npy files exist ────
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

        # ── Sort by dialogue + utterance order for context window ─────
        # IEMOCAP IDs: Ses01F_impro01_F000 — sort lexicographically keeps
        # dialogue session and utterance order intact.
        self.utt_ids = sorted(valid)

    def __len__(self):
        return len(self.utt_ids)

    # ── Helper: load individual modality features ────────────────
    def _load_text(self, uid):
        return torch.tensor(
            np.load(os.path.join(self.text_dir, f"{uid}.npy")),
            dtype=torch.float32)

    def _load_audio(self, uid):
        return torch.tensor(
            np.load(os.path.join(self.audio_dir, f"{uid}.npy")),
            dtype=torch.float32)

    def _load_visual(self, uid):
        path = os.path.join(self.visual_dir, f"{uid}.npy")
        if os.path.exists(path):
            return torch.tensor(np.load(path), dtype=torch.float32)
        return torch.zeros(30, 256, dtype=torch.float32)   # fallback

    # ── Helper: dialogue boundary check for context window ───
    @staticmethod
    def _same_dialogue(uid_a: str, uid_b: str) -> bool:
        """True when both utterances belong to the same IEMOCAP dialogue.
        IEMOCAP ID format: Ses01F_impro01_F000
        Dialogue = 'Ses01F_impro01' (everything before the last _SPEAKER+number)
        """
        def _dialogue_id(uid):
            # Split on '_' and drop the LAST part (speaker+index, e.g. F000)
            parts = uid.split('_')
            return '_'.join(parts[:-1])
        return _dialogue_id(uid_a) == _dialogue_id(uid_b)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]

        # Load pre-extracted features
        text   = self._load_text(uid)    # [T_t, text_dim]
        audio  = self._load_audio(uid)   # [T_a, audio_dim]
        visual = self._load_visual(uid)  # [30,  visual_dim]

        emotion = self.label_map[uid]
        label   = EMOTION_TO_IDX[emotion]

        sample = {
            "utt_id": uid,
            "text":   text,
            "audio":  audio,
            "visual": visual,
            "label":  torch.tensor(label, dtype=torch.long)
        }

        # ── Tier 3: Build 5-utterance context window ────────────
        # Only when use_conversation_context is enabled in config.
        use_ctx = (self.cfg.get("model", {})
                       .get("use_conversation_context", False))
        if use_ctx:
            d_model = self.cfg["model"]["d_model"]
            window_feats = []
            for offset in range(-2, 3):            # offsets: -2, -1, 0, +1, +2
                nb_idx = idx + offset
                # Use zero vector if outside this dialogue or out of bounds
                if (0 <= nb_idx < len(self.utt_ids) and
                        self._same_dialogue(self.utt_ids[nb_idx], uid)):
                    nb_uid  = self.utt_ids[nb_idx]
                    nb_t    = self._load_text(nb_uid).mean(0)    # [text_dim]
                    nb_a    = self._load_audio(nb_uid).mean(0)   # [audio_dim]
                    nb_v    = self._load_visual(nb_uid).mean(0)  # [visual_dim]
                    # Concatenate along feature axis and project to d_model via mean
                    # (simple proxy; the context module learns the actual mixing)
                    # Pad/truncate to d_model
                    raw = torch.cat([nb_t, nb_a, nb_v], dim=0)  # [text+audio+visual]
                    if raw.shape[0] >= d_model:
                        window_feats.append(raw[:d_model])
                    else:
                        pad = torch.zeros(d_model - raw.shape[0])
                        window_feats.append(torch.cat([raw, pad]))
                else:
                    window_feats.append(torch.zeros(d_model))

            sample["context_window"] = torch.stack(window_feats)  # [5, d_model]

        return sample


def collate_fn(batch):
    """
    Called automatically by DataLoader to combine a list of samples
    into one batch.

    Because text and audio have different lengths per utterance,
    we pad them to the longest in the batch.

    Returns tensors of shape:
      text:   [B, T_t_max, text_dim]   (1024 for RoBERTa-large)
      audio:  [B, T_a_max, audio_dim]  (1024 for WavLM-Base+)
      visual: [B, 30, 256]             (already fixed length)
      label:  [B]
      context_window: [B, 5, d_model]  (Tier 3, if present)
    """
    texts   = [item["text"]   for item in batch]
    audios  = [item["audio"]  for item in batch]
    visuals = [item["visual"] for item in batch]
    labels  = torch.stack([item["label"]  for item in batch])
    utt_ids = [item["utt_id"] for item in batch]

    # Pad variable-length sequences
    texts_padded   = pad_sequence(texts,   batch_first=True)   # [B, T_t, text_dim]
    audios_padded  = pad_sequence(audios,  batch_first=True)   # [B, T_a, audio_dim]
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

    result = {
        "utt_ids":     utt_ids,
        "text":        texts_padded,
        "audio":       audios_padded,
        "visual":      visuals_padded,
        "text_mask":   text_mask,
        "audio_mask":  audio_mask,
        "visual_mask": visual_mask,
        "label":       labels
    }

    # Tier 3: stack context windows if present in the batch
    if "context_window" in batch[0]:
        result["context_window"] = torch.stack(
            [item["context_window"] for item in batch])  # [B, 5, d_model]

    return result