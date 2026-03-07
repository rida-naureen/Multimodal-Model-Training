# models/missing_modality.py
# ============================================================
#  ET-TACFN Contribution 4 — Missing Modality Robustness
#
#  WHAT:
#    When a modality is unavailable at inference time,
#    replace it with a LEARNED fallback embedding instead of crashing.
#
#  WHY:
#    Real-world scenarios where modalities can be missing:
#      • Phone call        → no video
#      • Text chat         → no audio, no video
#      • Silent video      → no audio
#      • Low-light room    → unusable visual frames
#
#    Existing models (including TACFN) crash on missing modalities.
#    ET-TACFN handles this gracefully with trainable fallback tokens.
#
#  HOW:
#    Each modality has a learnable "missing" embedding (nn.Parameter).
#    These are trained alongside the rest of the model.
#    When modality is None → expand the learned token to full sequence length.
#
#    During training: randomly DROP one modality with 10% probability
#    This forces the model to learn to work without each modality,
#    making the learned fallbacks meaningful.
#
#  NOT in original TACFN — novel ET-TACFN contribution.
# ============================================================

import torch
import torch.nn as nn
import random


class MissingModalityHandler(nn.Module):
    """
    Provides learned fallback embeddings for absent modalities.

    Args:
        d_model    : feature dimension (after projection)
        max_text   : max text sequence length
        max_audio  : max audio sequence length
        max_visual : fixed visual frame count

    Usage at inference:
        text, audio, visual = handler(text=None, audio=audio_tensor, visual=visual_tensor)
        # text is now replaced with learned "missing text" embedding
    """

    def __init__(self, d_model=512, max_text=128, max_audio=300, max_visual=30):
        super().__init__()
        self.max_text   = max_text
        self.max_audio  = max_audio
        self.max_visual = max_visual

        # Learnable missing tokens — one sequence per modality
        # Shape [1, max_len, d_model] — will be expanded to batch size
        self.missing_text   = nn.Parameter(torch.randn(1, max_text,   d_model) * 0.02)
        self.missing_audio  = nn.Parameter(torch.randn(1, max_audio,  d_model) * 0.02)
        self.missing_visual = nn.Parameter(torch.randn(1, max_visual, d_model) * 0.02)

    def forward(self, text=None, audio=None, visual=None,
                text_mask=None, audio_mask=None, visual_mask=None,
                batch_size=1):
        """
        Args:
            text, audio, visual : modality tensors OR None if missing
            *_mask              : padding masks OR None
            batch_size          : B (needed to expand fallbacks)

        Returns:
            text, audio, visual  : original or fallback tensors
            text_mask, audio_mask, visual_mask : updated masks
        """
        device = (text.device   if text   is not None else
                  audio.device  if audio  is not None else
                  visual.device if visual is not None else
                  self.missing_text.device)

        # ── Text fallback ─────────────────────────────────────
        if text is None:
            text = self.missing_text.expand(batch_size, -1, -1).to(device)
            # All positions marked as "valid" — fallback has no padding
            text_mask = torch.zeros(
                batch_size, self.max_text, dtype=torch.bool, device=device)

        # ── Audio fallback ────────────────────────────────────
        if audio is None:
            audio = self.missing_audio.expand(batch_size, -1, -1).to(device)
            audio_mask = torch.zeros(
                batch_size, self.max_audio, dtype=torch.bool, device=device)

        # ── Visual fallback ───────────────────────────────────
        if visual is None:
            visual = self.missing_visual.expand(batch_size, -1, -1).to(device)
            visual_mask = torch.zeros(
                batch_size, self.max_visual, dtype=torch.bool, device=device)

        return text, audio, visual, text_mask, audio_mask, visual_mask


def apply_modality_dropout(text, audio, visual,
                           text_mask, audio_mask, visual_mask,
                           handler, dropout_prob=0.10):
    """
    Training-time modality dropout.

    Randomly drops one modality per batch with probability dropout_prob.
    This forces the model to learn to work without each modality,
    making the missing modality fallbacks meaningful.

    Args:
        text, audio, visual       : modality tensors
        *_mask                    : padding masks
        handler                   : MissingModalityHandler instance
        dropout_prob              : probability of dropping one modality

    Returns:
        Possibly modified text, audio, visual, masks
    """
    if not handler.training or random.random() > dropout_prob:
        return text, audio, visual, text_mask, audio_mask, visual_mask

    # Randomly choose which modality to drop
    choice = random.randint(0, 2)
    B      = text.size(0)

    if choice == 0:
        # Drop text — replace with learned fallback
        text, _, _, text_mask, _, _ = handler(
            text=None, audio=audio, visual=visual,
            text_mask=text_mask, audio_mask=audio_mask, visual_mask=visual_mask,
            batch_size=B
        )
    elif choice == 1:
        # Drop audio
        _, audio, _, _, audio_mask, _ = handler(
            text=text, audio=None, visual=visual,
            text_mask=text_mask, audio_mask=audio_mask, visual_mask=visual_mask,
            batch_size=B
        )
    else:
        # Drop visual
        _, _, visual, _, _, visual_mask = handler(
            text=text, audio=audio, visual=None,
            text_mask=text_mask, audio_mask=audio_mask, visual_mask=visual_mask,
            batch_size=B
        )

    return text, audio, visual, text_mask, audio_mask, visual_mask
