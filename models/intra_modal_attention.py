# models/intra_modal_attention.py
# ============================================================
#  ET-TACFN Contribution 1 — Trimodal Intra-Modal Self-Attention
#
#  WHAT:
#    Each modality attends to ITSELF before any cross-modal fusion.
#    This is Stage 1 of TACFN — extended here to include Text.
#
#  WHY:
#    Text  → removes filler words ("um","uh"), focuses on emotional tokens
#    Audio → suppresses silent frames, highlights expressive peaks
#    Visual→ down-weights neutral/blurry frames, focuses on key expressions
#
#  TACFN original: only Audio + Visual had this
#  ET-TACFN:       Text + Audio + Visual all have this  ← your extension
#
#  RESULT:
#    Cleaner features entering cross-modal attention →
#    less noise propagation across modalities →
#    better fusion quality
# ============================================================

import torch
import torch.nn as nn


class IntraModalSelfAttention(nn.Module):
    """
    Self-attention block for a single modality.
    Query = Key = Value = same modality features.

    Args:
        d_model   : feature dimension
        num_heads : parallel attention heads
        dropout   : regularization

    Input:  x    [B, T, d_model]  — modality features
            mask [B, T] bool      — True = padded position

    Output: [B, T, d_model]  — refined features (same shape)
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = num_heads,
            dropout      = dropout,
            batch_first  = True
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network — same as TACFN's design
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention: Q=K=V=x
        # need_weights=False during training: skip materialising the full
        # [B, heads, T, T] attn matrix — significant memory saving for long sequences.
        attn_out, _ = self.self_attn(
            query            = x,
            key              = x,
            value            = x,
            key_padding_mask = mask,
            need_weights     = not self.training
        )

        # Residual + LayerNorm (TACFN design)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class TrimodalIntraAttention(nn.Module):
    """
    Applies IntraModalSelfAttention independently to all 3 modalities.
    This is the complete Contribution 1 module.

    Args:
        d_model, num_heads, dropout : same as above

    Input:
        text, audio, visual : modality tensors
        text_mask, audio_mask, visual_mask : padding masks

    Output:
        refined_text, refined_audio, refined_visual
        (same shapes, noise removed, salient features emphasized)
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        # One self-attention block per modality
        self.text_self_attn   = IntraModalSelfAttention(d_model, num_heads, dropout)
        self.audio_self_attn  = IntraModalSelfAttention(d_model, num_heads, dropout)
        self.visual_self_attn = IntraModalSelfAttention(d_model, num_heads, dropout)

    def forward(self, text, audio, visual,
                text_mask=None, audio_mask=None, visual_mask=None):

        refined_text   = self.text_self_attn(text,   text_mask)
        refined_audio  = self.audio_self_attn(audio,  audio_mask)
        refined_visual = self.visual_self_attn(visual, visual_mask)

        return refined_text, refined_audio, refined_visual
