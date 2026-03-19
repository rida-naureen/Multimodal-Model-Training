# models/cross_modal_attention.py
# ============================================================
#  Cross-Modal Attention (CMA) Module
#
#  Allows modality A to attend to modality B:
#    Query (Q) = modality A (what we want to enrich)
#    Key   (K) = modality B (index of what B contains)
#    Value (V) = modality B (actual content of B)
#
#    Attention = softmax(Q·Kᵀ / √d) · V
#
#  ET-TACFN uses ALL 6 directions (vs 3 in the old version):
#    Text  ← Audio   Text  ← Visual
#    Audio ← Text    Audio ← Visual   ← NEW
#    Visual← Text    Visual← Audio    ← NEW
# ============================================================

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Single cross-modal attention block.

    Args:
        d_model   : shared feature dimension
        num_heads : attention heads
        dropout   : dropout rate

    Input:
        query_mod : [B, T_q,  d_model]  modality to enrich
        kv_mod    : [B, T_kv, d_model]  source modality
        key_mask  : [B, T_kv] bool      True = padded, ignore

    Output:
        out          : [B, T_q, d_model]  enriched query modality
        attn_weights : attention weights (for visualization)
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query_mod, kv_mod, key_mask=None, need_weights=True):
        attended, attn_weights = self.attn(
            query            = query_mod,
            key              = kv_mod,
            value            = kv_mod,
            key_padding_mask = key_mask,
            need_weights     = need_weights,        # False during training → ~40% less attn memory
            average_attn_weights = False
        )
        x = self.norm1(query_mod + self.dropout(attended))
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class ModalityProjector(nn.Module):
    """
    Projects modality features to shared d_model dimension.

    Text  : 768 → d_model
    Audio : 768 → d_model
    Visual: 256 → d_model
    """

    def __init__(self, input_dim, d_model=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, T, input_dim] → [B, T, d_model]
        return self.proj(x)
