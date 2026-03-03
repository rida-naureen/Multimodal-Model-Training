# models/cross_modal_attention.py
# ============================================================
#  Cross-Modal Attention (CMA) — the CORE of this project
#
#  WHAT IS CROSS-MODAL ATTENTION?
#  --------------------------------
#  Imagine you're reading a transcript of someone saying "I'm fine."
#  With text alone you'd think they're OK.
#  But if you also hear their voice (shaky, low) and see their face
#  (looking down, no eye contact), you understand they're actually sad.
#
#  Cross-Modal Attention lets modality A "ask questions" of modality B:
#    Text  asks Audio:  "Is this part of the audio relevant to what was said?"
#    Text  asks Visual: "Does the face match the sentiment of the words?"
#    Audio asks Text:   "Does the transcript clarify this vocal pattern?"
#
#  HOW IT WORKS (Q, K, V):
#  ------------------------
#    Query (Q)  = what modality A wants to know
#    Key   (K)  = index of what modality B contains
#    Value (V)  = actual content of modality B
#
#    Attention = softmax(Q · Kᵀ / √d) · V
#
#    Result: modality A features, enriched with relevant info from B
# ============================================================

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    One cross-modal attention block.

    Args:
        d_model   : feature dimension (both modalities must be this size)
        num_heads : how many parallel attention heads (more = richer)
        dropout   : regularization to prevent overfitting

    Input:
        query_mod : [B, T_q,  d_model]  modality to enrich (e.g. text)
        kv_mod    : [B, T_kv, d_model]  source modality    (e.g. audio)
        key_mask  : [B, T_kv] bool      True = padded, ignore this position

    Output:
        out          : [B, T_q, d_model]  query enriched by source
        attn_weights : [B, num_heads, T_q, T_kv]  (for visualization)
    """

    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()

        # PyTorch's built-in multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True    # expects [Batch, Time, Features]
        )

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Small feed-forward network to transform attended features
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # expand
            nn.GELU(),                         # activation
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),  # contract back
            nn.Dropout(dropout)
        )

    def forward(self, query_mod, kv_mod, key_mask=None):
        # ── Cross-attention: Q from query_mod, K/V from kv_mod ──
        attended, attn_weights = self.attn(
            query=query_mod,
            key=kv_mod,
            value=kv_mod,
            key_padding_mask=key_mask,   # ignore padded positions
            average_attn_weights=False   # keep per-head weights
        )

        # ── Residual connection + LayerNorm ──────────────────────
        # "Residual" = add original input back, so we don't lose it
        x = self.norm1(query_mod + self.dropout(attended))

        # ── Feed-forward + another residual ──────────────────────
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights


class ModalityProjector(nn.Module):
    """
    Projects a modality's features to the shared d_model dimension.

    Why needed:
      Text  features = 768-dim  (from RoBERTa)
      Audio features = 768-dim  (from Wav2Vec2)
      Visual features= 256-dim  (from ResNet, already projected)
      → All need to be the same size for attention to work.

    Args:
        input_dim : original feature size
        d_model   : target size (same for all modalities)
    """

    def __init__(self, input_dim, d_model=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, T, input_dim]  →  [B, T, d_model]
        return self.proj(x)
