# models/et_tacfn_fusion.py
# ============================================================
#  ET-TACFN — Enhanced Trimodal Adaptive Cross-Modal Fusion
#
#  Full pipeline:
#    1. Project each modality to shared d_model
#    2. Missing modality handling (learned fallback embeddings)
#    3. Trimodal intra-modal self-attention
#    4. Hierarchical cross-modal fusion (Text+Audio → Speech → +Visual)
#    5. Confidence-gated adaptive weighting
#    6. Final projection
# ============================================================

import torch
import torch.nn as nn

from models.cross_modal_attention   import ModalityProjector
from models.intra_modal_attention   import TrimodalIntraAttention
from models.hierarchical_fusion     import HierarchicalFusion
from models.confidence_gate         import TrimodalConfidenceGating
from models.missing_modality        import MissingModalityHandler, apply_modality_dropout


class ETTACFNFusion(nn.Module):
    """
    Full ET-TACFN Fusion Module.

    Args:
        text_dim   : input dim of text features   (e.g. 768 for RoBERTa-base)
        audio_dim  : input dim of audio features  (e.g. 768 for wav2vec2-base)
        visual_dim : input dim of visual features (e.g. 256 from ResNet-50)
        d_model    : shared hidden dimension
        num_heads  : number of attention heads
        dropout    : dropout rate
        cfg        : full config dict
    """

    def __init__(self, text_dim=768, audio_dim=768, visual_dim=256,
                 d_model=512, num_heads=8, dropout=0.1, cfg=None):
        super().__init__()

        self.modal_dropout = (
            cfg["training"].get("modality_dropout", 0.15) if cfg else 0.15
        )

        # ── Project all modalities to d_model ─────────────────
        self.text_proj   = ModalityProjector(text_dim,   d_model, dropout)
        self.audio_proj  = ModalityProjector(audio_dim,  d_model, dropout)
        self.visual_proj = ModalityProjector(visual_dim, d_model, dropout)

        # ── Missing modality handler ───────────────────────────
        # Provides learned fallback embeddings when a modality is absent.
        # During training also applies random modality dropout for robustness.
        self.missing_handler = MissingModalityHandler(
            d_model    = d_model,
            text_dim   = text_dim,
            audio_dim  = audio_dim,
            visual_dim = visual_dim
        )

        # ── Trimodal intra-modal self-attention ────────────────
        # Each modality attends to itself to clean its own representation.
        self.intra_attn = TrimodalIntraAttention(d_model, num_heads, dropout)

        # ── Hierarchical cross-modal fusion ───────────────────
        # Stage A: Text + Audio → Speech representation
        # Stage B: Speech + Visual → Final fused vector
        self.hierarchical = HierarchicalFusion(d_model, num_heads, dropout)

        # ── Confidence-gated fusion ────────────────────────────
        # Each modality estimates its own reliability (0–1 scalar gate).
        # Noisy or unreliable modalities are automatically down-weighted.
        self.conf_gate = TrimodalConfidenceGating(d_model)

        # ── Adaptive modality weighting ───────────────────────
        # Learns per-sample soft importance weights over the three modalities.
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

        # ── Final projection ───────────────────────────────────
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _mean_pool(self, x, mask=None):
        if mask is not None:
            valid  = (~mask).float().unsqueeze(-1)
            x      = x * valid
            pooled = x.sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return pooled

    def forward(self, text, audio, visual,
                text_mask=None, audio_mask=None, visual_mask=None):
        """
        Args:
            text   : [B, T_t, text_dim]   or None
            audio  : [B, T_a, audio_dim]  or None
            visual : [B, T_v, visual_dim] or None
            *_mask : [B, T] bool padding masks (True = padding position)

        Returns:
            fused : [B, d_model]  final emotion representation
            info  : dict of attention weights + confidence scores
        """
        B = (text.size(0)  if text   is not None else
             audio.size(0) if audio  is not None else
             visual.size(0))

        # ── Missing modality handling ──────────────────────────
        if self.training:
            text, audio, visual, text_mask, audio_mask, visual_mask = \
                apply_modality_dropout(
                    text, audio, visual,
                    text_mask, audio_mask, visual_mask,
                    self.missing_handler,
                    dropout_prob=self.modal_dropout,
                    is_training=self.training
                )
        else:
            text, audio, visual, text_mask, audio_mask, visual_mask = \
                self.missing_handler(
                    text, audio, visual,
                    text_mask, audio_mask, visual_mask,
                    batch_size=B
                )

        # ── Project all modalities to d_model ─────────────────
        T = self.text_proj(text)      # [B, T_t, d]
        A = self.audio_proj(audio)    # [B, T_a, d]
        V = self.visual_proj(visual)  # [B, T_v, d]

        # ── Intra-modal self-attention ─────────────────────────
        T, A, V = self.intra_attn(T, A, V, text_mask, audio_mask, visual_mask)

        # ── Hierarchical cross-modal fusion ───────────────────
        fused_hier, speech_repr, t_pool, a_pool, attn_weights = \
            self.hierarchical(T, A, V, text_mask, audio_mask, visual_mask)

        v_pool = self._mean_pool(V, visual_mask)

        # ── Confidence-gated adaptive weighting ───────────────
        gated_t, gated_a, gated_v, confidences = \
            self.conf_gate(t_pool, a_pool, v_pool)

        concat  = torch.cat([gated_t, gated_a, gated_v], dim=-1)  # [B, 3d]
        weights = torch.softmax(
            self.adaptive_weight_net(concat), dim=-1)              # [B, 3]

        weighted = (weights[:, 0:1] * gated_t +
                    weights[:, 1:2] * gated_a +
                    weights[:, 2:3] * gated_v)                     # [B, d]

        # Combine hierarchical output + weighted gated output
        fused = self.final_proj(fused_hier + weighted)             # [B, d]

        info = {
            **attn_weights,
            **confidences,
            "modal_weights": weights.detach().cpu()
        }

        return fused, info
