# models/et_tacfn_fusion.py
# ============================================================
#  ET-TACFN — Enhanced Trimodal Adaptive Cross-Modal Fusion
#
#  COMPLETE FUSION MODULE combining all 4 contributions:
#
#  Contribution 1: Trimodal Intra-Modal Self-Attention
#    → Extends TACFN Stage 1 to include Text modality
#    → Each modality cleans itself before cross-modal fusion
#
#  Contribution 2: Confidence-Gated Fusion
#    → Each modality estimates its own reliability (0–1)
#    → Noisy/unreliable modalities auto-suppressed
#    → Novel — not in original TACFN
#
#  Contribution 3: Hierarchical Speech-Visual Fusion
#    → Stage A: Text + Audio → Speech Representation
#    → Stage B: Speech + Visual → Final Emotion Vector
#    → Novel — not in original TACFN
#
#  Contribution 4: Missing Modality Robustness
#    → Learned fallback embeddings for absent modalities
#    → Training-time modality dropout for robustness
#    → Novel — not in original TACFN
#
#  FULL PIPELINE:
#    Input → Project → [Contribution 4: missing check]
#          → [Contribution 1: intra-modal self-attn]
#          → [Contribution 3: hierarchical cross-modal]
#          → [Contribution 2: confidence gating]
#          → Adaptive weighted sum
#          → Output [B, d_model]
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
        text_dim   : input dim of text features   (768 for RoBERTa)
        audio_dim  : input dim of audio features  (768 for Wav2Vec2)
        visual_dim : input dim of visual features (256 after ResNet projection)
        d_model    : shared hidden dimension
        num_heads  : attention heads
        dropout    : dropout rate
        cfg        : full config dict (for ET-TACFN flags)
    """

    def __init__(self, text_dim=768, audio_dim=768, visual_dim=256,
                 d_model=512, num_heads=8, dropout=0.1, cfg=None):
        super().__init__()

        # Read ET-TACFN feature flags from config
        self.use_intra      = cfg["model"].get("use_intra_modal_attn",  True) if cfg else True
        self.use_confidence = cfg["model"].get("use_confidence_gate",   True) if cfg else True
        self.use_hierarchical = cfg["model"].get("use_hierarchical",    True) if cfg else True
        self.use_missing    = cfg["model"].get("use_missing_modality",  True) if cfg else True
        self.modal_dropout  = cfg["training"].get("modality_dropout",   0.10) if cfg else 0.10

        # ── Modality Projectors (all → d_model) ───────────────
        self.text_proj   = ModalityProjector(text_dim,   d_model, dropout)
        self.audio_proj  = ModalityProjector(audio_dim,  d_model, dropout)
        self.visual_proj = ModalityProjector(visual_dim, d_model, dropout)

        # ── Contribution 4: Missing Modality Handler ──────────
        if self.use_missing:
            self.missing_handler = MissingModalityHandler(
                d_model    = d_model,
                text_dim   = text_dim,
                audio_dim  = audio_dim,
                visual_dim = visual_dim
            )

        # ── Contribution 1: Trimodal Intra-Modal Self-Attention
        if self.use_intra:
            self.intra_attn = TrimodalIntraAttention(d_model, num_heads, dropout)

        # ── Contribution 3: Hierarchical Fusion ───────────────
        if self.use_hierarchical:
            self.hierarchical = HierarchicalFusion(d_model, num_heads, dropout)
        else:
            # Fallback flat fusion if hierarchical disabled
            from models.cross_modal_attention import CrossModalAttention
            self.cma_ta = CrossModalAttention(d_model, num_heads, dropout)
            self.cma_tv = CrossModalAttention(d_model, num_heads, dropout)
            self.cma_at = CrossModalAttention(d_model, num_heads, dropout)

        # ── Contribution 2: Confidence-Gated Fusion ───────────
        if self.use_confidence:
            self.conf_gate = TrimodalConfidenceGating(d_model)

        # ── Adaptive Modality Weighting ───────────────────────
        # Learns per-sample importance of each modality
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

        # ── Final Projection ──────────────────────────────────
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
            *_mask : [B, T] bool padding masks

        Returns:
            fused       : [B, d_model]  final emotion representation
            info        : dict of attention weights + confidence scores
        """
        B = (text.size(0)   if text   is not None else
             audio.size(0)  if audio  is not None else
             visual.size(0))

        # ════════════════════════════════════════════════════════
        # CONTRIBUTION 4 — Missing Modality Handling
        # ════════════════════════════════════════════════════════
        if self.use_missing:
            # Training: randomly drop one modality (modality dropout)
            if self.training:
                text, audio, visual, text_mask, audio_mask, visual_mask = \
                    apply_modality_dropout(
                        text, audio, visual,
                        text_mask, audio_mask, visual_mask,
                        self.missing_handler,
                        dropout_prob=self.modal_dropout,
                        is_training=self.training
                    )
            # Inference: replace any None modality with learned fallback
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

        # ════════════════════════════════════════════════════════
        # CONTRIBUTION 1 — Trimodal Intra-Modal Self-Attention
        # ════════════════════════════════════════════════════════
        if self.use_intra:
            T, A, V = self.intra_attn(
                T, A, V, text_mask, audio_mask, visual_mask)

        # ════════════════════════════════════════════════════════
        # CONTRIBUTION 3 — Hierarchical Fusion
        # Stage A: Text+Audio → Speech
        # Stage B: Speech+Visual → Fused
        # ════════════════════════════════════════════════════════
        attn_weights = {}
        if self.use_hierarchical:
            fused_hier, speech_repr, t_pool, a_pool, attn_weights = self.hierarchical(
                T, A, V, text_mask, audio_mask, visual_mask)

            v_pool = self._mean_pool(V, visual_mask)  # visual stays mean-pooled

        else:
            # Fallback flat fusion
            ta, w_ta = self.cma_ta(T, A, audio_mask)
            tv, w_tv = self.cma_tv(T, V, visual_mask)
            at, w_at = self.cma_at(A, T, text_mask)
            t_pool = self._mean_pool((ta + tv) / 2, text_mask)
            a_pool = self._mean_pool(at, audio_mask)
            v_pool = self._mean_pool(V, visual_mask)
            fused_hier = (t_pool + a_pool + v_pool) / 3.0
            attn_weights = {"text_audio": w_ta, "text_visual": w_tv, "audio_text": w_at}

        # ════════════════════════════════════════════════════════
        # CONTRIBUTION 2 — Confidence-Gated Fusion
        # ════════════════════════════════════════════════════════
        confidences = {}
        if self.use_confidence:
            gated_t, gated_a, gated_v, confidences = self.conf_gate(
                t_pool, a_pool, v_pool)
        else:
            gated_t, gated_a, gated_v = t_pool, a_pool, v_pool

        # ── Adaptive Modality Weighting ───────────────────────
        concat  = torch.cat([gated_t, gated_a, gated_v], dim=-1)   # [B, 3d]
        weights = torch.softmax(
            self.adaptive_weight_net(concat), dim=-1)               # [B, 3]

        weighted = (weights[:, 0:1] * gated_t +
                    weights[:, 1:2] * gated_a +
                    weights[:, 2:3] * gated_v)                      # [B, d]

        # Combine hierarchical output + weighted gated output (residual)
        fused = fused_hier + weighted                                # [B, d]

        # Final projection
        fused = self.final_proj(fused)                              # [B, d]

        info = {
            **attn_weights,
            **confidences,
            "modal_weights": weights.detach().cpu()
        }

        return fused, info
