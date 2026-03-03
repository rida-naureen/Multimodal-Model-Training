# models/adaptive_fusion.py
# ============================================================
#  Adaptive Multimodal Fusion
#
#  Takes text + audio + visual features → one fused emotion vector
#
#  PIPELINE (step by step):
#  ─────────────────────────
#  Step 1:  Project all modalities to the same d_model size
#           (text: 768→256, audio: 768→256, visual: 256→256)
#
#  Step 2:  Cross-modal attention — each modality learns from others
#           • Text  attends to Audio   → T_A
#           • Text  attends to Visual  → T_V
#           • Audio attends to Text    → A_T
#
#  Step 3:  Mean-pool each sequence → single vector per modality
#           (respects padding masks)
#
#  Step 4:  Adaptive weights — learned per-sample, tells the model
#           "for THIS utterance, which modality is most reliable?"
#           e.g. a silent video → weight visual less
#
#  Step 5:  Weighted sum → one fused vector [B, d_model]
#
#  Step 6:  Final projection layer
# ============================================================

import torch
import torch.nn as nn
from models.cross_modal_attention import CrossModalAttention, ModalityProjector


class AdaptiveFusion(nn.Module):

    def __init__(self, text_dim=768, audio_dim=768, visual_dim=256,
                 d_model=256, num_heads=4, dropout=0.1):
        super().__init__()

        # ── Step 1: Projectors ────────────────────────────────
        self.text_proj   = ModalityProjector(text_dim,   d_model, dropout)
        self.audio_proj  = ModalityProjector(audio_dim,  d_model, dropout)
        self.visual_proj = ModalityProjector(visual_dim, d_model, dropout)

        # ── Step 2: Cross-modal attention pairs ───────────────
        self.cma_text_audio  = CrossModalAttention(d_model, num_heads, dropout)
        self.cma_text_visual = CrossModalAttention(d_model, num_heads, dropout)
        self.cma_audio_text  = CrossModalAttention(d_model, num_heads, dropout)

        # ── Step 4: Adaptive weight network ───────────────────
        # Input:  concatenated 3 modality vectors [B, 3*d_model]
        # Output: 3 scalar weights (softmax-normalized)
        self.weight_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

        # ── Step 6: Final projection ───────────────────────────
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _mean_pool(self, x, mask=None):
        """
        Pool [B, T, d] → [B, d] by averaging over time.
        Ignores padding positions (mask=True means padded).
        """
        if mask is not None:
            valid  = (~mask).float().unsqueeze(-1)          # [B, T, 1]  1=valid, 0=pad
            x      = x * valid
            pooled = x.sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return pooled

    def forward(self, text, audio, visual,
                text_mask=None, audio_mask=None, visual_mask=None):
        """
        Args:
            text, audio, visual : input feature tensors
            *_mask              : padding masks (True = padded position)

        Returns:
            fused        [B, d_model]  — final emotion representation
            attn_weights dict          — for visualization/analysis
        """

        # ── Step 1: Project to d_model ────────────────────────
        T = self.text_proj(text)       # [B, T_t, d]
        A = self.audio_proj(audio)     # [B, T_a, d]
        V = self.visual_proj(visual)   # [B, T_v, d]

        # ── Step 2: Cross-modal attention ─────────────────────
        # Text queries Audio
        TA, w_ta = self.cma_text_audio(
            query_mod=T, kv_mod=A, key_mask=audio_mask)

        # Text queries Visual
        TV, w_tv = self.cma_text_visual(
            query_mod=T, kv_mod=V, key_mask=visual_mask)

        # Audio queries Text
        AT, w_at = self.cma_audio_text(
            query_mod=A, kv_mod=T, key_mask=text_mask)

        # ── Step 3: Pool sequences → [B, d] ───────────────────
        f_ta = self._mean_pool(TA, text_mask)      # text enriched by audio
        f_tv = self._mean_pool(TV, text_mask)      # text enriched by visual
        f_at = self._mean_pool(AT, audio_mask)     # audio enriched by text
        f_v  = self._mean_pool(V,  visual_mask)    # visual (no CMA yet)

        # Merge two text-enriched streams
        f_text   = (f_ta + f_tv) / 2.0   # [B, d]
        f_audio  = f_at                   # [B, d]
        f_visual = f_v                    # [B, d]

        # ── Step 4: Compute adaptive per-sample weights ───────
        concat  = torch.cat([f_text, f_audio, f_visual], dim=-1)  # [B, 3d]
        weights = torch.softmax(self.weight_net(concat), dim=-1)  # [B, 3]

        # ── Step 5: Weighted sum ──────────────────────────────
        fused = (weights[:, 0:1] * f_text   +
                 weights[:, 1:2] * f_audio  +
                 weights[:, 2:3] * f_visual)   # [B, d]

        # ── Step 6: Final projection ──────────────────────────
        fused = self.fusion_proj(fused)         # [B, d]

        attn_weights = {
            "text_audio":    w_ta,
            "text_visual":   w_tv,
            "audio_text":    w_at,
            "modal_weights": weights.detach().cpu()  # [B, 3] per-sample
        }

        return fused, attn_weights
