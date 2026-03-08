# models/hierarchical_fusion.py
# ============================================================
#  ET-TACFN Contribution 3 — Hierarchical Speech-Visual Fusion
#
#  WHAT:
#    Two-stage fusion instead of flat single-stage:
#
#    Stage A — Speech Fusion:
#      Text + Audio → fused together first
#      Rationale: both come from same spoken utterance,
#                 naturally aligned in time and meaning
#
#    Stage B — Speech-Visual Fusion:
#      Speech representation + Visual → final emotion vector
#      Rationale: visual (face) provides additional context
#                 on top of what speech already captured
#
#  WHY BETTER THAN FLAT FUSION:
#    Flat fusion: Text + Audio + Visual all at once
#      → cross-modal noise propagates in all directions simultaneously
#      → harder to train, noisier gradients
#
#    Hierarchical: Text+Audio first → then + Visual
#      → respects natural grouping of speech modalities
#      → cleaner intermediate "speech" representation
#      → visual enriches speech context, not the other way around
#      → mirrors how humans process emotion in conversation
#
#  NOT in original TACFN — novel ET-TACFN contribution.
# ============================================================

import torch
import torch.nn as nn
from models.cross_modal_attention import CrossModalAttention


class HierarchicalFusion(nn.Module):
    """
    Two-stage hierarchical cross-modal fusion.

    Stage A: Text ←→ Audio  →  Speech Representation
    Stage B: Speech ←→ Visual → Final Emotion Vector

    Args:
        d_model   : shared feature dimension
        num_heads : attention heads
        dropout   : dropout rate
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()

        # ── Stage A: Speech Fusion (Text ←→ Audio) ───────────
        self.cma_text_audio  = CrossModalAttention(d_model, num_heads, dropout)
        self.cma_audio_text  = CrossModalAttention(d_model, num_heads, dropout)

        # Combine T←A and A←T into single speech vector
        self.speech_combiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ── Stage B: Speech-Visual Fusion ────────────────────
        self.cma_speech_visual = CrossModalAttention(d_model, num_heads, dropout)
        self.cma_visual_speech = CrossModalAttention(d_model, num_heads, dropout)

        # Also use remaining pairs: Audio←Visual, Visual←Audio
        self.cma_audio_visual  = CrossModalAttention(d_model, num_heads, dropout)
        self.cma_visual_audio  = CrossModalAttention(d_model, num_heads, dropout)
        # And Text←Visual, Visual←Text
        self.cma_text_visual   = CrossModalAttention(d_model, num_heads, dropout)
        self.cma_visual_text   = CrossModalAttention(d_model, num_heads, dropout)

        # Final combiner
        self.final_combiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _mean_pool(self, x, mask=None):
        """Pool [B, T, d] → [B, d], ignoring padded positions."""
        if mask is not None:
            valid  = (~mask).float().unsqueeze(-1)    # [B, T, 1]
            x      = x * valid
            pooled = x.sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return pooled

    def forward(self, text, audio, visual,
                text_mask=None, audio_mask=None, visual_mask=None):
        """
        Args:
            text, audio, visual : [B, T, d_model] refined modality features
            *_mask              : [B, T] bool padding masks

        Returns:
            fused       : [B, d_model]  final emotion representation
            attn_weights: dict          attention weights for analysis
        """

        # ════════════════════════════════════════════════════
        # STAGE A — Speech Fusion (Text ←→ Audio)
        # ════════════════════════════════════════════════════
        # Text enriched by Audio context
        ta, w_ta = self.cma_text_audio(
            query_mod=text,  kv_mod=audio, key_mask=audio_mask)

        # Audio enriched by Text context
        at, w_at = self.cma_audio_text(
            query_mod=audio, kv_mod=text,  key_mask=text_mask)

        # Also enrich Text with Visual and Audio with Visual here
        tv, w_tv = self.cma_text_visual(
            query_mod=text,  kv_mod=visual, key_mask=visual_mask)
        av, w_av = self.cma_audio_visual(
            query_mod=audio, kv_mod=visual, key_mask=visual_mask)

        # Pool text and audio streams
        t_pool = self._mean_pool(ta + tv, text_mask)    # [B, d]
        a_pool = self._mean_pool(at + av, audio_mask)   # [B, d]

        # Combine into unified speech representation
        speech = self.speech_combiner(
            torch.cat([t_pool, a_pool], dim=-1)
        )   # [B, d]

        # ════════════════════════════════════════════════════
        # STAGE B — Speech-Visual Fusion
        # ════════════════════════════════════════════════════
        # Expand speech to sequence length for attention
        speech_seq = speech.unsqueeze(1).expand(
            -1, visual.size(1), -1).contiguous()   # [B, T_v, d]

        # Speech enriched by Visual
        sv, w_sv = self.cma_speech_visual(
            query_mod=speech_seq, kv_mod=visual, key_mask=visual_mask)

        # Visual enriched by Speech
        vs, w_vs = self.cma_visual_speech(
            query_mod=visual, kv_mod=speech_seq, key_mask=None)

        # Visual also enriched by original Text and Audio
        vt, w_vt = self.cma_visual_text(
            query_mod=visual, kv_mod=text,  key_mask=text_mask)
        va, w_va = self.cma_visual_audio(
            query_mod=visual, kv_mod=audio, key_mask=audio_mask)

        # Pool stage B outputs
        sv_pool = self._mean_pool(sv, visual_mask)           # [B, d]
        vs_pool = self._mean_pool(vs + vt + va, visual_mask) # [B, d]

        # Final combination
        fused = self.final_combiner(
            torch.cat([sv_pool, vs_pool], dim=-1)
        )   # [B, d]

        attn_weights = {
            "text_audio":    w_ta,
            "audio_text":    w_at,
            "text_visual":   w_tv,
            "audio_visual":  w_av,
            "speech_visual": w_sv,
            "visual_speech": w_vs,
            "visual_text":   w_vt,
            "visual_audio":  w_va,
        }

        return fused, speech, attn_weights
