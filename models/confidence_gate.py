# models/confidence_gate.py
# ============================================================
#  ET-TACFN Contribution 2 — Confidence-Gated Fusion
#
#  WHAT:
#    Each modality estimates its own reliability as a scalar (0–1).
#    Low reliability → features suppressed before fusion.
#    High reliability → features pass through fully.
#
#  WHY:
#    TACFN uses fixed residual connections — same weight for all samples.
#    But modality quality varies per utterance:
#      • Noisy recording     → audio confidence low  → suppress audio
#      • Poor lighting       → visual confidence low → suppress visual
#      • Ambiguous wording   → text confidence low   → suppress text
#
#    The gate is LEARNED from data — no manual rules needed.
#    The model trains itself to know when to trust each modality.
#
#  NOT in original TACFN — this is a novel ET-TACFN contribution.
#
#  RESULT:
#    More robust predictions on noisy/incomplete inputs.
#    Better accuracy on difficult samples where one modality misleads.
# ============================================================

import torch
import torch.nn as nn


class ConfidenceGate(nn.Module):
    """
    Produces a scalar confidence score for one modality.

    Architecture:
        mean-pool [B, T, d] → [B, d]
        linear(d → d//4) → ReLU → linear(d//4 → 1) → Sigmoid
        output: [B, 1]  scalar in range (0, 1)

    Args:
        d_model : feature dimension

    Input:  x_pooled [B, d_model]  — mean-pooled modality features
    Output: [B, 1]                 — confidence score per sample
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()    # clamp output to (0, 1)
        )

    def forward(self, x_pooled):
        # x_pooled: [B, d_model]
        return self.gate_net(x_pooled)   # [B, 1]


class TrimodalConfidenceGating(nn.Module):
    """
    Applies confidence gating to all 3 modalities.

    Pipeline:
        1. Mean-pool each modality sequence → [B, d]
        2. Compute confidence score for each modality → [B, 1]
        3. Multiply features by confidence → gated features [B, d]

    Input:
        text_feat, audio_feat, visual_feat : [B, d_model] pooled vectors

    Output:
        gated_text, gated_audio, gated_visual : [B, d_model]
        confidences : dict of confidence scores (for logging/analysis)
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.gate_text   = ConfidenceGate(d_model)
        self.gate_audio  = ConfidenceGate(d_model)
        self.gate_visual = ConfidenceGate(d_model)

    def forward(self, text_feat, audio_feat, visual_feat):
        # Compute confidence scores
        conf_t = self.gate_text(text_feat)     # [B, 1]
        conf_a = self.gate_audio(audio_feat)   # [B, 1]
        conf_v = self.gate_visual(visual_feat) # [B, 1]

        # Gate: multiply features by their confidence
        # Low confidence → scalar near 0 → features suppressed
        gated_text   = conf_t * text_feat      # [B, d]
        gated_audio  = conf_a * audio_feat     # [B, d]
        gated_visual = conf_v * visual_feat    # [B, d]

        confidences = {
            "text_conf":   conf_t.detach().cpu(),
            "audio_conf":  conf_a.detach().cpu(),
            "visual_conf": conf_v.detach().cpu()
        }

        return gated_text, gated_audio, gated_visual, confidences
