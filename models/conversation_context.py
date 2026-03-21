# models/conversation_context.py
# ============================================================
#  ET-TACFN Contribution C3 — Conversation Context Module
#  (Tier 3 novel contribution)
#
#  WHAT:
#    Runs a bidirectional GRU over a sliding window of 5 consecutive
#    utterance embeddings (i-2, i-1, i, i+1, i+2), enriching the
#    centre utterance with its conversational context.
#
#  WHY:
#    Emotion in IEMOCAP is conversational — "I hate you" is sarcastic
#    after "You did an amazing job" but sincere after a fight.
#    Neither TACFN, MulT, nor PMR model utterance-level context.
#    This is a publishable novel contribution over all three baselines.
#
#  ARCHITECTURE:
#    [utt_{i-2}, utt_{i-1}, utt_i, utt_{i+1}, utt_{i+2}]
#                         ↓  BiGRU (d_model → d_model, bidirectional)
#                     context_vec  [B, d_model]
#                         ↓  residual + LayerNorm
#                  context-enriched output  [B, d_model]
#
#    The residual ensures that if context adds noise, the original
#    utterance representation is not corrupted.
#
#  INPUT:
#    window: [B, window_size, d_model]
#      — sequence of fused utterance embeddings from ETTACFNFusion
#      — window_size = 5 by default (configurable, must be odd)
#
#  OUTPUT:
#    enriched: [B, d_model]
#      — context-enriched centre utterance embedding
# ============================================================

import torch
import torch.nn as nn


class ConversationContextModule(nn.Module):
    """
    Bidirectional GRU context encoder for utterance-level emotion context.

    Args:
        d_model     : feature dimension (must match ETTACFNFusion d_model)
        window_size : number of utterances in the context window (must be odd)
        dropout     : dropout rate applied after GRU output
    """

    def __init__(self, d_model: int, window_size: int = 5, dropout: float = 0.1):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd so centre utterance is well-defined"
        assert d_model % 2 == 0,     "d_model must be even for bidirectional GRU concat output"

        self.window_size = window_size
        self.centre_idx  = window_size // 2   # index of the target utterance in the window

        # Bidirectional GRU:
        #   hidden_size = d_model // 2
        #   concat of forward + backward → d_model (preserves dimensionality)
        self.gru = nn.GRU(
            input_size  = d_model,
            hidden_size = d_model // 2,
            num_layers  = 1,
            batch_first = True,
            bidirectional = True,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            window   : [B, window_size, d_model] — context window of fused embeddings
        Returns:
            enriched : [B, d_model] — context-enriched centre utterance embedding
        """
        # Extract centre utterance embedding for residual connection
        centre = window[:, self.centre_idx, :]      # [B, d_model]

        # Run BiGRU over the full context window
        gru_out, _ = self.gru(window)               # [B, window_size, d_model]

        # Take the output at the centre position
        # (forward GRU sees i-2, i-1, i; backward sees i+2, i+1, i — both meet here)
        context_vec = gru_out[:, self.centre_idx, :]   # [B, d_model]
        context_vec = self.dropout(context_vec)

        # Residual: original utterance info + context signal, normalised
        enriched = self.norm(context_vec + centre)     # [B, d_model]
        return enriched
