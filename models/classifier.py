# models/classifier.py
# ============================================================
#  Emotion Classifier + Full End-to-End Model
#
#  EmotionClassifier:
#    Takes fused vector [B, d_model] → emotion logits [B, 4]
#    A simple 2-layer MLP.
#
#  MultimodalEmotionModel:
#    Combines AdaptiveFusion + EmotionClassifier.
#    This is the ONLY class you need to import in train.py.
# ============================================================

import torch.nn as nn


class EmotionClassifier(nn.Module):
    """
    2-layer MLP to classify emotions.

    Input : [B, d_model]    fused multimodal features
    Output: [B, num_classes] raw scores (logits)
    """

    def __init__(self, d_model=256, num_classes=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),   # 256 → 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes) # 128 → 4
        )

    def forward(self, x):
        return self.net(x)   # [B, num_classes]


class MultimodalEmotionModel(nn.Module):
    """
    Full pipeline:
      Text + Audio + Visual
          ↓
      AdaptiveFusion  (cross-modal attention + weighted sum)
          ↓
      EmotionClassifier  (MLP)
          ↓
      4 emotion class logits

    Usage:
        model  = MultimodalEmotionModel(cfg)
        logits, attn = model(text, audio, visual, t_mask, a_mask, v_mask)
        loss   = criterion(logits, labels)
    """

    def __init__(self, cfg):
        super().__init__()
        from models.adaptive_fusion import AdaptiveFusion

        m = cfg["model"]

        self.fusion = AdaptiveFusion(
            text_dim   = m["text_input_dim"],
            audio_dim  = m["audio_input_dim"],
            visual_dim = m["visual_input_dim"],
            d_model    = m["d_model"],
            num_heads  = m["num_heads"],
            dropout    = m["dropout"]
        )

        self.classifier = EmotionClassifier(
            d_model    = m["d_model"],
            num_classes= m["num_classes"],
            dropout    = m["dropout"]
        )

    def forward(self, text, audio, visual,
                text_mask=None, audio_mask=None, visual_mask=None):
        """
        Returns:
            logits       [B, num_classes]  — raw class scores
            attn_weights dict              — for analysis
        """
        fused, attn_weights = self.fusion(
            text, audio, visual,
            text_mask, audio_mask, visual_mask
        )
        logits = self.classifier(fused)
        return logits, attn_weights
