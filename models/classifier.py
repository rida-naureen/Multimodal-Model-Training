# models/classifier.py
# ============================================================
#  ET-TACFN Model: ETTACFNFusion + EmotionClassifier
#
#  Pipeline:
#    Text + Audio + Visual
#        ↓
#    ETTACFNFusion
#        ↓
#    EmotionClassifier  (3-layer MLP head)
#        ↓
#    4 emotion logits: Happy / Sad / Angry / Neutral
# ============================================================

import torch.nn as nn
from models.et_tacfn_fusion import ETTACFNFusion


class EmotionClassifier(nn.Module):
    """
    3-layer MLP classifier on top of fused features.

    Input:  [B, d_model]     fused emotion representation
    Output: [B, num_classes] raw logits
    """

    def __init__(self, d_model=512, num_classes=4, dropout=0.1):
        super().__init__()
        clf_drop = max(dropout, 0.35)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(clf_drop),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(clf_drop),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)   # [B, num_classes]


class MultimodalEmotionModel(nn.Module):
    """
    Complete ET-TACFN model.

    Usage:
        model = MultimodalEmotionModel(cfg)
        logits, info = model(text, audio, visual,
                             text_mask, audio_mask, visual_mask)
        loss = criterion(logits, labels)
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]

        self.fusion = ETTACFNFusion(
            text_dim   = m["text_input_dim"],
            audio_dim  = m["audio_input_dim"],
            visual_dim = m["visual_input_dim"],
            d_model    = m["d_model"],
            num_heads  = m["num_heads"],
            dropout    = m["dropout"],
            cfg        = cfg
        )

        self.classifier = EmotionClassifier(
            d_model     = m["d_model"],
            num_classes = m["num_classes"],
            dropout     = m["dropout"]
        )

    def forward(self, text=None, audio=None, visual=None,
                text_mask=None, audio_mask=None, visual_mask=None):
        """
        Args:
            text, audio, visual         : modality feature tensors
            text_mask, audio_mask, visual_mask : padding masks (True = padding)

        Returns:
            logits : [B, num_classes]  raw class scores
            info   : dict of attention weights + confidence scores
        """
        fused, info = self.fusion(
            text, audio, visual,
            text_mask, audio_mask, visual_mask
        )
        logits = self.classifier(fused)
        return logits, info
