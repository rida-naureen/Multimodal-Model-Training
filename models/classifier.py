# models/classifier.py
# ============================================================
#  Full ET-TACFN Model: ETTACFNFusion + EmotionClassifier
#
#  This is the ONLY class you import in train.py and evaluate.py.
#
#  Pipeline:
#    Text + Audio + Visual
#        ↓
#    ETTACFNFusion  (all 4 contributions)
#        ↓
#    EmotionClassifier  (MLP head)
#        ↓
#    4 emotion logits: Happy / Sad / Angry / Neutral
# ============================================================

import torch.nn as nn
from models.et_tacfn_fusion import ETTACFNFusion


class EmotionClassifier(nn.Module):
    """
    3-layer MLP classifier on top of fused features.
    Deeper than before — matches increased d_model=512.

    Input:  [B, d_model]     fused emotion representation
    Output: [B, num_classes] raw logits
    """

    def __init__(self, d_model=512, num_classes=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),         # 512 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),    # 512 → 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes) # 256 → 4
        )

    def forward(self, x):
        return self.net(x)   # [B, num_classes]


class MultimodalEmotionModel(nn.Module):
    """
    Complete ET-TACFN model.

    Usage in train.py:
        model  = MultimodalEmotionModel(cfg)
        logits, info = model(text, audio, visual,
                             text_mask, audio_mask, visual_mask)
        loss   = criterion(logits, labels)

    Usage at inference with missing modality:
        logits, info = model(text=None, audio=audio, visual=visual,
                             audio_mask=audio_mask, visual_mask=visual_mask)
        # text replaced automatically with learned fallback
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
