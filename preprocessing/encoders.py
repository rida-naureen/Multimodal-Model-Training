"""
Centralized Encoders for ET-TACFN

Contains singleton-pattern encoders for Audio (WavLM), Text (Whisper + RoBERTa),
and Visual (ResNet50) modalities.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModel
import whisper
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# ── Audio Encoder (WavLM) ─────────────────────────────────────────────────────

class AudioEncoder:
    """
    Singleton audio encoder using WavLM.
    Loads WavLM model once and reuses it for all encoding requests.
    """
    
    def __init__(self):
        """Initialize and load WavLM model."""
        logger.info(f"Loading WavLM model on {DEVICE}...")
        
        self.device = DEVICE
        self.target_sr = TARGET_SR
        
        # Load feature extractor (using wavlm-base as requested)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base"
        )
        
        # Load WavLM model
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base"
        ).to(self.device)
        
        self.model.eval()
        
        logger.info("✅ WavLM model loaded successfully")
    
    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode audio to frame-level features using WavLM.

        Args:
            audio: np.ndarray (N,) mono audio @ 16kHz

        Returns:
            np.ndarray: WavLM frame embeddings [T, 768]
            (variable T depending on audio length)
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Feature extraction
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs["input_values"].to(self.device)

        # Forward pass
        outputs = self.model(input_values)

        # Return full frame-level sequence [T, 768] — NOT mean-pooled
        # This preserves temporal structure for the cross-modal attention
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()

_audio_encoder_instance = None

def get_audio_encoder() -> AudioEncoder:
    global _audio_encoder_instance
    if _audio_encoder_instance is None:
        _audio_encoder_instance = AudioEncoder()
    return _audio_encoder_instance

# ── Text Encoder (Whisper + RoBERTa) ──────────────────────────────────────────

class SpeechTextPipeline:
    """
    Singleton pipeline for speech-to-text-to-features.
    Combines Whisper (transcription) + RoBERTa (text encoding).
    """
    
    def __init__(self):
        """Initialize and load models."""
        logger.info(f"Loading Whisper + RoBERTa models on {DEVICE}...")
        
        self.device = DEVICE
        
        # Load Whisper (using small as requested)
        self.whisper_model = whisper.load_model("small", device=self.device)
        logger.info("✅ Whisper model loaded")
        
        # Load RoBERTa (using roberta-base as requested)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.text_model = AutoModel.from_pretrained("roberta-base").to(self.device)
        self.text_model.eval()
        logger.info("✅ RoBERTa model loaded")
    
    def process(self, wav_path: str) -> dict:
        """
        Process audio: transcribe with Whisper, then encode text with RoBERTa.
        """
        try:
            # Step 1: Language detection
            audio = whisper.load_audio(wav_path)
            audio = whisper.pad_or_trim(audio)
            
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            _, probs = self.whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            if detected_lang != "en":
                logger.warning(f"Non-English audio detected: {detected_lang}")
            
            # Step 2: Transcription
            result = self.whisper_model.transcribe(
                wav_path,
                language="en",
                task="transcribe",
                fp16=(self.device == "cuda"),
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                verbose=False
            )
            
            # Collect full transcript
            segments = result.get("segments", [])
            text = " ".join(seg["text"].strip() for seg in segments).strip()
            
            if not text:
                logger.warning("No speech detected in audio")
                return {
                    "transcript": "",
                    "text_features": np.zeros(768, dtype=np.float32).tolist()
                }
            
            # Step 3: Text encoding
            text_features = self.encode_text(text)
            
            return {
                "transcript": text,
                "text_features": text_features.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "transcript": "",
                "text_features": np.zeros(768, dtype=np.float32).tolist()
            }
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using RoBERTa. Returns token-level embeddings [T, 768]."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        outputs = self.text_model(**inputs)

        # Return full token-level sequence [T, 768] — NOT mean-pooled
        # This lets the cross-modal attention attend to individual tokens
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()

_text_encoder_instance = None

def get_text_encoder() -> SpeechTextPipeline:
    global _text_encoder_instance
    if _text_encoder_instance is None:
        _text_encoder_instance = SpeechTextPipeline()
    return _text_encoder_instance

# ── Visual Encoder (ResNet50) ───────────────────────────────────────────────

class VisualEncoder:
    """
    Singleton visual encoder using ResNet50.
    """

    def __init__(self):
        logger.info(f"Loading ResNet50 model on {DEVICE}...")
        self.device = DEVICE

        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()   # Remove classification head → (2048,)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("✅ ResNet50 model loaded successfully")

    @torch.no_grad()
    def encode(self, frames: list) -> np.ndarray:
        """
        Encode a list of video frames to a single (2048,) feature vector.
        """
        if not frames:
            logger.warning("No frames provided — returning zero vector")
            return np.zeros(2048, dtype=np.float32)

        tensors = []
        for frame in frames:
            if frame is None:
                continue
            if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            tensors.append(self.transform(frame))

        if not tensors:
            return np.zeros(2048, dtype=np.float32)

        # Single batched GPU forward pass
        batch = torch.stack(tensors).to(self.device)   # (N, 3, 224, 224)
        embeddings = self.model(batch)                  # (N, 2048)
        features = embeddings.mean(dim=0)               # (2048,)
        return features.cpu().numpy()

_visual_encoder_instance = None

def get_visual_encoder() -> VisualEncoder:
    global _visual_encoder_instance
    if _visual_encoder_instance is None:
        _visual_encoder_instance = VisualEncoder()
    return _visual_encoder_instance
