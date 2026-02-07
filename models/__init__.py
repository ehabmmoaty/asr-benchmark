
from __future__ import annotations
from models.base import ASRModel, TranscriptionResult, Segment
from models.whisper_hf import WhisperHFModel
from models.vibevoice_asr import VibeVoiceASRModel
from models.azure_speech import AzureSpeechModel
from models.generic_hf import GenericHFModel
from models.qwen3_asr import Qwen3ASRModel
from models.xeus_espnet import XEUSModel

__all__ = [
    "ASRModel",
    "TranscriptionResult",
    "Segment",
    "WhisperHFModel",
    "VibeVoiceASRModel",
    "AzureSpeechModel",
    "GenericHFModel",
    "Qwen3ASRModel",
    "XEUSModel",
]
