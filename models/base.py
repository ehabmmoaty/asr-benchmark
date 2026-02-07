"""Abstract base class for ASR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Segment:
    """A single transcription segment with timing and optional speaker info."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float | None = None
    language: str | None = None


@dataclass
class TranscriptionResult:
    """Result of running an ASR model on an audio file."""

    text: str
    segments: list[Segment] = field(default_factory=list)
    language_detected: str = ""
    processing_time_seconds: float = 0.0
    model_name: str = ""
    audio_duration_seconds: float = 0.0
    error: str | None = None

    @property
    def rtf(self) -> float | None:
        """Real-Time Factor: processing time / audio duration. Lower is faster."""
        if self.audio_duration_seconds > 0:
            return self.processing_time_seconds / self.audio_duration_seconds
        return None

    @property
    def processing_time_per_minute(self) -> float | None:
        """Processing seconds per minute of audio."""
        if self.audio_duration_seconds > 0:
            minutes = self.audio_duration_seconds / 60.0
            return self.processing_time_seconds / minutes
        return None


class ASRModel(ABC):
    """Abstract base class that all ASR model wrappers must implement."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Call before transcribe()."""
        ...

    @abstractmethod
    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Optional language hint (e.g. "ar", "en").
            context: Optional context/prompt for models that support it.

        Returns:
            TranscriptionResult with transcript text, segments, and metadata.
        """
        ...

    def supports_diarization(self) -> bool:
        """Whether this model produces speaker labels."""
        return False

    def supports_context_injection(self) -> bool:
        """Whether this model accepts a context/prompt to improve accuracy."""
        return False

    def get_model_info(self) -> dict:
        """Return metadata about the model."""
        return {
            "model_id": self.model_id,
            "name": self.__class__.__name__,
            "supports_diarization": self.supports_diarization(),
            "supports_context_injection": self.supports_context_injection(),
            "loaded": self._loaded,
            "device": self.device,
        }

    def unload(self) -> None:
        """Release model from memory. Override if cleanup is needed."""
        self._loaded = False
