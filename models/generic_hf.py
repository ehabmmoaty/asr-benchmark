"""Generic HuggingFace ASR model loader for extensibility."""

from __future__ import annotations

import time
import logging

import torch
import librosa

from models.base import ASRModel, TranscriptionResult, Segment

logger = logging.getLogger(__name__)


class GenericHFModel(ASRModel):
    """
    Generic wrapper for any HuggingFace ASR model.

    Uses the transformers pipeline API, which supports most
    speech recognition models on the Hub.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        trust_remote_code: bool = False,
    ):
        super().__init__(model_id, device)
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.trust_remote_code = trust_remote_code
        self.pipe = None

    def load(self) -> None:
        from transformers import pipeline

        logger.info("Loading generic HF model: %s", self.model_id)

        device_arg = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=self.torch_dtype,
            device=device_arg,
            trust_remote_code=self.trust_remote_code,
        )

        if device_arg == -1:
            self.device = "cpu"

        self._loaded = True
        logger.info("Generic HF model loaded: %s on %s", self.model_id, self.device)

    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        try:
            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio_array) / sr

            kwargs = {}
            if language:
                kwargs["generate_kwargs"] = {"language": language}

            result = self.pipe(
                audio_array,
                return_timestamps=True,
                chunk_length_s=30,
                batch_size=8,
                **kwargs,
            )

            elapsed = time.perf_counter() - t0

            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    ts = chunk.get("timestamp", (None, None))
                    segments.append(
                        Segment(
                            start=ts[0] if ts[0] is not None else 0.0,
                            end=ts[1] if ts[1] is not None else 0.0,
                            text=chunk.get("text", "").strip(),
                        )
                    )

            return TranscriptionResult(
                text=result.get("text", "").strip(),
                segments=segments,
                language_detected=language or "",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=audio_duration,
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("Generic HF model transcription failed: %s", e)
            return TranscriptionResult(
                text="",
                segments=[],
                language_detected="",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=0.0,
                error=str(e),
            )

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "family": "Generic HuggingFace",
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": str(self.torch_dtype),
            }
        )
        return info

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
