"""Whisper model wrapper using HuggingFace Transformers."""

from __future__ import annotations

import time
import logging

import torch
import librosa

from models.base import ASRModel, TranscriptionResult, Segment

logger = logging.getLogger(__name__)


class WhisperHFModel(ASRModel):
    """Wrapper for OpenAI Whisper models via HuggingFace (large-v3, large-v3-turbo, etc.)."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        super().__init__(model_id, device)
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.processor = None
        self.model = None

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        logger.info("Loading Whisper model: %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")
        self._loaded = True
        logger.info("Whisper model loaded on %s", self.device)

    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        from transformers import pipeline

        t0 = time.perf_counter()

        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(audio_array) / sr

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device if self.device != "cpu" else -1,
        )

        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        if context:
            generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(
                context, return_tensors="pt"
            )

        result = pipe(
            audio_array,
            return_timestamps=True,
            generate_kwargs=generate_kwargs if generate_kwargs else None,
            chunk_length_s=30,
            batch_size=8,
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

    def supports_diarization(self) -> bool:
        return False

    def supports_context_injection(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "family": "Whisper",
                "torch_dtype": str(self.torch_dtype),
            }
        )
        return info

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
