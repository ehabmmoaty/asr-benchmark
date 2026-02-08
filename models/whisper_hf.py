"""Whisper model wrapper using HuggingFace Transformers."""

from __future__ import annotations

import time
import logging

import torch
import librosa
import numpy as np

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

        # Use float32 on CPU for stability
        if self.device == "cuda" and torch.cuda.is_available():
            dtype = self.torch_dtype
        else:
            self.device = "cpu"
            dtype = torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self._loaded = True
        logger.info("Whisper model loaded on %s (dtype=%s)", self.device, dtype)

    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(audio_array) / sr

        # Use model.generate() directly — more reliable than pipeline for Whisper
        input_features = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self.device, dtype=self.model.dtype)

        generate_kwargs = {"return_timestamps": True}
        if language:
            generate_kwargs["language"] = language
        if context:
            generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(
                context, return_tensors="pt"
            )

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs,
            )

        # Decode with timestamps
        output = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True, decode_with_timestamps=True
        )

        # Also decode without timestamps for clean text
        clean_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True, decode_with_timestamps=False
        )

        elapsed = time.perf_counter() - t0

        # Parse timestamp tokens from output
        segments = self._parse_timestamp_output(
            output[0] if output else "", audio_duration
        )

        return TranscriptionResult(
            text=(clean_text[0] if clean_text else "").strip(),
            segments=segments,
            language_detected=language or "",
            processing_time_seconds=elapsed,
            model_name=self.model_id,
            audio_duration_seconds=audio_duration,
        )

    def _parse_timestamp_output(self, text: str, audio_duration: float) -> list[Segment]:
        """Parse Whisper timestamp tokens like <|0.00|>text<|2.50|> into segments."""
        import re

        segments = []
        # Pattern: <|start|>text<|end|>
        pattern = r"<\|(\d+\.?\d*)\|>(.*?)(?=<\|(\d+\.?\d*)\|>|$)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            # No timestamp tokens — return the whole text as one segment
            cleaned = re.sub(r"<\|.*?\|>", "", text).strip()
            if cleaned:
                segments.append(
                    Segment(start=0.0, end=audio_duration, text=cleaned)
                )
            return segments

        i = 0
        while i < len(matches):
            start_time = float(matches[i].group(1))
            segment_text = matches[i].group(2).strip()

            # The end timestamp is the start of the next segment or audio end
            if i + 1 < len(matches):
                end_time = float(matches[i + 1].group(1))
            else:
                end_time = audio_duration

            if segment_text:
                segments.append(
                    Segment(start=start_time, end=end_time, text=segment_text)
                )
            i += 1

        return segments

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
