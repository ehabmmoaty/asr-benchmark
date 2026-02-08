"""Qwen3-ASR model wrapper (Qwen/Qwen3-ASR-1.7B).

Uses the `qwen-asr` package which provides a dedicated inference API
with automatic language detection, context biasing, and optional
word-level timestamps via Qwen3-ForcedAligner.

Install:  pip install qwen-asr
"""

from __future__ import annotations

import time
import logging

from models.base import ASRModel, TranscriptionResult, Segment

logger = logging.getLogger(__name__)


class Qwen3ASRModel(ASRModel):
    """
    Wrapper for Qwen/Qwen3-ASR-1.7B.

    1.7B-parameter end-to-end ASR model supporting 30 languages + 22 Chinese
    dialects with automatic language detection. Uses the ``qwen-asr`` Python
    package (not raw transformers).

    Optional companion model ``Qwen/Qwen3-ForcedAligner-0.6B`` provides
    word-level timestamps for 11 languages including Arabic and English.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        use_forced_aligner: bool = False,
        forced_aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ):
        super().__init__(model_id, device)
        self.use_forced_aligner = use_forced_aligner
        self.forced_aligner_id = forced_aligner_id
        self.model = None

    def load(self) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel as _Qwen3ASR
        except ImportError:
            raise ImportError(
                "qwen-asr package is required for Qwen3-ASR. "
                "Install with: pip install qwen-asr"
            )

        logger.info("Loading Qwen3-ASR: %s", self.model_id)

        if self.device == "cuda" and torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16
        else:
            self.device = "cpu"
            device_map = "cpu"
            dtype = torch.float32

        kwargs = dict(
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

        if self.use_forced_aligner:
            kwargs["forced_aligner"] = self.forced_aligner_id
            kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=device_map,
            )

        self.model = _Qwen3ASR.from_pretrained(self.model_id, **kwargs)
        self._loaded = True
        logger.info("Qwen3-ASR loaded on %s", self.device)

    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        import librosa

        t0 = time.perf_counter()

        try:
            audio_duration = librosa.get_duration(path=audio_path)

            # Map our language codes to Qwen's expected format
            qwen_lang = None
            if language:
                lang_map = {
                    "ar": "Arabic",
                    "ar-ae": "Arabic",
                    "ar-sa": "Arabic",
                    "en": "English",
                    "en-us": "English",
                }
                qwen_lang = lang_map.get(language.lower(), language)

            kwargs = dict(
                audio=audio_path,
                language=qwen_lang,  # None = auto-detect
                return_time_stamps=self.use_forced_aligner,
            )

            if context:
                kwargs["context"] = context

            results = self.model.transcribe(**kwargs)
            r = results[0]

            elapsed = time.perf_counter() - t0

            # Build segments from timestamps if available
            segments = []
            if hasattr(r, "time_stamps") and r.time_stamps:
                for ts in r.time_stamps:
                    segments.append(
                        Segment(
                            start=ts.start_time,
                            end=ts.end_time,
                            text=ts.text,
                        )
                    )

            detected_lang = ""
            if hasattr(r, "language") and r.language:
                detected_lang = r.language

            return TranscriptionResult(
                text=r.text.strip(),
                segments=segments,
                language_detected=detected_lang,
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=audio_duration,
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("Qwen3-ASR transcription failed: %s", e)
            return TranscriptionResult(
                text="",
                segments=[],
                language_detected="",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=0.0,
                error=str(e),
            )

    def supports_diarization(self) -> bool:
        return False

    def supports_context_injection(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "family": "Qwen3-ASR",
                "parameters": "1.7B",
                "languages": "30 languages + 22 Chinese dialects",
                "timestamps": self.use_forced_aligner,
                "forced_aligner": self.forced_aligner_id if self.use_forced_aligner else None,
                "package": "qwen-asr",
            }
        )
        return info

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._loaded = False
