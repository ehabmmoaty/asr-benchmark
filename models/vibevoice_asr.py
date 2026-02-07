"""VibeVoice-ASR model wrapper (microsoft/VibeVoice-ASR)."""

from __future__ import annotations

import time
import logging

import torch
import librosa

from models.base import ASRModel, TranscriptionResult, Segment

logger = logging.getLogger(__name__)


class VibeVoiceASRModel(ASRModel):
    """
    Wrapper for Microsoft VibeVoice-ASR.

    9B parameter model with built-in diarization, timestamps,
    and context injection support. Processes up to 60 minutes
    in a single pass. Requires ~20GB VRAM.
    """

    def __init__(
        self,
        model_id: str = "microsoft/VibeVoice-ASR",
        device: str = "cuda",
    ):
        super().__init__(model_id, device)
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        logger.info("Loading VibeVoice-ASR: %s (requires ~20GB VRAM)", self.model_id)

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "VibeVoice-ASR requires CUDA GPU with ~20GB VRAM. No CUDA device found."
            )

        if self.device == "cuda":
            free_mem = torch.cuda.get_device_properties(0).total_mem
            if free_mem < 18 * 1024**3:
                logger.warning(
                    "GPU may not have enough VRAM for VibeVoice-ASR. "
                    "Need ~20GB, available: %.1f GB",
                    free_mem / 1024**3,
                )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)

        self._loaded = True
        logger.info("VibeVoice-ASR loaded on %s", self.device)

    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        try:
            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio_array) / sr

            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
            ).to(self.device)

            generate_kwargs = {}
            if context:
                generate_kwargs["prompt"] = context
            if language:
                generate_kwargs["language"] = language

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                    return_timestamps=True,
                    return_segments=True,
                )

            decoded = self.processor.batch_decode(output, skip_special_tokens=False)
            result_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]

            # Parse segments with diarization from model output
            segments = self._parse_segments(decoded[0] if decoded else "")

            elapsed = time.perf_counter() - t0

            return TranscriptionResult(
                text=result_text.strip(),
                segments=segments,
                language_detected=language or "",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=audio_duration,
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("VibeVoice-ASR transcription failed: %s", e)
            return TranscriptionResult(
                text="",
                segments=[],
                language_detected="",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=0.0,
                error=str(e),
            )

    def _parse_segments(self, raw_output: str) -> list[Segment]:
        """
        Parse VibeVoice-ASR raw output into segments with timestamps and speakers.

        The model uses special tokens for timestamps and speaker labels.
        Format varies by model version; this attempts common patterns.
        """
        import re

        segments = []

        # Pattern: <|speaker_N|> <|start_time|> text <|end_time|>
        # Also handles: <|N.NN|> text <|N.NN|>
        speaker_pattern = re.compile(
            r"<\|(?:speaker[_\s]*)?(\w+)\|>"
        )
        time_pattern = re.compile(
            r"<\|(\d+\.?\d*)\|>"
        )

        current_speaker = None
        tokens = re.split(r"(<\|[^|]+\|>)", raw_output)

        i = 0
        while i < len(tokens):
            token = tokens[i].strip()

            # Check for speaker token
            speaker_match = speaker_pattern.match(token)
            if speaker_match:
                current_speaker = f"Speaker {speaker_match.group(1)}"
                i += 1
                continue

            # Check for timestamp token (start)
            time_match = time_pattern.match(token)
            if time_match:
                start_time = float(time_match.group(1))
                # Collect text until next timestamp
                text_parts = []
                end_time = start_time
                i += 1
                while i < len(tokens):
                    inner = tokens[i].strip()
                    inner_time = time_pattern.match(inner)
                    if inner_time:
                        end_time = float(inner_time.group(1))
                        i += 1
                        break
                    inner_speaker = speaker_pattern.match(inner)
                    if inner_speaker:
                        break
                    if inner:
                        text_parts.append(inner)
                    i += 1

                text = " ".join(text_parts).strip()
                if text:
                    segments.append(
                        Segment(
                            start=start_time,
                            end=end_time,
                            text=text,
                            speaker=current_speaker,
                        )
                    )
                continue

            i += 1

        return segments

    def supports_diarization(self) -> bool:
        return True

    def supports_context_injection(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "family": "VibeVoice-ASR",
                "parameters": "9B",
                "max_audio_minutes": 60,
                "gpu_memory_gb": 20,
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
