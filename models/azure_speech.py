"""Azure Speech Services REST API wrapper."""

from __future__ import annotations

import io
import json
import os
import time
import logging
from dataclasses import dataclass

import requests
import librosa
import soundfile as sf

from models.base import ASRModel, TranscriptionResult, Segment

logger = logging.getLogger(__name__)

# Azure Speech language codes relevant to our use case
AZURE_LANGUAGE_MAP = {
    "ar": "ar-AE",
    "ar-ae": "ar-AE",
    "ar-sa": "ar-SA",
    "en": "en-US",
    "en-us": "en-US",
    "en-gb": "en-GB",
}


class AzureSpeechModel(ASRModel):
    """
    Azure Speech Services via REST API.

    Supports continuous recognition with diarization.
    API key provided via AZURE_SPEECH_KEY env var or constructor.
    """

    def __init__(
        self,
        model_id: str = "azure-speech",
        device: str = "cpu",
        api_key: str | None = None,
        region: str = "uaenorth",
    ):
        super().__init__(model_id, device="cpu")
        self.api_key = api_key or os.environ.get("AZURE_SPEECH_KEY", "")
        self.region = region
        self._endpoint = f"https://{region}.stt.speech.microsoft.com"

    def load(self) -> None:
        """Validate API key is available."""
        if not self.api_key:
            raise ValueError(
                "Azure Speech API key required. Set AZURE_SPEECH_KEY env var "
                "or pass api_key to constructor."
            )
        self._loaded = True
        logger.info("Azure Speech configured for region: %s", self.region)

    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()

        try:
            # Load and convert to 16kHz mono WAV for API
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio) / sr

            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, 16000, format="WAV", subtype="PCM_16")
            wav_bytes = wav_buffer.getvalue()

            # Resolve language code
            lang_code = "ar-AE"
            if language:
                lang_code = AZURE_LANGUAGE_MAP.get(language.lower(), language)

            # Use batch transcription for longer files, simple for short
            if audio_duration > 60:
                result = self._transcribe_batch(wav_bytes, lang_code, context)
            else:
                result = self._transcribe_simple(wav_bytes, lang_code, context)

            elapsed = time.perf_counter() - t0
            result.processing_time_seconds = elapsed
            result.audio_duration_seconds = audio_duration
            result.model_name = f"Azure Speech ({self.region})"
            return result

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("Azure Speech transcription failed: %s", e)
            return TranscriptionResult(
                text="",
                segments=[],
                language_detected="",
                processing_time_seconds=elapsed,
                model_name=f"Azure Speech ({self.region})",
                audio_duration_seconds=0.0,
                error=str(e),
            )

    def _transcribe_simple(
        self, wav_bytes: bytes, language: str, context: str | None
    ) -> TranscriptionResult:
        """Use the simple REST endpoint for short audio."""
        url = (
            f"{self._endpoint}/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={language}&format=detailed"
            f"&profanity=raw&diarizationEnabled=true"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
            "Accept": "application/json",
        }

        if context:
            # Azure supports phrase list for context
            headers["Pronunciation-Assessment"] = json.dumps(
                {"referenceText": context}
            )

        response = requests.post(url, headers=headers, data=wav_bytes, timeout=120)
        response.raise_for_status()
        data = response.json()

        text = data.get("DisplayText", "")
        segments = []

        if "NBest" in data and data["NBest"]:
            best = data["NBest"][0]
            if "Words" in best:
                for word_info in best["Words"]:
                    offset = word_info.get("Offset", 0) / 10_000_000  # ticks → sec
                    duration = word_info.get("Duration", 0) / 10_000_000
                    segments.append(
                        Segment(
                            start=offset,
                            end=offset + duration,
                            text=word_info.get("Word", ""),
                            confidence=word_info.get("Confidence"),
                        )
                    )

        lang_detected = data.get("PrimaryLanguage", {}).get("Language", language)

        return TranscriptionResult(
            text=text,
            segments=segments,
            language_detected=lang_detected,
        )

    def _transcribe_batch(
        self, wav_bytes: bytes, language: str, context: str | None
    ) -> TranscriptionResult:
        """
        Use batch transcription API for longer audio.

        This uploads the audio, creates a transcription job, and polls for results.
        """
        # Step 1: Create transcription job
        create_url = f"{self._endpoint}/speechtotext/v3.1/transcriptions"

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }

        body = {
            "contentUrls": [],  # We'll use inline content
            "locale": language,
            "displayName": "ASR Benchmark Transcription",
            "properties": {
                "diarizationEnabled": True,
                "wordLevelTimestampsEnabled": True,
                "profanityFilterMode": "None",
            },
        }

        if context:
            body["properties"]["customPhrases"] = [{"phrase": p.strip()} for p in context.split(",")]

        # For batch API, we need to upload audio to a URL first
        # Using the direct content approach instead
        upload_url = f"{self._endpoint}/speechtotext/v3.1/transcriptions:transcribe"

        definition = {
            "locales": [language],
            "profanityFilterMode": "None",
            "diarizationEnabled": True,
        }

        import io

        files = {
            "audio": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            "definition": (
                "definition.json",
                io.BytesIO(json.dumps(definition).encode()),
                "application/json",
            ),
        }

        response = requests.post(
            upload_url,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            files=files,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        # Parse batch response
        combined_text = ""
        segments = []

        for phrase in data.get("combinedPhrases", []):
            combined_text += phrase.get("text", "") + " "

        for phrase in data.get("phrases", []):
            best = phrase.get("best", {})
            segments.append(
                Segment(
                    start=phrase.get("offsetMilliseconds", 0) / 1000,
                    end=(
                        phrase.get("offsetMilliseconds", 0)
                        + phrase.get("durationMilliseconds", 0)
                    )
                    / 1000,
                    text=best.get("display", phrase.get("text", "")),
                    speaker=f"Speaker {phrase.get('speaker', 'Unknown')}",
                    confidence=phrase.get("confidence"),
                )
            )

        return TranscriptionResult(
            text=combined_text.strip(),
            segments=segments,
            language_detected=language,
        )

    def supports_diarization(self) -> bool:
        return True

    def supports_context_injection(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "family": "Azure Speech Services",
                "region": self.region,
                "type": "api",
                "requires_key": True,
                "languages": ["ar-AE", "ar-SA", "en-US", "en-GB"],
            }
        )
        return info

    def unload(self) -> None:
        self._loaded = False
