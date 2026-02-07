"""Aggregate metrics computation and reporting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

from models.base import TranscriptionResult, Segment
from evaluation.wer import calculate_wer_arabic, WERResult
from evaluation.der import calculate_der, DERResult


@dataclass
class GroundTruth:
    """Ground truth transcript loaded from a JSON file."""

    language: str
    transcript: str
    segments: list[Segment] = field(default_factory=list)
    file_path: str = ""

    @classmethod
    def from_json(cls, json_path: str) -> "GroundTruth":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        for seg in data.get("segments", []):
            segments.append(
                Segment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    speaker=seg.get("speaker"),
                    language=seg.get("language"),
                )
            )

        return cls(
            language=data.get("language", ""),
            transcript=data.get("transcript", ""),
            segments=segments,
            file_path=json_path,
        )


@dataclass
class MetricsResult:
    """Complete metrics for one model on one audio file."""

    model_name: str
    audio_file: str
    language: str
    wer_result: WERResult | None = None
    der_result: DERResult | None = None
    processing_time_seconds: float = 0.0
    audio_duration_seconds: float = 0.0
    rtf: float | None = None
    processing_time_per_minute: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = {
            "model_name": self.model_name,
            "audio_file": self.audio_file,
            "language": self.language,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "audio_duration_seconds": round(self.audio_duration_seconds, 2),
            "rtf": round(self.rtf, 4) if self.rtf else None,
            "processing_time_per_minute": (
                round(self.processing_time_per_minute, 2)
                if self.processing_time_per_minute
                else None
            ),
            "error": self.error,
        }
        if self.wer_result:
            d["wer_overall"] = round(self.wer_result.wer_overall, 4)
            d["cer_overall"] = round(self.wer_result.cer_overall, 4)
            d["wer_arabic"] = (
                round(self.wer_result.wer_arabic, 4)
                if self.wer_result.wer_arabic is not None
                else None
            )
            d["wer_english"] = (
                round(self.wer_result.wer_english, 4)
                if self.wer_result.wer_english is not None
                else None
            )
        if self.der_result:
            d["der"] = round(self.der_result.der, 4)
            d["missed_speech"] = round(self.der_result.missed_speech, 4)
            d["false_alarm"] = round(self.der_result.false_alarm, 4)
            d["confusion"] = round(self.der_result.confusion, 4)
        return d


def compute_all_metrics(
    transcription: TranscriptionResult,
    ground_truth: GroundTruth,
    audio_file: str,
) -> MetricsResult:
    """
    Compute all metrics comparing a transcription result to ground truth.
    """
    if transcription.error:
        return MetricsResult(
            model_name=transcription.model_name,
            audio_file=audio_file,
            language=ground_truth.language,
            error=transcription.error,
        )

    # WER
    wer_result = calculate_wer_arabic(ground_truth.transcript, transcription.text)

    # DER
    der_result = calculate_der(ground_truth.segments, transcription.segments)

    return MetricsResult(
        model_name=transcription.model_name,
        audio_file=audio_file,
        language=ground_truth.language,
        wer_result=wer_result,
        der_result=der_result,
        processing_time_seconds=transcription.processing_time_seconds,
        audio_duration_seconds=transcription.audio_duration_seconds,
        rtf=transcription.rtf,
        processing_time_per_minute=transcription.processing_time_per_minute,
    )


def find_ground_truth(audio_path: str) -> str | None:
    """Find a ground truth JSON file matching an audio file."""
    base, _ = os.path.splitext(audio_path)
    json_path = base + ".json"
    if os.path.exists(json_path):
        return json_path
    return None
