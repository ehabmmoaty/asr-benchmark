"""Diarization Error Rate calculation."""

from __future__ import annotations

from dataclasses import dataclass

from models.base import Segment


@dataclass
class DERResult:
    """Diarization Error Rate result."""

    der: float
    missed_speech: float
    false_alarm: float
    confusion: float
    total_reference_duration: float


def calculate_der(
    reference_segments: list[Segment],
    hypothesis_segments: list[Segment],
) -> DERResult | None:
    """
    Calculate Diarization Error Rate using pyannote.metrics.

    Args:
        reference_segments: Ground truth segments with speaker labels.
        hypothesis_segments: Model output segments with speaker labels.

    Returns:
        DERResult or None if segments lack speaker labels.
    """
    # Check that we have speaker labels
    ref_has_speakers = any(s.speaker for s in reference_segments)
    hyp_has_speakers = any(s.speaker for s in hypothesis_segments)
    if not ref_has_speakers or not hyp_has_speakers:
        return None

    try:
        from pyannote.core import Annotation, Segment as PySegment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        return None

    # Build pyannote Annotations
    reference = Annotation()
    for seg in reference_segments:
        if seg.speaker:
            reference[PySegment(seg.start, seg.end)] = seg.speaker

    hypothesis = Annotation()
    for seg in hypothesis_segments:
        if seg.speaker:
            hypothesis[PySegment(seg.start, seg.end)] = seg.speaker

    metric = DiarizationErrorRate()
    der_value = metric(reference, hypothesis)
    details = metric(reference, hypothesis, detailed=True)

    total_ref = sum(s.end - s.start for s in reference_segments if s.speaker)

    return DERResult(
        der=der_value,
        missed_speech=details.get("missed detection", 0.0),
        false_alarm=details.get("false alarm", 0.0),
        confusion=details.get("confusion", 0.0),
        total_reference_duration=total_ref,
    )
