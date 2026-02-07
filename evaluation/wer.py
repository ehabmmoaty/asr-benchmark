"""Word Error Rate calculation with Arabic normalization support."""

from __future__ import annotations

from dataclasses import dataclass

from jiwer import wer, cer

from utils.arabic import normalize_arabic, extract_language_portions


@dataclass
class WERResult:
    """WER calculation result with language breakdowns."""

    wer_overall: float
    cer_overall: float
    wer_arabic: float | None = None
    wer_english: float | None = None
    reference_word_count: int = 0
    hypothesis_word_count: int = 0
    reference_text: str = ""
    hypothesis_text: str = ""


def _safe_wer(reference: str, hypothesis: str) -> float:
    """Calculate WER safely, returning 1.0 if reference is empty."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return wer(ref, hyp)


def _safe_cer(reference: str, hypothesis: str) -> float:
    """Calculate CER safely."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return cer(ref, hyp)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Simple WER between two strings (no normalization)."""
    return _safe_wer(reference, hypothesis)


def calculate_wer_arabic(reference: str, hypothesis: str) -> WERResult:
    """
    Calculate WER with Arabic normalization and language breakdowns.

    Both reference and hypothesis are normalized before comparison.
    Arabic and English portions are scored separately when possible.
    """
    # Normalize both texts
    norm_ref = normalize_arabic(reference)
    norm_hyp = normalize_arabic(hypothesis)

    # Overall WER/CER on normalized text
    wer_overall = _safe_wer(norm_ref, norm_hyp)
    cer_overall = _safe_cer(norm_ref, norm_hyp)

    # Language-specific breakdowns
    ref_parts = extract_language_portions(reference)
    hyp_parts = extract_language_portions(hypothesis)

    wer_ar = None
    if ref_parts["ar"].strip():
        norm_ref_ar = normalize_arabic(ref_parts["ar"])
        norm_hyp_ar = normalize_arabic(hyp_parts["ar"])
        wer_ar = _safe_wer(norm_ref_ar, norm_hyp_ar)

    wer_en = None
    if ref_parts["en"].strip():
        ref_en = ref_parts["en"].lower().strip()
        hyp_en = hyp_parts["en"].lower().strip()
        wer_en = _safe_wer(ref_en, hyp_en)

    return WERResult(
        wer_overall=wer_overall,
        cer_overall=cer_overall,
        wer_arabic=wer_ar,
        wer_english=wer_en,
        reference_word_count=len(norm_ref.split()),
        hypothesis_word_count=len(norm_hyp.split()),
        reference_text=norm_ref,
        hypothesis_text=norm_hyp,
    )
