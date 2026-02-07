"""Arabic text normalization for fair WER comparison."""

from __future__ import annotations

import re
import unicodedata


# Arabic diacritics (tashkeel) Unicode range
_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Tatweel / kashida
_TATWEEL = "\u0640"

# Alef variants → bare alef
_ALEF_VARIANTS = {
    "\u0622": "\u0627",  # آ → ا
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0671": "\u0627",  # ٱ → ا
}

# Taa marbuta → haa (for spoken Arabic comparison)
_TAA_MARBUTA = "\u0629"
_HAA = "\u0647"

# Hamza variants → plain hamza or remove
_HAMZA_ABOVE = "\u0654"
_HAMZA_BELOW = "\u0655"
_HAMZA_VARIANTS = {
    "\u0624": "\u0648",  # ؤ → و
    "\u0626": "\u064A",  # ئ → ي
}

# Alef maksura → yaa
_ALEF_MAKSURA = "\u0649"
_YAA = "\u064A"


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for fair ASR WER comparison.

    Steps:
    1. Unicode NFKC normalization
    2. Remove diacritics (tashkeel)
    3. Remove tatweel (kashida)
    4. Normalize alef variants (أ إ آ ٱ → ا)
    5. Normalize taa marbuta (ة → ه) for spoken comparison
    6. Normalize hamza on carriers (ؤ → و, ئ → ي)
    7. Normalize alef maksura (ى → ي)
    8. Collapse whitespace
    9. Strip and lowercase English portions
    """
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Remove diacritics
    text = _DIACRITICS.sub("", text)

    # Remove tatweel
    text = text.replace(_TATWEEL, "")

    # Normalize alef variants
    for variant, replacement in _ALEF_VARIANTS.items():
        text = text.replace(variant, replacement)

    # Normalize taa marbuta
    text = text.replace(_TAA_MARBUTA, _HAA)

    # Normalize hamza variants
    for variant, replacement in _HAMZA_VARIANTS.items():
        text = text.replace(variant, replacement)

    # Remove standalone hamza diacritics
    text = text.replace(_HAMZA_ABOVE, "")
    text = text.replace(_HAMZA_BELOW, "")

    # Normalize alef maksura
    text = text.replace(_ALEF_MAKSURA, _YAA)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase (for English portions in code-switched text)
    text = text.lower()

    return text


def is_arabic_char(char: str) -> bool:
    """Check if a character is Arabic."""
    cp = ord(char)
    return (
        (0x0600 <= cp <= 0x06FF)  # Arabic
        or (0x0750 <= cp <= 0x077F)  # Arabic Supplement
        or (0x08A0 <= cp <= 0x08FF)  # Arabic Extended-A
        or (0xFB50 <= cp <= 0xFDFF)  # Arabic Presentation Forms-A
        or (0xFE70 <= cp <= 0xFEFF)  # Arabic Presentation Forms-B
    )


def split_arabic_english(text: str) -> list[dict]:
    """
    Split text into Arabic and English segments.

    Returns a list of dicts: {"text": ..., "language": "ar" | "en" | "mixed"}
    """
    if not text.strip():
        return []

    segments = []
    current_text = []
    current_lang = None

    for char in text:
        if char.isspace():
            current_text.append(char)
            continue

        if is_arabic_char(char):
            lang = "ar"
        elif char.isalpha():
            lang = "en"
        else:
            # Punctuation/numbers: keep with current segment
            current_text.append(char)
            continue

        if current_lang is None:
            current_lang = lang
        elif lang != current_lang:
            seg_text = "".join(current_text).strip()
            if seg_text:
                segments.append({"text": seg_text, "language": current_lang})
            current_text = []
            current_lang = lang

        current_text.append(char)

    # Flush last segment
    seg_text = "".join(current_text).strip()
    if seg_text:
        segments.append({"text": seg_text, "language": current_lang or "en"})

    return segments


def extract_language_portions(text: str) -> dict[str, str]:
    """
    Extract Arabic-only and English-only portions from a text.

    Returns {"ar": "arabic text...", "en": "english text..."}
    """
    segments = split_arabic_english(text)
    ar_parts = [s["text"] for s in segments if s["language"] == "ar"]
    en_parts = [s["text"] for s in segments if s["language"] == "en"]
    return {
        "ar": " ".join(ar_parts),
        "en": " ".join(en_parts),
    }
