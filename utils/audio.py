"""Audio loading, resampling, and format utilities."""

from __future__ import annotations

import os
import logging

import librosa
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TARGET_SAMPLE_RATE = 16000


def get_audio_info(audio_path: str) -> dict:
    """Get audio file metadata without loading full audio."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    info = sf.info(audio_path)
    return {
        "path": audio_path,
        "filename": os.path.basename(audio_path),
        "duration_seconds": info.duration,
        "duration_minutes": info.duration / 60.0,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "file_size_mb": os.path.getsize(audio_path) / (1024 * 1024),
    }


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    return librosa.get_duration(path=audio_path)


def load_audio(
    audio_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample, and convert to mono.

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
    return audio, sr


def convert_to_wav(audio_path: str, output_path: str | None = None) -> str:
    """
    Convert an audio file to 16kHz mono WAV.

    Args:
        audio_path: Path to source audio file.
        output_path: Optional output path. Defaults to same dir with .wav extension.

    Returns:
        Path to the converted WAV file.
    """
    if output_path is None:
        base, _ = os.path.splitext(audio_path)
        output_path = base + "_converted.wav"

    audio, sr = load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE, mono=True)
    sf.write(output_path, audio, sr, subtype="PCM_16")
    logger.info("Converted %s → %s", audio_path, output_path)
    return output_path


def validate_audio_file(audio_path: str, max_duration_minutes: float = 60.0) -> str | None:
    """
    Validate an audio file for processing.

    Returns None if valid, or an error message string.
    """
    if not os.path.exists(audio_path):
        return f"File not found: {audio_path}"

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}"

    try:
        duration = get_audio_duration(audio_path)
    except Exception as e:
        return f"Cannot read audio file: {e}"

    if duration / 60.0 > max_duration_minutes:
        return (
            f"Audio too long: {duration / 60:.1f} min "
            f"(max {max_duration_minutes:.0f} min)"
        )

    return None


def list_audio_files(directory: str) -> list[str]:
    """List all supported audio files in a directory."""
    files = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_FORMATS:
            files.append(os.path.join(directory, fname))
    return files
