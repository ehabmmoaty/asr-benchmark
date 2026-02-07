"""XEUS (espnet/xeus) speech encoder wrapper.

XEUS is a pre-trained speech encoder (577M params, 4000+ languages) from
CMU's WAVLab. It is NOT a standalone ASR system — it outputs feature vectors,
not text. To produce transcriptions it must be paired with a fine-tuned
decoder head.

This wrapper supports two modes:

1. **Fine-tuned checkpoint** (recommended) — If you have a fine-tuned XEUS
   model from the xeus-finetune pipeline, point ``ckpt_path`` to it and
   this wrapper will produce text.

2. **Encoder-only / feature extraction** — If no fine-tuned checkpoint is
   available, this wrapper runs the raw encoder and returns a placeholder
   message explaining that fine-tuning is needed. The features are still
   extracted (useful for pipeline debugging).

Install:
    pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
    git lfs install && git clone https://huggingface.co/espnet/xeus
"""

from __future__ import annotations

import os
import time
import logging

from models.base import ASRModel, TranscriptionResult, Segment

logger = logging.getLogger(__name__)


class XEUSModel(ASRModel):
    """
    Wrapper for espnet/xeus speech encoder.

    Parameters
    ----------
    model_id : str
        HuggingFace repo or local path to the XEUS checkpoint directory
        (the folder containing ``checkpoint.pth``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    ckpt_path : str | None
        Path to a fine-tuned xeus-finetune checkpoint directory.  When
        provided the model can produce text transcriptions.  When ``None``
        only feature extraction is available.
    """

    def __init__(
        self,
        model_id: str = "espnet/xeus",
        device: str = "cuda",
        ckpt_path: str | None = None,
    ):
        super().__init__(model_id, device)
        self.ckpt_path = ckpt_path
        self._encoder = None
        self._finetuned_model = None
        self._has_finetuned = False

    # --------------------------------------------------------------------- #
    #  Load
    # --------------------------------------------------------------------- #
    def load(self) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for XEUS.")

        # Resolve checkpoint path
        checkpoint_file = self._resolve_checkpoint()

        if self.ckpt_path and os.path.isdir(self.ckpt_path):
            self._load_finetuned(torch)
        else:
            self._load_encoder_only(checkpoint_file, torch)

        self._loaded = True

    def _resolve_checkpoint(self) -> str:
        """Find the raw encoder checkpoint file."""
        candidates = [
            os.path.join(self.model_id, "checkpoint.pth"),
            os.path.join(self.model_id, "model", "xeus_checkpoint.pth"),
            self.model_id,  # user may pass direct path to .pth
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

        # If model_id is a HF repo, try to download it
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=self.model_id,
                filename="checkpoint.pth",
            )
            return local
        except Exception:
            pass

        raise FileNotFoundError(
            f"Cannot find XEUS checkpoint. Clone the model first:\n"
            f"  git lfs install && git clone https://huggingface.co/{self.model_id}\n"
            f"Then pass the local directory path as model_id."
        )

    def _load_encoder_only(self, checkpoint_file: str, torch) -> None:
        """Load the raw SSL encoder."""
        try:
            from espnet2.tasks.ssl import SSLTask
        except ImportError:
            raise ImportError(
                "ESPnet SSL branch is required for XEUS. Install with:\n"
                "  pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'"
            )

        logger.info("Loading XEUS encoder from %s", checkpoint_file)
        self._encoder, _ = SSLTask.build_model_from_file(
            config=None,
            model_file=checkpoint_file,
            device=self.device,
        )
        self._encoder.eval()
        self._has_finetuned = False
        logger.info("XEUS encoder loaded (feature-extraction only)")

    def _load_finetuned(self, torch) -> None:
        """Load a fine-tuned xeus-finetune checkpoint for end-to-end ASR."""
        logger.info("Loading fine-tuned XEUS from %s", self.ckpt_path)
        try:
            # xeus-finetune saves a full model directory with config
            from espnet2.tasks.ssl import SSLTask

            # Look for the model file inside the checkpoint dir
            model_file = None
            for fname in os.listdir(self.ckpt_path):
                if fname.endswith(".pth") or fname.endswith(".pt"):
                    model_file = os.path.join(self.ckpt_path, fname)
                    break

            config_file = None
            for fname in os.listdir(self.ckpt_path):
                if fname.endswith(".yaml") or fname.endswith(".yml"):
                    config_file = os.path.join(self.ckpt_path, fname)
                    break

            if model_file:
                self._finetuned_model, _ = SSLTask.build_model_from_file(
                    config=config_file,
                    model_file=model_file,
                    device=self.device,
                )
                self._finetuned_model.eval()
                self._has_finetuned = True
                logger.info("Fine-tuned XEUS loaded for ASR inference")
            else:
                logger.warning(
                    "No .pth/.pt file found in %s — falling back to encoder-only",
                    self.ckpt_path,
                )
                checkpoint_file = self._resolve_checkpoint()
                self._load_encoder_only(checkpoint_file, torch)

        except Exception as e:
            logger.warning("Failed to load fine-tuned model: %s — falling back", e)
            checkpoint_file = self._resolve_checkpoint()
            self._load_encoder_only(checkpoint_file, torch)

    # --------------------------------------------------------------------- #
    #  Transcribe
    # --------------------------------------------------------------------- #
    def transcribe(
        self, audio_path: str, language: str | None = None, context: str | None = None
    ) -> TranscriptionResult:
        if not self._loaded:
            self.load()

        import torch
        import librosa
        from torch.nn.utils.rnn import pad_sequence

        t0 = time.perf_counter()

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio) / 16000

            wav_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
            wav_lengths = torch.LongTensor([len(audio)]).to(self.device)
            wavs = pad_sequence([wav_tensor], batch_first=True)

            if self._has_finetuned and self._finetuned_model is not None:
                text = self._run_finetuned(wavs, wav_lengths)
            else:
                text = self._run_encoder_only(wavs, wav_lengths)

            elapsed = time.perf_counter() - t0

            return TranscriptionResult(
                text=text,
                segments=[],
                language_detected=language or "",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=audio_duration,
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("XEUS transcription failed: %s", e)
            return TranscriptionResult(
                text="",
                segments=[],
                language_detected="",
                processing_time_seconds=elapsed,
                model_name=self.model_id,
                audio_duration_seconds=0.0,
                error=str(e),
            )

    def _run_encoder_only(self, wavs, wav_lengths) -> str:
        """Extract features and return info message (no decoder available)."""
        import torch

        with torch.no_grad():
            feats = self._encoder.encode(
                wavs,
                wav_lengths,
                use_mask=False,
                use_final_output=False,
            )[0][-1]  # last encoder layer

        return (
            f"[XEUS encoder-only mode] Extracted features: "
            f"shape {list(feats.shape)}. "
            f"Fine-tune with a CTC/attention decoder for text output. "
            f"See: github.com/pashanitw/xeus-finetune"
        )

    def _run_finetuned(self, wavs, wav_lengths) -> str:
        """Run fine-tuned model for actual transcription."""
        import torch

        with torch.no_grad():
            # The fine-tuned model should have an inference/recognize method
            if hasattr(self._finetuned_model, "recognize"):
                result = self._finetuned_model.recognize(wavs, wav_lengths)
                if isinstance(result, (list, tuple)):
                    return result[0] if result else ""
                return str(result)
            elif hasattr(self._finetuned_model, "inference"):
                result = self._finetuned_model.inference(wavs, wav_lengths)
                if isinstance(result, dict):
                    return result.get("text", str(result))
                return str(result)
            else:
                # Fallback: try encode + decode pipeline
                feats = self._finetuned_model.encode(
                    wavs, wav_lengths,
                    use_mask=False,
                    use_final_output=True,
                )[0]
                return f"[XEUS] Feature shape: {list(feats.shape)} — model lacks recognize() method"

    # --------------------------------------------------------------------- #
    #  Metadata
    # --------------------------------------------------------------------- #
    def supports_diarization(self) -> bool:
        return False

    def supports_context_injection(self) -> bool:
        return False

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(
            {
                "family": "XEUS (ESPnet)",
                "parameters": "577M",
                "type": "speech_encoder" if not self._has_finetuned else "finetuned_asr",
                "languages": "4000+ (pre-training)",
                "architecture": "E-Branchformer",
                "requires_finetuning": not self._has_finetuned,
                "note": (
                    "Encoder-only — produces features, not text. "
                    "Fine-tune with a decoder for ASR."
                    if not self._has_finetuned
                    else "Fine-tuned checkpoint loaded for ASR."
                ),
            }
        )
        return info

    def unload(self) -> None:
        if self._encoder is not None:
            del self._encoder
            self._encoder = None
        if self._finetuned_model is not None:
            del self._finetuned_model
            self._finetuned_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._loaded = False
