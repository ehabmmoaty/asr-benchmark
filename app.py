"""
Anees ASR Benchmark — Streamlit App

Compare HuggingFace ASR models for Arabic/English transcription.
"""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.base import ASRModel, TranscriptionResult, Segment
from models.whisper_hf import WhisperHFModel
from models.vibevoice_asr import VibeVoiceASRModel
from models.azure_speech import AzureSpeechModel
from models.generic_hf import GenericHFModel
from models.qwen3_asr import Qwen3ASRModel
from models.xeus_espnet import XEUSModel
from evaluation.wer import calculate_wer_arabic
from evaluation.der import calculate_der
from evaluation.metrics import (
    compute_all_metrics,
    MetricsResult,
    GroundTruth,
    find_ground_truth,
)
from utils.audio import (
    get_audio_info,
    get_audio_duration,
    validate_audio_file,
    list_audio_files,
    SUPPORTED_FORMATS,
)
from utils.arabic import normalize_arabic, split_arabic_english
from utils.export import export_csv_bytes, export_html_bytes, metrics_to_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DEFAULT_MODELS = {
    "whisper_large_v3": {
        "hub_id": "openai/whisper-large-v3",
        "class": "WhisperHFModel",
        "enabled": True,
        "supports_context": True,
        "supports_diarization": False,
    },
    "whisper_large_v3_turbo": {
        "hub_id": "openai/whisper-large-v3-turbo",
        "class": "WhisperHFModel",
        "enabled": True,
        "supports_context": True,
        "supports_diarization": False,
    },
    "vibevoice_asr": {
        "hub_id": "microsoft/VibeVoice-ASR",
        "class": "VibeVoiceASRModel",
        "enabled": True,
        "supports_context": True,
        "supports_diarization": True,
    },
    "azure_speech": {
        "hub_id": "azure-speech",
        "class": "AzureSpeechModel",
        "enabled": True,
        "type": "api",
        "supports_context": True,
        "supports_diarization": True,
    },
    "qwen3_asr": {
        "hub_id": "Qwen/Qwen3-ASR-1.7B",
        "class": "Qwen3ASRModel",
        "enabled": True,
        "supports_context": True,
        "supports_diarization": False,
    },
    "xeus": {
        "hub_id": "espnet/xeus",
        "class": "XEUSModel",
        "enabled": True,
        "supports_context": False,
        "supports_diarization": False,
        "type": "speech_encoder",
    },
}


def load_config() -> dict:
    """Load config.yaml or fall back to defaults."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {"models": DEFAULT_MODELS}


def get_model_registry(config: dict) -> dict[str, dict]:
    """Build model registry from config."""
    return config.get("models", DEFAULT_MODELS)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    "WhisperHFModel": WhisperHFModel,
    "VibeVoiceASRModel": VibeVoiceASRModel,
    "AzureSpeechModel": AzureSpeechModel,
    "GenericHFModel": GenericHFModel,
    "Qwen3ASRModel": Qwen3ASRModel,
    "XEUSModel": XEUSModel,
}


def create_model(key: str, model_cfg: dict, **overrides) -> ASRModel:
    """Instantiate a model from its config entry."""
    cls_name = model_cfg.get("class", "GenericHFModel")
    cls = MODEL_CLASSES.get(cls_name, GenericHFModel)
    hub_id = model_cfg.get("hub_id", key)

    kwargs = {"model_id": hub_id}

    if cls_name == "AzureSpeechModel":
        kwargs["api_key"] = overrides.get("azure_key", os.environ.get("AZURE_SPEECH_KEY", ""))
        kwargs["region"] = model_cfg.get("region", overrides.get("azure_region", "uaenorth"))
    else:
        kwargs["device"] = overrides.get("device", "cuda")

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session():
    """Initialize Streamlit session state."""
    if "loaded_models" not in st.session_state:
        st.session_state.loaded_models = {}
    if "results" not in st.session_state:
        st.session_state.results = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar(config: dict) -> dict:
    """Render sidebar with model selection and settings."""
    st.sidebar.title("ASR Benchmark Settings")

    settings = {}

    # Mode selection
    settings["mode"] = st.sidebar.radio(
        "Mode",
        ["Single File Comparison", "Batch Benchmark", "Results Dashboard"],
        index=0,
    )

    st.sidebar.markdown("---")

    # Model selection
    st.sidebar.subheader("Models")
    registry = get_model_registry(config)
    selected = {}

    for key, cfg in registry.items():
        enabled = cfg.get("enabled", True)
        label = cfg.get("hub_id", key)
        if cfg.get("type") == "api":
            label += " (API)"
        selected[key] = st.sidebar.checkbox(label, value=enabled, key=f"model_{key}")

    settings["selected_models"] = {k: v for k, v in selected.items() if v}

    st.sidebar.markdown("---")

    # Azure API key
    if selected.get("azure_speech"):
        settings["azure_key"] = st.sidebar.text_input(
            "Azure Speech API Key",
            value=os.environ.get("AZURE_SPEECH_KEY", ""),
            type="password",
        )
        settings["azure_region"] = st.sidebar.selectbox(
            "Azure Region",
            ["uaenorth", "eastus", "westeurope", "southeastasia"],
            index=0,
        )

    st.sidebar.markdown("---")

    # Custom model
    st.sidebar.subheader("Add Custom HF Model")
    custom_id = st.sidebar.text_input("HuggingFace Model ID", placeholder="org/model-name")
    if custom_id and st.sidebar.button("Add Model"):
        settings["custom_model_id"] = custom_id
    else:
        settings["custom_model_id"] = None

    st.sidebar.markdown("---")

    # Language hint
    settings["language"] = st.sidebar.selectbox(
        "Language Hint",
        [None, "ar", "en", "ar-AE", "ar-SA"],
        index=0,
        format_func=lambda x: "Auto-detect" if x is None else x,
    )

    # Context injection
    settings["context"] = st.sidebar.text_area(
        "Context / Prompt",
        placeholder="e.g. attendee names, department terms...",
        help="Injected as prompt for models that support context (Whisper, VibeVoice-ASR).",
    )

    return settings


def render_audio_player(audio_path: str):
    """Render audio player and file info."""
    try:
        info = get_audio_info(audio_path)
        cols = st.columns(4)
        cols[0].metric("Duration", f"{info['duration_minutes']:.1f} min")
        cols[1].metric("Sample Rate", f"{info['sample_rate']} Hz")
        cols[2].metric("Channels", info["channels"])
        cols[3].metric("File Size", f"{info['file_size_mb']:.1f} MB")

        with open(audio_path, "rb") as f:
            st.audio(f.read(), format=f"audio/{Path(audio_path).suffix.lstrip('.')}")
    except Exception as e:
        st.error(f"Cannot read audio file: {e}")


def render_transcript_column(result: TranscriptionResult, col):
    """Render a single model's transcription result in a column."""
    with col:
        st.markdown(f"**{result.model_name}**")

        if result.error:
            st.error(f"Error: {result.error}")
            return

        # Metrics row
        m1, m2 = st.columns(2)
        m1.metric("Time", f"{result.processing_time_seconds:.1f}s")
        if result.rtf is not None:
            m2.metric("RTF", f"{result.rtf:.3f}")

        # Full transcript
        st.markdown("**Transcript:**")
        # Detect if Arabic-heavy → RTL
        text = result.text
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        if arabic_chars > len(text) * 0.3:
            st.markdown(
                f'<div dir="rtl" style="text-align:right; font-size:1.1em; '
                f'line-height:1.8; background:#f8f9fa; padding:12px; '
                f'border-radius:8px;">{text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.text_area("", text, height=200, key=f"txt_{result.model_name}_{id(result)}")

        # Segments with timestamps
        if result.segments:
            with st.expander(f"Segments ({len(result.segments)})"):
                for seg in result.segments:
                    speaker_tag = f"[{seg.speaker}] " if seg.speaker else ""
                    conf = f" ({seg.confidence:.2f})" if seg.confidence else ""
                    st.markdown(
                        f"`{seg.start:.1f}s–{seg.end:.1f}s` "
                        f"{speaker_tag}{seg.text}{conf}"
                    )


def render_comparison_table(results: list[TranscriptionResult]):
    """Render a comparison table across models."""
    if not results:
        return

    rows = []
    for r in results:
        rows.append(
            {
                "Model": r.model_name,
                "Processing Time (s)": round(r.processing_time_seconds, 2),
                "RTF": round(r.rtf, 4) if r.rtf else "N/A",
                "Time/min (s)": (
                    round(r.processing_time_per_minute, 2)
                    if r.processing_time_per_minute
                    else "N/A"
                ),
                "Words": len(r.text.split()),
                "Segments": len(r.segments),
                "Speakers": len({s.speaker for s in r.segments if s.speaker}),
                "Language": r.language_detected or "N/A",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Single File Mode
# ---------------------------------------------------------------------------

def run_single_file_mode(config: dict, settings: dict):
    """Single file upload and comparison mode."""
    st.header("Single File Comparison")

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=[fmt.lstrip(".") for fmt in SUPPORTED_FORMATS],
        help="Supported: WAV, MP3, M4A, FLAC, OGG — up to 60 minutes",
    )

    # Optional ground truth
    gt_uploaded = st.file_uploader(
        "Upload ground truth JSON (optional)",
        type=["json"],
        help="JSON with 'transcript', 'language', and optional 'segments' array.",
    )

    if not uploaded:
        st.info("Upload an audio file to begin comparison.")
        return

    # Save uploaded file to temp
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        audio_path = tmp.name

    # Validate
    err = validate_audio_file(audio_path)
    if err:
        st.error(err)
        return

    render_audio_player(audio_path)

    # Parse ground truth if provided
    ground_truth = None
    if gt_uploaded:
        try:
            gt_data = json.load(gt_uploaded)
            gt_segments = [
                Segment(
                    start=s.get("start", 0),
                    end=s.get("end", 0),
                    text=s.get("text", ""),
                    speaker=s.get("speaker"),
                    language=s.get("language"),
                )
                for s in gt_data.get("segments", [])
            ]
            ground_truth = GroundTruth(
                language=gt_data.get("language", ""),
                transcript=gt_data.get("transcript", ""),
                segments=gt_segments,
            )
            st.success(f"Ground truth loaded: {ground_truth.language}, {len(ground_truth.transcript)} chars")
        except Exception as e:
            st.warning(f"Could not parse ground truth: {e}")

    st.markdown("---")

    # Context injection comparison for VibeVoice-ASR
    selected = settings["selected_models"]
    registry = get_model_registry(config)

    context = settings.get("context", "").strip()
    compare_context = False
    if context and any(
        registry.get(k, {}).get("supports_context") for k in selected
    ):
        compare_context = st.checkbox(
            "Compare WITH and WITHOUT context injection",
            value=False,
            help="Runs context-supporting models twice for comparison.",
        )

    # Run button
    if not st.button("Run Transcription", type="primary"):
        return

    results = []

    progress = st.progress(0.0)
    status = st.status("Running transcriptions...", expanded=True)

    model_keys = list(selected.keys())

    # Add custom model if specified
    custom_id = settings.get("custom_model_id")
    if custom_id:
        model_keys.append(f"custom:{custom_id}")

    total_runs = len(model_keys)
    if compare_context:
        context_models = [
            k
            for k in model_keys
            if registry.get(k, {}).get("supports_context")
        ]
        total_runs += len(context_models)

    for i, key in enumerate(model_keys):
        is_custom = key.startswith("custom:")
        if is_custom:
            model_name = key.split(":", 1)[1]
            cfg = {"hub_id": model_name, "class": "GenericHFModel"}
        else:
            cfg = registry.get(key, {})
            model_name = cfg.get("hub_id", key)

        status.write(f"Loading {model_name}...")

        try:
            model = create_model(key if not is_custom else model_name, cfg, **settings)
            model.load()

            status.write(f"Transcribing with {model_name}...")
            result = model.transcribe(
                audio_path,
                language=settings.get("language"),
                context=context if context else None,
            )
            results.append(result)

            # Context comparison: run again without context
            if compare_context and cfg.get("supports_context") and context:
                status.write(f"Transcribing with {model_name} (no context)...")
                result_no_ctx = model.transcribe(
                    audio_path,
                    language=settings.get("language"),
                    context=None,
                )
                result_no_ctx.model_name += " (no context)"
                results.append(result_no_ctx)
                # Rename the contextual one
                result.model_name += " (with context)"

            model.unload()

        except Exception as e:
            results.append(
                TranscriptionResult(
                    text="",
                    model_name=model_name,
                    error=str(e),
                )
            )
            logger.error("Model %s failed: %s", model_name, e)

        progress.progress((i + 1) / total_runs)

    status.update(label="Transcription complete!", state="complete")
    st.session_state.results = results

    # Display results side by side
    st.markdown("---")
    st.subheader("Results")

    render_comparison_table(results)

    # Side-by-side transcripts
    valid_results = [r for r in results if not r.error]
    if valid_results:
        cols = st.columns(min(len(valid_results), 3))
        for i, result in enumerate(valid_results):
            render_transcript_column(result, cols[i % len(cols)])

    # WER against ground truth
    if ground_truth:
        st.markdown("---")
        st.subheader("WER Analysis (vs Ground Truth)")

        wer_rows = []
        for result in valid_results:
            wer_result = calculate_wer_arabic(ground_truth.transcript, result.text)
            wer_rows.append(
                {
                    "Model": result.model_name,
                    "WER (Overall)": f"{wer_result.wer_overall:.2%}",
                    "CER (Overall)": f"{wer_result.cer_overall:.2%}",
                    "WER (Arabic)": (
                        f"{wer_result.wer_arabic:.2%}"
                        if wer_result.wer_arabic is not None
                        else "N/A"
                    ),
                    "WER (English)": (
                        f"{wer_result.wer_english:.2%}"
                        if wer_result.wer_english is not None
                        else "N/A"
                    ),
                }
            )

        st.dataframe(pd.DataFrame(wer_rows), use_container_width=True)

        # WER chart
        chart_data = []
        for result in valid_results:
            wer_result = calculate_wer_arabic(ground_truth.transcript, result.text)
            chart_data.append({"Model": result.model_name, "Metric": "WER Overall", "Value": wer_result.wer_overall})
            if wer_result.wer_arabic is not None:
                chart_data.append({"Model": result.model_name, "Metric": "WER Arabic", "Value": wer_result.wer_arabic})
            if wer_result.wer_english is not None:
                chart_data.append({"Model": result.model_name, "Metric": "WER English", "Value": wer_result.wer_english})

        if chart_data:
            fig = px.bar(
                pd.DataFrame(chart_data),
                x="Model",
                y="Value",
                color="Metric",
                barmode="group",
                title="Word Error Rate Comparison",
                labels={"Value": "WER"},
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # DER if diarization available
    diar_results = [r for r in valid_results if any(s.speaker for s in r.segments)]
    if diar_results and ground_truth and ground_truth.segments:
        st.markdown("---")
        st.subheader("Diarization Error Rate")

        der_rows = []
        for result in diar_results:
            der_result = calculate_der(ground_truth.segments, result.segments)
            if der_result:
                der_rows.append(
                    {
                        "Model": result.model_name,
                        "DER": f"{der_result.der:.2%}",
                        "Missed Speech": f"{der_result.missed_speech:.2%}",
                        "False Alarm": f"{der_result.false_alarm:.2%}",
                        "Confusion": f"{der_result.confusion:.2%}",
                    }
                )

        if der_rows:
            st.dataframe(pd.DataFrame(der_rows), use_container_width=True)

    # Cleanup temp file
    try:
        os.unlink(audio_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Batch Benchmark Mode
# ---------------------------------------------------------------------------

def run_batch_mode(config: dict, settings: dict):
    """Batch benchmarking across a corpus folder."""
    st.header("Batch Benchmark")

    corpus_dir = st.text_input(
        "Test Corpus Directory",
        value=os.path.join(os.path.dirname(__file__), "test_corpus"),
        help="Directory containing audio files and matching .json ground truth files.",
    )

    if not os.path.isdir(corpus_dir):
        st.warning(f"Directory not found: {corpus_dir}")
        return

    audio_files = list_audio_files(corpus_dir)
    if not audio_files:
        st.warning("No supported audio files found in the directory.")
        return

    # Show corpus summary
    st.write(f"Found **{len(audio_files)}** audio files:")
    file_info = []
    for af in audio_files:
        gt_path = find_ground_truth(af)
        file_info.append(
            {
                "File": os.path.basename(af),
                "Ground Truth": "Yes" if gt_path else "No",
                "Duration": f"{get_audio_duration(af) / 60:.1f} min",
            }
        )
    st.dataframe(pd.DataFrame(file_info), use_container_width=True)

    if not st.button("Run Batch Benchmark", type="primary"):
        return

    selected = settings["selected_models"]
    registry = get_model_registry(config)
    all_metrics: list[MetricsResult] = []

    total_steps = len(audio_files) * len(selected)
    progress = st.progress(0.0)
    status = st.status("Running batch benchmark...", expanded=True)
    step = 0

    for model_key in selected:
        cfg = registry.get(model_key, {})
        model_name = cfg.get("hub_id", model_key)

        status.write(f"Loading {model_name}...")
        try:
            model = create_model(model_key, cfg, **settings)
            model.load()
        except Exception as e:
            status.write(f"Failed to load {model_name}: {e}")
            step += len(audio_files)
            progress.progress(step / total_steps)
            continue

        for audio_path in audio_files:
            fname = os.path.basename(audio_path)
            status.write(f"[{model_name}] Transcribing {fname}...")

            result = model.transcribe(
                audio_path,
                language=settings.get("language"),
                context=settings.get("context") or None,
            )

            gt_path = find_ground_truth(audio_path)
            if gt_path:
                gt = GroundTruth.from_json(gt_path)
                metrics = compute_all_metrics(result, gt, fname)
            else:
                metrics = MetricsResult(
                    model_name=result.model_name,
                    audio_file=fname,
                    language="",
                    processing_time_seconds=result.processing_time_seconds,
                    audio_duration_seconds=result.audio_duration_seconds,
                    rtf=result.rtf,
                    processing_time_per_minute=result.processing_time_per_minute,
                )

            all_metrics.append(metrics)
            step += 1
            progress.progress(step / total_steps)

        model.unload()

    status.update(label="Batch benchmark complete!", state="complete")
    st.session_state.batch_results = all_metrics

    # Display results
    render_batch_results(all_metrics)


def render_batch_results(all_metrics: list[MetricsResult]):
    """Display batch benchmark results."""
    if not all_metrics:
        return

    df = metrics_to_dataframe(all_metrics)

    st.markdown("---")
    st.subheader("Batch Results")

    # Summary: Model × Metric matrix
    st.markdown("### Summary (Averages)")
    summary_cols = [
        "model_name",
        "wer_overall",
        "wer_arabic",
        "wer_english",
        "rtf",
        "processing_time_per_minute",
    ]
    available = [c for c in summary_cols if c in df.columns]
    summary = df[available].groupby("model_name").mean(numeric_only=True).round(4)
    st.dataframe(summary, use_container_width=True)

    # WER comparison chart
    if "wer_overall" in df.columns:
        st.markdown("### WER Comparison")
        fig_wer = px.bar(
            df,
            x="audio_file",
            y="wer_overall",
            color="model_name",
            barmode="group",
            title="WER by File and Model",
            labels={"wer_overall": "WER", "audio_file": "Audio File", "model_name": "Model"},
        )
        fig_wer.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_wer, use_container_width=True)

    # Arabic vs English WER breakdown
    if "wer_arabic" in df.columns and "wer_english" in df.columns:
        st.markdown("### Arabic vs English WER")
        lang_data = []
        for _, row in df.iterrows():
            if pd.notna(row.get("wer_arabic")):
                lang_data.append(
                    {"Model": row["model_name"], "Language": "Arabic", "WER": row["wer_arabic"]}
                )
            if pd.notna(row.get("wer_english")):
                lang_data.append(
                    {"Model": row["model_name"], "Language": "English", "WER": row["wer_english"]}
                )
        if lang_data:
            fig_lang = px.bar(
                pd.DataFrame(lang_data),
                x="Model",
                y="WER",
                color="Language",
                barmode="group",
                title="Average WER by Language",
            )
            fig_lang.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_lang, use_container_width=True)

    # Processing time
    if "processing_time_per_minute" in df.columns:
        st.markdown("### Processing Speed")
        fig_speed = px.bar(
            df.groupby("model_name")["processing_time_per_minute"]
            .mean()
            .reset_index(),
            x="model_name",
            y="processing_time_per_minute",
            title="Average Processing Time per Minute of Audio",
            labels={
                "processing_time_per_minute": "Seconds / minute",
                "model_name": "Model",
            },
        )
        st.plotly_chart(fig_speed, use_container_width=True)

    # RTF comparison
    if "rtf" in df.columns:
        st.markdown("### Real-Time Factor (RTF)")
        fig_rtf = px.bar(
            df.groupby("model_name")["rtf"].mean().reset_index(),
            x="model_name",
            y="rtf",
            title="Average RTF (lower = faster)",
            labels={"rtf": "RTF", "model_name": "Model"},
        )
        fig_rtf.add_hline(y=1.0, line_dash="dash", annotation_text="Real-time")
        st.plotly_chart(fig_rtf, use_container_width=True)

    # Detail table
    st.markdown("### Detailed Results")
    st.dataframe(df, use_container_width=True)

    # Export
    st.markdown("### Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=export_csv_bytes(all_metrics),
            file_name="asr_benchmark_results.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download HTML Report",
            data=export_html_bytes(all_metrics),
            file_name="asr_benchmark_report.html",
            mime="text/html",
        )


# ---------------------------------------------------------------------------
# Results Dashboard Mode
# ---------------------------------------------------------------------------

def run_dashboard_mode():
    """Display results dashboard from previous runs."""
    st.header("Results Dashboard")

    # Check for saved results
    batch_results = st.session_state.get("batch_results", [])
    single_results = st.session_state.get("results", [])

    if not batch_results and not single_results:
        st.info(
            "No results available. Run a Single File Comparison or Batch Benchmark first, "
            "or upload a previous results CSV."
        )

        uploaded_csv = st.file_uploader("Upload previous results CSV", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df, use_container_width=True)

            # Recreate charts from CSV data
            _render_dashboard_from_df(df)
        return

    if batch_results:
        st.subheader("Batch Benchmark Results")
        render_batch_results(batch_results)

    if single_results:
        st.subheader("Single File Results")
        render_comparison_table(single_results)

        # Processing time chart
        valid = [r for r in single_results if not r.error]
        if valid:
            fig = px.bar(
                pd.DataFrame(
                    [
                        {
                            "Model": r.model_name,
                            "Processing Time (s)": r.processing_time_seconds,
                        }
                        for r in valid
                    ]
                ),
                x="Model",
                y="Processing Time (s)",
                title="Processing Time Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_dashboard_from_df(df: pd.DataFrame):
    """Render dashboard charts from a DataFrame (e.g., uploaded CSV)."""
    if "wer_overall" in df.columns and "model_name" in df.columns:
        st.markdown("### WER Overview")
        fig = px.bar(
            df.groupby("model_name")["wer_overall"].mean().reset_index(),
            x="model_name",
            y="wer_overall",
            title="Average WER by Model",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if "rtf" in df.columns and "model_name" in df.columns:
        st.markdown("### RTF Overview")
        fig = px.bar(
            df.groupby("model_name")["rtf"].mean().reset_index(),
            x="model_name",
            y="rtf",
            title="Average RTF by Model",
        )
        fig.add_hline(y=1.0, line_dash="dash", annotation_text="Real-time")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Anees ASR Benchmark",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Anees ASR Benchmark")
    st.caption(
        "Compare ASR models for Arabic (MSA + Khaleeji), English, "
        "and code-switched transcription."
    )

    init_session()
    config = load_config()
    settings = render_sidebar(config)

    mode = settings["mode"]

    if mode == "Single File Comparison":
        run_single_file_mode(config, settings)
    elif mode == "Batch Benchmark":
        run_batch_mode(config, settings)
    elif mode == "Results Dashboard":
        run_dashboard_mode()


if __name__ == "__main__":
    main()
