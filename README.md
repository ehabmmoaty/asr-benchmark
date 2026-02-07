# Anees ASR Benchmark

Compare HuggingFace ASR models for Arabic (MSA + Khaleeji dialect), English, and code-switched Arabic/English transcription. Built for evaluating models for the Anees AI companion used by Abu Dhabi Government professionals.

## Models Supported

| Model | Type | Diarization | Context Injection | VRAM |
|---|---|---|---|---|
| Whisper Large v3 | HuggingFace | No | Yes (prompt) | ~4 GB |
| Whisper Large v3 Turbo | HuggingFace | No | Yes (prompt) | ~3 GB |
| VibeVoice-ASR (9B) | HuggingFace | Yes | Yes | ~20 GB |
| Qwen3-ASR (1.7B) | qwen-asr | No | Yes (context hints) | ~4 GB |
| XEUS (577M) | ESPnet | No | No | ~3 GB |
| Azure Speech Services | REST API | Yes | Yes (phrase list) | N/A |
| Any HF ASR model | HuggingFace | Varies | Varies | Varies |

## Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA (required for VibeVoice-ASR; Whisper and Qwen3-ASR can run on CPU)
- ffmpeg installed (`apt install ffmpeg` or `brew install ffmpeg`)

### Install

```bash
cd asr-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Optional Model Dependencies

```bash
# Qwen3-ASR (Qwen/Qwen3-ASR-1.7B)
pip install qwen-asr

# XEUS encoder (espnet/xeus) — requires ESPnet SSL branch
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
git lfs install && git clone https://huggingface.co/espnet/xeus

# DER calculation (speaker diarization evaluation)
pip install pyannote.audio
```

### Azure Speech (Optional)

Set your API key as an environment variable or enter it in the sidebar:

```bash
export AZURE_SPEECH_KEY="your-key-here"
```

## Usage

### Single File Comparison

1. Select models in the sidebar
2. Upload an audio file (WAV, MP3, M4A, FLAC, OGG — up to 60 min)
3. Optionally upload a ground truth JSON for WER calculation
4. Optionally enter context (attendee names, department terms) for context injection
5. Click "Run Transcription"
6. View side-by-side transcripts, timestamps, speaker labels, and metrics

### Batch Benchmark

1. Place audio files and matching `.json` ground truth files in `test_corpus/`
2. Select "Batch Benchmark" mode in the sidebar
3. Point to your corpus directory
4. Click "Run Batch Benchmark"
5. View WER, DER, RTF breakdowns per model, per file, per language
6. Export results as CSV or HTML report

### Ground Truth Format

Create a JSON file with the same name as your audio file:

```json
{
  "language": "ar",
  "transcript": "Full transcript text...",
  "segments": [
    {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "Segment text..."}
  ]
}
```

### Adding Custom Models

Enter any HuggingFace model ID in the sidebar under "Add Custom HF Model" (e.g., `facebook/wav2vec2-large-960h`). The app uses the `transformers` pipeline API.

To permanently add a model, edit `config.yaml`:

```yaml
models:
  my_custom_model:
    hub_id: "org/model-name"
    class: "GenericHFModel"
    enabled: true
```

## Arabic Text Normalization

WER calculation normalizes Arabic text before comparison:

- Removes diacritics (tashkeel)
- Normalizes alef variants (أ إ آ → ا)
- Normalizes taa marbuta (ة → ه)
- Normalizes hamza on carriers
- Removes tatweel (kashida)
- Lowercases English portions

This ensures fair comparison across models that produce different levels of diacritization.

## Metrics

| Metric | Description |
|---|---|
| WER | Word Error Rate (lower is better) |
| CER | Character Error Rate |
| WER Arabic | WER on Arabic-only portions |
| WER English | WER on English-only portions |
| DER | Diarization Error Rate (speaker identification accuracy) |
| RTF | Real-Time Factor (processing time / audio duration, < 1 = faster than real-time) |

## Docker Deployment

```bash
docker build -t asr-benchmark .
docker run --gpus all -p 8501:8501 \
  -e AZURE_SPEECH_KEY="your-key" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  asr-benchmark
```

For Azure VM (Standard_NC24ads_A100_v4 in UAE North):

```bash
docker run --gpus all -p 8501:8501 \
  -e AZURE_SPEECH_KEY="your-key" \
  -v /mnt/models:/root/.cache/huggingface \
  asr-benchmark
```

## Project Structure

```
asr-benchmark/
├── app.py                    # Main Streamlit app
├── config.yaml               # Model configuration
├── models/
│   ├── base.py               # Abstract model interface
│   ├── whisper_hf.py         # Whisper via HuggingFace
│   ├── vibevoice_asr.py      # VibeVoice-ASR wrapper
│   ├── azure_speech.py       # Azure Speech REST API
│   ├── qwen3_asr.py          # Qwen3-ASR via qwen-asr package
│   ├── xeus_espnet.py        # XEUS encoder via ESPnet
│   └── generic_hf.py         # Generic HF ASR model loader
├── evaluation/
│   ├── wer.py                # WER with Arabic normalization
│   ├── der.py                # Diarization error rate
│   └── metrics.py            # Aggregate metrics
├── utils/
│   ├── audio.py              # Audio loading/validation
│   ├── arabic.py             # Arabic text normalization
│   └── export.py             # CSV/HTML export
├── test_corpus/              # Sample ground truth files
├── requirements.txt
└── Dockerfile
```

## Security

- All audio processing is local — no data leaves the machine
- Azure Speech API calls route to UAE North endpoint only
- HuggingFace models are cached locally at `~/.cache/huggingface/`
