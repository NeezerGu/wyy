# WhisperX on-device evaluation and Core ML conversion plan

This document outlines a practical approach to benchmark WhisperX models (e.g., Whisper large-v2) and convert them for on-device inference on iOS using Core ML. It also includes a PyTorch Lite + Metal fallback option and Swift-side integration notes.

## 1) Model sizing and latency evaluation
- Start with **medium** and **small** model variants to gauge feasibility for real-time use. Only attempt **large-v2** after establishing a baseline.
- Evaluate both FP32 and FP16 checkpoints; if available, also test INT8 dynamic quantization (CPU) for PyTorch Lite.
- Use the provided `scripts/profile_whisper.py` to measure:
  - **Model file size** and peak memory during load/inference.
  - **End-to-end latency** on 10–30s audio clips (single-segment and chunked).
  - **Throughput** (seconds processed per second) to verify real-time performance.
- Prefer GPU/ANE execution where available; fall back to CPU when Core ML or Metal is not available.

## 2) Core ML conversion (no alignment first)
- Convert the base Whisper encoder/decoder first (no alignment). Alignment (VAD/CTC) can be layered later if conversion is stable.
- Use `scripts/convert_whisper_to_coreml.py`:
  - Supports exporting Hugging Face Whisper checkpoints (e.g., `openai/whisper-small` or `openai/whisper-medium`).
  - Emits FP16 Core ML models by default; optional INT8 linear quantization reduces size at some quality cost.
  - Exports separate encoder/decoder models to simplify streaming.
- Validate conversion by running the resulting `MLModel` in a small Python harness before shipping.

## 3) PyTorch Lite + Metal fallback
- If Core ML conversion fails (especially for alignment layers), export TorchScript and convert to PyTorch Lite using `torch.utils.mobile_optimizer.optimize_for_mobile`.
- Enable the Metal backend on iOS with `torch::jit::load` for mobile or `TorchModule` in Swift.
- Measure memory and latency in the same profiling script, using the `--lite` flag.

## 4) iOS audio pipeline and model inputs
- Configure `AVAudioSession` with category `.playAndRecord`, mode `.default`, 16 kHz sample rate, and mono channel.
- Capture PCM buffers via `AVAudioEngine` tap. Use Accelerate vDSP for resampling/normalization if the hardware format differs from 16 kHz mono.
- Ensure buffer windowing aligns with model expectations (full clip or streaming chunks). Keep a small overlap if performing streaming decoding to preserve context.
- Normalize float samples to the range expected by Whisper (typically `float32` PCM in `[-1, 1]`).

## 5) App integration
- Load models from the app bundle on first launch, or unzip them into the sandbox Documents directory. Do not rely on network access.
- Encapsulate inference in a `Transcriber` service with `transcribe(audioBuffer) -> text`. Manage timestamps across chunks if streaming; return plain text when alignment is absent.
- Run inference on a background queue; surface UI state for recording, processing, completion, and error cases.
- Use the SwiftUI skeleton under `ios/Transcriber` as a starting point for the UI.

## 6) Measurement and tuning
- On an iPhone 17 device (or simulator with metric caveats), measure:
  - 10–30s clip latency (end-to-end), peak memory, and model load time.
  - Compare FP16 vs INT8 and small vs medium checkpoints.
  - Re-run after enabling alignment/VAD to understand overhead.
- Adjust model size, quantization, and whether alignment is enabled to hit real-time targets.

## 7) Data privacy
- Keep all inference local. Ensure no network paths are used during recording or transcription.

## 8) Next steps
- If a more detailed Core ML conversion script, SwiftUI implementation, or PyTorch Lite integration is needed, expand the provided samples into a full build script or Xcode target.
