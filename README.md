# WhisperX on-device toolkit (skeleton)

This repo contains scripts and Swift scaffolding to evaluate Whisper/WhisperX models for on-device iOS deployment.

## Contents
- `docs/whisperx_coreml_plan.md`: end-to-end plan for sizing, conversion, and measurement.
- `scripts/convert_whisper_to_coreml.py`: convert Hugging Face Whisper checkpoints to Core ML encoder/decoder packages.
- `scripts/profile_whisper.py`: measure model size, latency, and memory; optional PyTorch Lite export for Metal.
- `ios/Transcriber`: SwiftUI skeleton with `AVAudioSession` capture and background transcription service.

## Quickstart
1. Use `scripts/profile_whisper.py` to choose a starting checkpoint (medium/small recommended for real-time).
2. Convert the selected checkpoint with `scripts/convert_whisper_to_coreml.py` (FP16 by default; `--quantize` for INT8).
3. Drop the resulting `.mlpackage` files into the Xcode project bundle or unpack on first launch.
4. Wire the Swift `TranscriberService` to your Core ML or PyTorch Lite runtime and test on-device with 10–30s clips.
