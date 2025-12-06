"""Profile Whisper (or WhisperX) model size, latency, and memory use.

Example usage:
    python scripts/profile_whisper.py --model openai/whisper-small --audio sample.wav \
        --device mps --chunk-length 30 --stream
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Tuple

import psutil
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_audio(path: Path, processor: WhisperProcessor, device: str) -> torch.Tensor:
    import soundfile as sf  # lazy import

    audio, sr = sf.read(path)
    if sr != processor.feature_extractor.sampling_rate:
        raise ValueError(f"Audio sample rate {sr} != expected {processor.feature_extractor.sampling_rate}")
    audio = torch.tensor(audio, dtype=torch.float32, device=device)
    return audio


def chunk_audio(audio: torch.Tensor, chunk_length: int, sr: int) -> Iterable[torch.Tensor]:
    step = chunk_length * sr
    for start in range(0, audio.shape[0], step):
        yield audio[start : start + step]


def profile(model_id: str, audio_path: Path, device: str, chunk_length: int | None, use_lite: bool) -> None:
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()

    if use_lite:
        from torch.utils.mobile_optimizer import optimize_for_mobile

        traced = torch.jit.trace(model, (torch.randn(1, 80, 3000, device=device), torch.ones(1, 1, device=device, dtype=torch.long)))
        model = optimize_for_mobile(traced)
        print("Converted to PyTorch Lite (for Metal/CPU mobile backends)")

    # Measure size on disk
    with torch.no_grad():
        tmp = Path("/tmp") / f"{model_id.replace('/', '_')}_tmp.pt"
        torch.jit.script(model if isinstance(model, torch.jit.ScriptModule) else model).save(tmp)
        size_mb = tmp.stat().st_size / (1024 * 1024)
        tmp.unlink()
    print(f"Model serialized size: {size_mb:.1f} MB")

    audio = load_audio(audio_path, processor, device)
    sr = processor.feature_extractor.sampling_rate
    chunks = list(chunk_audio(audio, chunk_length, sr)) if chunk_length else [audio]

    peak_mem = 0.0
    for idx, chunk in enumerate(chunks):
        inputs = processor(chunk.cpu().numpy(), sampling_rate=sr, return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs)
        elapsed = time.perf_counter() - start
        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        peak_mem = max(peak_mem, mem_mb)
        print(f"Chunk {idx}: {elapsed:.2f}s, RSS {mem_mb:.1f} MB")

    print(f"Peak RSS: {peak_mem:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Whisper model latency and memory")
    parser.add_argument("--model", required=True, help="Hugging Face id, e.g. openai/whisper-small")
    parser.add_argument("--audio", required=True, type=Path, help="Path to WAV/FLAC audio")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda")
    parser.add_argument("--chunk-length", type=int, default=None, help="Chunk length in seconds (None = whole clip)")
    parser.add_argument("--lite", action="store_true", help="Convert to PyTorch Lite before profiling")

    args = parser.parse_args()
    profile(args.model, args.audio, args.device, args.chunk_length, args.lite)
