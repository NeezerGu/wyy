"""Convert a Whisper model to Core ML encoder/decoder pairs.

This script targets Hugging Face Whisper checkpoints (e.g., openai/whisper-small).
It exports FP16 Core ML models by default and supports optional linear INT8
quantization. Alignment/VAD layers are intentionally excluded for a simpler
first pass; add them later once base conversion is stable.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    import coremltools as ct
except ImportError as exc:  # pragma: no cover - guidance only
    raise SystemExit("coremltools is required for conversion: pip install coremltools>=7.0") from exc


def _load_model(model_id: str, device: str) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model


def _convert_encoder(model: WhisperForConditionalGeneration, sample_length: int, compute_units: str,
                     quantize: bool, output_dir: pathlib.Path) -> pathlib.Path:
    encoder = model.model.encoder
    dummy_audio = torch.randn(1, sample_length, device=next(model.parameters()).device)

    traced = torch.jit.trace(encoder, dummy_audio)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio_features", shape=dummy_audio.shape)],
        compute_units=getattr(ct.ComputeUnit, compute_units),
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quantize:
        mlmodel = ct.compression_utils.linear_quantize_weights(mlmodel, nbits=8)

    path = output_dir / "whisper_encoder.mlpackage"
    mlmodel.save(path)
    return path


def _convert_decoder(model: WhisperForConditionalGeneration, max_tokens: int, vocab_size: int, compute_units: str,
                     quantize: bool, output_dir: pathlib.Path) -> pathlib.Path:
    decoder = model.model.decoder
    # Inputs: previous tokens (batch=1, seq_len=max_tokens) and encoder hidden states.
    dummy_tokens = torch.ones(1, max_tokens, device=next(model.parameters()).device, dtype=torch.long)
    dummy_encoder = torch.randn(1, 1500, model.config.d_model, device=next(model.parameters()).device)

    traced = torch.jit.trace(decoder, (dummy_tokens, dummy_encoder))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="tokens", shape=dummy_tokens.shape, dtype=int),
            ct.TensorType(name="encoder_hidden_states", shape=dummy_encoder.shape),
        ],
        outputs=[
            ct.TensorType(name="logits", shape=(1, max_tokens, vocab_size)),
        ],
        compute_units=getattr(ct.ComputeUnit, compute_units),
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quantize:
        mlmodel = ct.compression_utils.linear_quantize_weights(mlmodel, nbits=8)

    path = output_dir / "whisper_decoder.mlpackage"
    mlmodel.save(path)
    return path


def convert(model_id: str, output_dir: pathlib.Path, sample_length: int, max_tokens: int, compute_units: str,
            quantize: bool, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_id, device=device)

    print(f"Loaded {model_id} to {device}")
    print("Exporting encoder…")
    encoder_path = _convert_encoder(model, sample_length, compute_units, quantize, output_dir)
    print(f"Saved encoder to {encoder_path}")

    print("Exporting decoder…")
    decoder_path = _convert_decoder(model, max_tokens, model.config.vocab_size, compute_units, quantize, output_dir)
    print(f"Saved decoder to {decoder_path}")

    processor = WhisperProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir / "processor")
    print("Saved processor (tokenizer + feature extractor)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper to Core ML encoder/decoder mlpackages")
    parser.add_argument("model_id", help="Hugging Face model id, e.g. openai/whisper-small")
    parser.add_argument("output_dir", type=pathlib.Path, help="Where to place mlpackages")
    parser.add_argument("--sample-length", type=int, default=3000, help="Dummy audio frames for tracing (per chunk)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max decoder tokens for tracing")
    parser.add_argument("--compute-units", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU"], default="ALL")
    parser.add_argument("--quantize", action="store_true", help="Apply 8-bit linear quantization to weights")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu, mps, cuda)")

    args = parser.parse_args()
    convert(
        model_id=args.model_id,
        output_dir=args.output_dir,
        sample_length=args.sample_length,
        max_tokens=args.max_tokens,
        compute_units=args.compute_units,
        quantize=args.quantize,
        device=args.device,
    )
