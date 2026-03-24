#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-extract CLAP embeddings from wav files.

Walks a directory of wav files, extracts 512-d CLAP embeddings for each,
and saves them as .npy files with matching directory structure. The output
directory can then be used with ``clap_data_mode: embeddings`` in the
training config.

Usage:
    python scripts/extract_clap_embeddings.py \
        --wav_root /path/to/wav/files \
        --out_root /path/to/clap/embeddings

The script mirrors the input directory structure:
    wav_root/
        classA/file1.wav  ->  out_root/classA/file1.npy  (shape [512])
        classB/file2.wav  ->  out_root/classB/file2.npy  (shape [512])
"""

import argparse
import os
import sys

import numpy as np
import torch
from glob import glob
from tqdm import tqdm


def _import_laion_clap():
    """Lazy import with the same numba workaround used in the training code."""
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key.startswith('libfewshot_core'):
            saved_modules[key] = sys.modules.pop(key)
    try:
        import laion_clap
        return laion_clap
    except ImportError:
        raise ImportError("laion-clap is required. Install with: pip install laion-clap")
    finally:
        sys.modules.update(saved_modules)


def main():
    parser = argparse.ArgumentParser(description="Pre-extract CLAP embeddings from wav files")
    parser.add_argument("--wav_root", required=True, help="Root directory containing wav files (category/file.wav)")
    parser.add_argument("--out_root", required=True, help="Output directory for .npy embedding files")
    parser.add_argument("--enable_fusion", action="store_true", help="Enable CLAP fusion mode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    laion_clap = _import_laion_clap()

    print(f"Loading CLAP model (enable_fusion={args.enable_fusion})...")
    model = laion_clap.CLAP_Module(enable_fusion=args.enable_fusion, device=args.device)
    model.load_ckpt()
    model.eval()
    print("CLAP model loaded.")

    wav_files = sorted(glob(os.path.join(args.wav_root, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"No .wav files found under {args.wav_root}")
        return

    print(f"Found {len(wav_files)} wav files. Extracting embeddings...")

    processed = 0
    skipped = 0
    for wav_path in tqdm(wav_files):
        rel_path = os.path.relpath(wav_path, args.wav_root)
        out_path = os.path.join(args.out_root, rel_path.replace(".wav", ".npy"))

        if os.path.exists(out_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with torch.no_grad():
            embedding = model.get_audio_embedding_from_filelist(
                x=[wav_path], use_tensor=False
            )

        np.save(out_path, embedding.squeeze())  # shape [512]
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped (already exist): {skipped}")
    print(f"Embeddings saved to: {args.out_root}")


if __name__ == "__main__":
    main()
