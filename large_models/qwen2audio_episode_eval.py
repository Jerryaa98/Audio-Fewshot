#!/usr/bin/env python3
"""Qwen2-Audio encoder episodic few-shot evaluation with LibFewShot-style IID/OOD sampling.

Uses the Whisper-large-v3-based audio encoder from Qwen2-Audio-7B to produce
1280-D embeddings via mean-pooling over encoder hidden states.

Implements test-split-only episodic sampling for two scenarios:
- IID: support/query sampled randomly from class pool (without overlap)
- OOD: support from dominant-background group (team_a), query from other backgrounds (team_b)

Evaluation heads: proto, protolp, baseline, baseline++, dn4, mcl.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from protolp_head import add_protolp_args, protolp_accuracy_from_args
from transductive_heads import (
    add_transductive_head_args,
    extend_algorithm_choices,
    is_transductive_algorithm,
    transductive_accuracy_from_args,
)


@dataclass
class Sample:
    sample_id: str
    class_name: str
    background: str
    wav_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2-Audio encoder IID/OOD episodic evaluation on test split")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to spectrogram root (e.g., .../KOS_1_alpha_spec)")
    parser.add_argument("--split-file", type=str, required=True,
                        help="Path to class_per_split .npy (test classes taken from index 2)")
    parser.add_argument("--sorted-root", type=str, default=None,
                        help="Path to wav root (e.g., .../Sorted). If unset, inferred from data-root")
    parser.add_argument("--spec-root-name", type=str, default="KOS_1_alpha_spec")
    parser.add_argument("--sorted-root-name", type=str, default="Sorted")

    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-seed", type=int, default=42)

    parser.add_argument("--scenarios", nargs="+", default=["iid", "ood"], choices=["iid", "ood"])
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["proto", "baseline", "baseline++", "dn4"],
        choices=extend_algorithm_choices(["proto", "protolp", "baseline", "baseline++", "dn4", "mcl"]),
        help="Episode heads to evaluate on Qwen2-Audio embeddings",
    )
    parser.add_argument("--strict", action="store_true",
                        help="If set, fail on infeasible classes/episodes instead of skipping")

    parser.add_argument("--baseline-optim", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--baseline-lr", type=float, default=0.01)
    parser.add_argument("--baseline-iters", type=int, default=100)
    parser.add_argument("--baseline-batch-size", type=int, default=0,
                        help="0 means full-batch support adaptation")
    parser.add_argument("--baseline-weight-decay", type=float, default=0.0)
    parser.add_argument("--baselinepp-scale", type=float, default=2.0)
    parser.add_argument("--dn4-k", type=int, default=3,
                        help="Top-k neighbors per class for DN4-style scoring")
    parser.add_argument("--mcl-katz-factor", type=float, default=0.5)
    parser.add_argument("--mcl-gamma", type=float, default=20.0)
    parser.add_argument("--mcl-gamma2", type=float, default=20.0)
    parser.add_argument("--mcl-pool-h", type=int, default=16,
                        help="Adaptive pooled token-grid height for MCL map features")
    parser.add_argument("--mcl-pool-w", type=int, default=4,
                        help="Adaptive pooled token-grid width for MCL map features")
    add_protolp_args(parser)
    add_transductive_head_args(parser)

    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto|cpu|cuda|cuda:N (auto uses CUDA only if available)")
    parser.add_argument("--qwen2audio-model-id", type=str, default="Qwen/Qwen2-Audio-7B-Instruct",
                        help="HuggingFace model ID for Qwen2-Audio (audio encoder is extracted from this)")

    parser.add_argument("--cache-path", type=str, default="cache/qwen2audio_test_embeddings.npz")
    parser.add_argument("--recompute-cache", action="store_true")

    parser.add_argument("--output-csv", type=str, default="results/qwen2audio_iid_ood_test_eval.csv")
    parser.add_argument("--save-episodes-json", type=str, default=None,
                        help="Optional path to dump sampled episode indices")

    return parser.parse_args()


# ── Utilities ──────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_test_classes(split_file: Path) -> List[str]:
    class_per_split = np.load(split_file, allow_pickle=True)
    if len(class_per_split) < 3:
        raise ValueError(f"Invalid split file {split_file}: expected train/val/test arrays")
    return [str(c) for c in class_per_split[2]]


def infer_sorted_root(data_root: Path, spec_root_name: str, sorted_root_name: str) -> Path:
    root_str = str(data_root)
    if spec_root_name in root_str:
        return Path(root_str.replace(spec_root_name, sorted_root_name))
    return data_root.parent / sorted_root_name


def parse_background(file_name: str) -> str:
    match = re.search(r"-(.*?)_alpha=", file_name)
    return match.group(1) if match else "unknown"


def resolve_runtime_device(device_arg: str) -> str:
    import torch

    requested = device_arg.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


# ── Data building ──────────────────────────────────────────────────────────────


def build_samples(data_root: Path, sorted_root: Path, test_classes: Sequence[str]) -> List[Sample]:
    samples: List[Sample] = []
    for class_name in test_classes:
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue
        for npy_path in sorted(class_dir.glob("*.npy")):
            background = parse_background(npy_path.name)
            wav_path = sorted_root / class_name / npy_path.name.replace(".npy", ".wav")
            if not wav_path.exists():
                continue
            sample_id = f"{class_name}/{npy_path.stem}"
            samples.append(Sample(sample_id=sample_id, class_name=class_name, background=background, wav_path=wav_path))
    if not samples:
        raise RuntimeError("No test samples found after filtering. Check paths/split file.")
    return samples


def build_index(samples: Sequence[Sample]) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, List[int]]]]:
    class_to_indices: Dict[str, List[int]] = {}
    class_bg_counts: Dict[str, Dict[str, List[int]]] = {}

    for idx, sample in enumerate(samples):
        class_to_indices.setdefault(sample.class_name, []).append(idx)
        class_bg_counts.setdefault(sample.class_name, {}).setdefault(sample.background, []).append(idx)

    class_background_groups: Dict[str, Dict[str, List[int]]] = {}
    for class_name, bg_map in class_bg_counts.items():
        dominant_bg = max(bg_map.items(), key=lambda kv: len(kv[1]))[0]
        team_a = list(bg_map[dominant_bg])
        team_b: List[int] = []
        for bg_name, idxs in bg_map.items():
            if bg_name != dominant_bg:
                team_b.extend(idxs)
        class_background_groups[class_name] = {"team_a": team_a, "team_b": team_b}

    return class_to_indices, class_background_groups


def compute_ci95(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    ci = 1.96 * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return mean, ci


# ── Model loading ──────────────────────────────────────────────────────────────


def _load_qwen2audio_encoder(model_id: str, device: str):
    """Load the Whisper-based audio encoder from Qwen2-Audio, discarding the LLM."""
    import torch
    from transformers import Qwen2AudioForConditionalGeneration

    print(f"[Qwen2Audio] Loading full model to extract audio encoder: {model_id}")
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu",
    )
    encoder = full_model.audio_tower
    # Move encoder out before deleting the full model
    encoder = encoder.cpu()
    del full_model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    encoder = encoder.to(device)
    encoder.eval()
    print(f"[Qwen2Audio] Audio encoder loaded ({sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params)")
    return encoder


def _load_feature_extractor(model_id: str):
    """Load the WhisperFeatureExtractor from Qwen2-Audio processor."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    return processor.feature_extractor


# ── Embedding extraction ───────────────────────────────────────────────────────


def _qwen2audio_embed_file(
    encoder,
    feature_extractor,
    wav_path: Path,
    audio_sr: int,
    device: str,
    need_maps: bool = False,
    pool_h: int = 16,
    pool_w: int = 4,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract embedding (and optional MCL map) from a single audio file."""
    import librosa
    import torch
    import torch.nn.functional as F

    audio, _ = librosa.load(str(wav_path), sr=audio_sr, mono=True)
    audio = audio.astype(np.float32)

    inputs = feature_extractor(
        audio, sampling_rate=audio_sr, return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        out = encoder(input_features=input_features)
        hidden = out.last_hidden_state  # (1, T, 1280)

        # Mean-pool over sequence → L2 normalize
        emb = hidden.mean(dim=1)  # (1, 1280)
        emb = emb / (torch.norm(emb, p=2, dim=-1, keepdim=True) + 1e-12)
        emb_np = emb[0].detach().cpu().numpy().astype(np.float32)

        map_np = None
        if need_maps:
            tokens = hidden.transpose(1, 2)  # (1, 1280, T)
            pooled = F.adaptive_avg_pool1d(tokens, output_size=pool_h * pool_w)
            pooled = pooled.view(1, hidden.shape[-1], pool_h, pool_w)
            map_np = pooled[0].detach().cpu().numpy().astype(np.float32)

    return emb_np, map_np


# ── Cache system ───────────────────────────────────────────────────────────────


def _cache_get_scalar(cache, key: str):
    if key not in cache.files:
        return None
    return cache[key].item()


def _cache_matches(
    cache,
    *,
    qwen2audio_model_id: str,
    audio_sr: int,
    need_maps: bool,
    mcl_pool_h: int,
    mcl_pool_w: int,
) -> bool:
    if str(_cache_get_scalar(cache, "backend") or "") != "qwen2audio":
        return False

    cached_model_id = str(_cache_get_scalar(cache, "qwen2audio_model_id") or "")
    if cached_model_id and cached_model_id != qwen2audio_model_id:
        return False

    cached_sr = _cache_get_scalar(cache, "audio_sr")
    if cached_sr is not None and int(cached_sr) != int(audio_sr):
        return False

    maps_required = 1 if need_maps else 0
    maps_cached = _cache_get_scalar(cache, "maps_cached")
    if maps_cached is not None and maps_required and int(maps_cached) == 0:
        return False

    if need_maps:
        for key, val in [("mcl_pool_h", mcl_pool_h), ("mcl_pool_w", mcl_pool_w)]:
            v = _cache_get_scalar(cache, key)
            if v is not None and int(v) != int(val):
                return False

    return True


def load_or_build_embeddings(
    samples: Sequence[Sample],
    cache_path: Path,
    recompute: bool,
    qwen2audio_model_id: str,
    audio_sr: int,
    device: str,
    need_maps: bool,
    mcl_pool_h: int,
    mcl_pool_w: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sample_ids = np.array([s.sample_id for s in samples], dtype=object)

    if cache_path.exists() and not recompute:
        cache = np.load(cache_path, allow_pickle=True)
        cached_ids = cache["sample_ids"]
        if (
            len(cached_ids) == len(sample_ids)
            and np.all(cached_ids == sample_ids)
            and _cache_matches(
                cache,
                qwen2audio_model_id=qwen2audio_model_id,
                audio_sr=audio_sr,
                need_maps=need_maps,
                mcl_pool_h=mcl_pool_h,
                mcl_pool_w=mcl_pool_w,
            )
        ):
            embeddings = cache["embeddings"].astype(np.float32)
            map_features = None
            if need_maps and "map_features" in cache.files:
                map_features = cache["map_features"].astype(np.float32)
            print(f"[cache] loaded {len(embeddings)} embeddings from {cache_path}")
            return embeddings, map_features

    encoder = _load_qwen2audio_encoder(qwen2audio_model_id, device)
    feature_extractor = _load_feature_extractor(qwen2audio_model_id)

    embeddings = []
    map_features_list = []
    for sample in tqdm(samples, desc="Qwen2Audio embedding", leave=True):
        emb, map_feat = _qwen2audio_embed_file(
            encoder, feature_extractor, sample.wav_path, audio_sr, device,
            need_maps=need_maps, pool_h=mcl_pool_h, pool_w=mcl_pool_w,
        )
        embeddings.append(emb)
        if need_maps and map_feat is not None:
            map_features_list.append(map_feat)

    features = np.stack(embeddings, axis=0)
    maps_arr = np.stack(map_features_list, axis=0) if map_features_list else None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "sample_ids": sample_ids,
        "embeddings": features,
        "backend": np.asarray("qwen2audio", dtype=object),
        "qwen2audio_model_id": np.asarray(qwen2audio_model_id, dtype=object),
        "audio_sr": np.asarray(audio_sr, dtype=np.int64),
        "maps_cached": np.asarray(1 if need_maps else 0, dtype=np.int64),
        "mcl_pool_h": np.asarray(mcl_pool_h, dtype=np.int64),
        "mcl_pool_w": np.asarray(mcl_pool_w, dtype=np.int64),
    }
    if maps_arr is not None:
        save_dict["map_features"] = maps_arr
    np.savez_compressed(cache_path, **save_dict)
    return features, maps_arr


# ── Episode sampling ───────────────────────────────────────────────────────────


def sample_iid_episode(
    classes: Sequence[str],
    class_to_indices: Dict[str, List[int]],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    chosen_classes = random.sample(list(classes), n_way)
    support_idx, support_y, query_idx, query_y = [], [], [], []

    for y, class_name in enumerate(chosen_classes):
        pool = class_to_indices[class_name]
        if len(pool) < k_shot + q_query:
            raise ValueError(f"IID infeasible for class={class_name}: need {k_shot + q_query}, found {len(pool)}")
        picked = random.sample(pool, k_shot + q_query)
        support = picked[:k_shot]
        query = picked[k_shot:]

        support_idx.extend(support)
        support_y.extend([y] * k_shot)
        query_idx.extend(query)
        query_y.extend([y] * q_query)

    return (
        np.asarray(support_idx, dtype=np.int64),
        np.asarray(support_y, dtype=np.int64),
        np.asarray(query_idx, dtype=np.int64),
        np.asarray(query_y, dtype=np.int64),
        chosen_classes,
    )


def sample_ood_episode(
    classes: Sequence[str],
    class_bg_groups: Dict[str, Dict[str, List[int]]],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feasible_classes = [
        c for c in classes
        if len(class_bg_groups[c]["team_a"]) >= k_shot and len(class_bg_groups[c]["team_b"]) >= q_query
    ]
    if len(feasible_classes) < n_way:
        raise ValueError(
            f"OOD infeasible: only {len(feasible_classes)} classes have team_a>={k_shot} and team_b>={q_query}"
        )

    chosen_classes = random.sample(feasible_classes, n_way)
    support_idx, support_y, query_idx, query_y = [], [], [], []

    for y, class_name in enumerate(chosen_classes):
        team_a = class_bg_groups[class_name]["team_a"]
        team_b = class_bg_groups[class_name]["team_b"]

        support = random.sample(team_a, k_shot)
        query = random.sample(team_b, q_query)

        support_idx.extend(support)
        support_y.extend([y] * k_shot)
        query_idx.extend(query)
        query_y.extend([y] * q_query)

    return (
        np.asarray(support_idx, dtype=np.int64),
        np.asarray(support_y, dtype=np.int64),
        np.asarray(query_idx, dtype=np.int64),
        np.asarray(query_y, dtype=np.int64),
        chosen_classes,
    )


# ── Episode classification heads ──────────────────────────────────────────────


def proto_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
) -> float:
    s = features[support_idx]
    q = features[query_idx]

    protos = []
    for class_id in range(n_way):
        cls_feat = s[support_y == class_id]
        proto = cls_feat.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        protos.append(proto)
    protos = np.stack(protos, axis=0)

    logits = q @ protos.T
    pred = logits.argmax(axis=1)
    acc = float((pred == query_y).mean())
    return acc


def _episode_tensors(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    device: str,
):
    import torch

    support_feat = torch.from_numpy(features[support_idx]).float().to(device)
    support_lbl = torch.from_numpy(support_y).long().to(device)
    query_feat = torch.from_numpy(features[query_idx]).float().to(device)
    query_lbl = torch.from_numpy(query_y).long().to(device)
    return support_feat, support_lbl, query_feat, query_lbl


def baseline_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    device: str,
    optim_name: str,
    lr: float,
    iters: int,
    batch_size: int,
    weight_decay: float,
) -> float:
    import torch
    import torch.nn as nn

    support_feat, support_lbl, query_feat, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, device
    )
    feat_dim = support_feat.shape[-1]

    classifier = nn.Linear(feat_dim, n_way).to(device)
    criterion = nn.CrossEntropyLoss()
    if optim_name == "adam":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    support_n = support_feat.shape[0]
    eff_bs = support_n if batch_size <= 0 else min(batch_size, support_n)

    classifier.train()
    for _ in range(iters):
        perm = torch.randperm(support_n, device=device)
        for start in range(0, support_n, eff_bs):
            sel = perm[start:start + eff_bs]
            logits = classifier(support_feat[sel])
            loss = criterion(logits, support_lbl[sel])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        pred = classifier(query_feat).argmax(dim=1)
        acc = (pred == query_lbl).float().mean().item()
    return float(acc)


def baselinepp_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    device: str,
    optim_name: str,
    lr: float,
    iters: int,
    batch_size: int,
    weight_decay: float,
    scale: float,
) -> float:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    support_feat, support_lbl, query_feat, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, device
    )
    feat_dim = support_feat.shape[-1]

    weight = nn.Parameter(torch.randn(n_way, feat_dim, device=device) * 0.01)
    criterion = nn.CrossEntropyLoss()
    if optim_name == "adam":
        optimizer = torch.optim.Adam([weight], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD([weight], lr=lr, momentum=0.9, weight_decay=weight_decay)

    support_n = support_feat.shape[0]
    eff_bs = support_n if batch_size <= 0 else min(batch_size, support_n)

    def logits_for(x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(weight, p=2, dim=-1)
        return scale * (x_norm @ w_norm.T)

    for _ in range(iters):
        perm = torch.randperm(support_n, device=device)
        for start in range(0, support_n, eff_bs):
            sel = perm[start:start + eff_bs]
            logits = logits_for(support_feat[sel])
            loss = criterion(logits, support_lbl[sel])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred = logits_for(query_feat).argmax(dim=1)
        acc = (pred == query_lbl).float().mean().item()
    return float(acc)


def dn4_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    dn4_k: int,
) -> float:
    support_feat = features[support_idx]
    query_feat = features[query_idx]

    support_feat = support_feat / (np.linalg.norm(support_feat, axis=1, keepdims=True) + 1e-12)
    query_feat = query_feat / (np.linalg.norm(query_feat, axis=1, keepdims=True) + 1e-12)

    class_support = [support_feat[support_y == class_id] for class_id in range(n_way)]

    preds: List[int] = []
    for q in query_feat:
        scores = np.full((n_way,), -1e9, dtype=np.float64)
        for class_id in range(n_way):
            s = class_support[class_id]
            if s.size == 0:
                continue
            sims = s @ q
            top_k = min(max(dn4_k, 1), sims.shape[0])
            if top_k == sims.shape[0]:
                scores[class_id] = float(sims.sum())
            else:
                idx = np.argpartition(sims, -top_k)[-top_k:]
                scores[class_id] = float(sims[idx].sum())
        preds.append(int(scores.argmax()))

    pred_arr = np.asarray(preds, dtype=np.int64)
    acc = float((pred_arr == query_y).mean())
    return acc


def mcl_accuracy(
    map_features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    k_shot: int,
    device: str,
    katz_factor: float,
    gamma: float,
    gamma2: float,
) -> float:
    import torch

    support = torch.from_numpy(map_features[support_idx]).float().to(device)  # (S, C, H, W)
    query = torch.from_numpy(map_features[query_idx]).float().to(device)      # (Q, C, H, W)

    q_count = query.shape[0]
    _, channels, h, w = support.shape
    support = support.view(n_way, k_shot, channels, h, w).mean(dim=1)  # (N, C, H, W)

    support_xf = support.unsqueeze(0)  # (1, N, C, H, W)
    query_xf = query.unsqueeze(0)      # (1, Q, C, H, W)

    b, q, c, h, w = query_xf.shape
    s = support_xf.shape[1]
    support_xf = support_xf.view(b, s, c, h * w)
    query_xf = query_xf.view(b, q, c, h * w)

    support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1)
    query_xf = query_xf.unsqueeze(2).expand(-1, -1, s, -1, -1)

    support_xf = support_xf / (1e-16 + torch.norm(support_xf, p=2, dim=-2, keepdim=True))
    query_xf = query_xf / (1e-16 + torch.norm(query_xf, p=2, dim=-2, keepdim=True))
    sim = torch.transpose(query_xf, 3, 4) @ support_xf  # (b, q, s, M_q, M_s)

    m_q = sim.shape[-2]
    m_s = sim.shape[2] * sim.shape[-1]
    sim = sim.permute(0, 1, 3, 2, 4).contiguous().view(b * q, m_q, m_s)
    st = sim.transpose(-2, -1)

    t_sq = torch.exp(gamma * (sim - sim.max(-1, keepdim=True)[0]))
    t_sq = t_sq / (t_sq.sum(-1, keepdim=True) + 1e-12)
    t_qs = torch.exp(gamma2 * (st - st.max(-1, keepdim=True)[0]))
    t_qs = t_qs / (t_qs.sum(-1, keepdim=True) + 1e-12)

    n_examples = b * q
    zeros_ss = torch.zeros((n_examples, m_s, m_s), device=sim.device)
    zeros_qq = torch.zeros((n_examples, m_q, m_q), device=sim.device)
    transition = torch.cat(
        [
            torch.cat([zeros_ss, t_sq.transpose(-2, -1)], dim=-1),
            torch.cat([t_qs.transpose(-2, -1), zeros_qq], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(m_s + m_q, device=sim.device).unsqueeze(0).repeat(n_examples, 1, 1)
    ones = torch.ones((n_examples, m_s + m_q, 1), device=sim.device)
    katz = (torch.inverse(eye - katz_factor * transition) - eye) @ ones
    partial = katz.squeeze(-1)[:, :m_s] / (katz.squeeze(-1)[:, :m_s].sum(-1, keepdim=True) + 1e-12)
    logits = partial.view(n_examples, n_way, -1).sum(-1)

    pred = logits.argmax(dim=1).detach().cpu().numpy()
    return float((pred == query_y).mean())


# ── Episode dispatcher ─────────────────────────────────────────────────────────


def episode_accuracy(
    algorithm: str,
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    k_shot: int,
    device: str,
    args: argparse.Namespace,
    map_features: Optional[np.ndarray] = None,
) -> float:
    if algorithm == "proto":
        return proto_accuracy(features, support_idx, support_y, query_idx, query_y, n_way)
    if algorithm == "protolp":
        return protolp_accuracy_from_args(
            features, support_idx, support_y, query_idx, query_y, n_way, args, device=device,
        )
    if algorithm == "baseline":
        return baseline_accuracy(
            features, support_idx, support_y, query_idx, query_y, n_way, device,
            args.baseline_optim, args.baseline_lr, args.baseline_iters,
            args.baseline_batch_size, args.baseline_weight_decay,
        )
    if algorithm == "baseline++":
        return baselinepp_accuracy(
            features, support_idx, support_y, query_idx, query_y, n_way, device,
            args.baseline_optim, args.baseline_lr, args.baseline_iters,
            args.baseline_batch_size, args.baseline_weight_decay, args.baselinepp_scale,
        )
    if algorithm == "dn4":
        return dn4_accuracy(features, support_idx, support_y, query_idx, query_y, n_way, args.dn4_k)
    if algorithm == "mcl":
        if map_features is None:
            raise RuntimeError("MCL requires map features but none were computed (add 'mcl' to --algorithms)")
        return mcl_accuracy(
            map_features, support_idx, support_y, query_idx, query_y,
            n_way, k_shot, device, args.mcl_katz_factor, args.mcl_gamma, args.mcl_gamma2,
        )
    if is_transductive_algorithm(algorithm):
        return transductive_accuracy_from_args(
            algorithm,
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            args,
            device=device,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


# ── Main evaluation ───────────────────────────────────────────────────────────


def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.episode_seed)
    runtime_device = resolve_runtime_device(args.device)

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    sorted_root = Path(args.sorted_root) if args.sorted_root else infer_sorted_root(data_root, args.spec_root_name, args.sorted_root_name)

    test_classes = read_test_classes(split_file)
    samples = build_samples(data_root=data_root, sorted_root=sorted_root, test_classes=test_classes)
    class_to_indices, class_bg_groups = build_index(samples)

    classes_available = [c for c in test_classes if c in class_to_indices and len(class_to_indices[c]) > 0]
    if len(classes_available) < args.n_way:
        raise ValueError(f"Only {len(classes_available)} non-empty test classes found; n_way={args.n_way}")

    need_maps = "mcl" in args.algorithms
    cache_path = Path(args.cache_path)
    features, map_features = load_or_build_embeddings(
        samples=samples,
        cache_path=cache_path,
        recompute=args.recompute_cache,
        qwen2audio_model_id=args.qwen2audio_model_id,
        audio_sr=args.audio_sr,
        device=runtime_device,
        need_maps=need_maps,
        mcl_pool_h=args.mcl_pool_h,
        mcl_pool_w=args.mcl_pool_w,
    )

    rows: List[Dict] = []
    episode_dump: Dict[str, List[Dict]] = {
        f"{scenario}:{algorithm}": []
        for scenario in args.scenarios
        for algorithm in args.algorithms
    }

    for scenario in args.scenarios:
        for algorithm in args.algorithms:
            accs: List[float] = []
            skipped = 0

            episode_desc = f"{scenario.upper()} | {algorithm}"
            for _ in tqdm(range(args.episodes), desc=episode_desc, leave=True):
                try:
                    if scenario == "iid":
                        s_idx, s_y, q_idx, q_y, chosen_classes = sample_iid_episode(
                            classes=classes_available,
                            class_to_indices=class_to_indices,
                            n_way=args.n_way,
                            k_shot=args.k_shot,
                            q_query=args.q_query,
                        )
                    else:
                        s_idx, s_y, q_idx, q_y, chosen_classes = sample_ood_episode(
                            classes=classes_available,
                            class_bg_groups=class_bg_groups,
                            n_way=args.n_way,
                            k_shot=args.k_shot,
                            q_query=args.q_query,
                        )
                except ValueError:
                    if args.strict:
                        raise
                    skipped += 1
                    continue

                acc = episode_accuracy(
                    algorithm=algorithm,
                    features=features,
                    support_idx=s_idx,
                    support_y=s_y,
                    query_idx=q_idx,
                    query_y=q_y,
                    n_way=args.n_way,
                    k_shot=args.k_shot,
                    device=runtime_device,
                    args=args,
                    map_features=map_features,
                )
                accs.append(acc)

                if args.save_episodes_json:
                    episode_dump[f"{scenario}:{algorithm}"].append(
                        {
                            "classes": chosen_classes,
                            "support_indices": s_idx.tolist(),
                            "query_indices": q_idx.tolist(),
                            "accuracy": acc,
                        }
                    )

            mean, ci = compute_ci95(accs)
            rows.append(
                {
                    "scenario": scenario,
                    "algorithm": algorithm,
                    "accuracy": f"{mean:.4f}",
                    "ci95": f"{ci:.4f}",
                    "episodes_target": args.episodes,
                    "episodes_ran": len(accs),
                    "episodes_skipped": skipped,
                    "n_way": args.n_way,
                    "k_shot": args.k_shot,
                    "q_query": args.q_query,
                    "sampler_seed": args.episode_seed,
                    "cache_path": str(cache_path),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=== Qwen2-Audio IID/OOD Episodic Results ===")
    for row in rows:
        print(
            f"{row['scenario']:<4} {row['algorithm']:<10} "
            f"acc={float(row['accuracy'])*100:.2f}% ± {float(row['ci95'])*100:.2f}%  "
            f"ran={row['episodes_ran']}/{row['episodes_target']}"
        )
    print(f"Saved CSV: {output_csv}")

    if args.save_episodes_json:
        dump_path = Path(args.save_episodes_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(episode_dump, f)
        print(f"Saved episode dump: {dump_path}")


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
