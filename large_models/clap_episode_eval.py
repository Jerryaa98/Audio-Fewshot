#!/usr/bin/env python3
"""LAION-CLAP episodic few-shot evaluation with LibFewShot-style IID/OOD sampling.

Implements test-split-only episodic sampling for two scenarios:
- IID: support/query sampled randomly from class pool (without overlap)
- OOD: support from dominant-background group (team_a), query from other backgrounds (team_b)

OOD grouping follows LibFewShot logic in `dataset.py` + `samplers.py`:
- dominant background within each class => team_a
- all non-dominant backgrounds => team_b

Evaluation heads include prototypical classification and transductive ProtoLP
over CLAP embeddings.
"""

from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="CLAP IID/OOD episodic evaluation on test split")
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
        choices=extend_algorithm_choices(["proto", "protolp", "baseline", "baseline++", "dn4"]),
        help="Episode heads to evaluate on cached CLAP embeddings",
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
    add_protolp_args(parser)
    add_transductive_head_args(parser)

    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto|cpu|cuda|cuda:N (auto uses CUDA only if available)")
    parser.add_argument("--amodel", type=str, default="HTSAT-base")
    parser.add_argument("--enable-fusion", dest="enable_fusion", action="store_true",
                        help="Enable CLAP fusion model branch")
    parser.add_argument("--disable-fusion", dest="enable_fusion", action="store_false",
                        help="Disable CLAP fusion model branch")
    parser.set_defaults(enable_fusion=None)
    parser.add_argument("--clap-ckpt", type=str, default=None,
                        help="Optional LAION-CLAP checkpoint path; if empty, use default load_ckpt()")
    parser.add_argument("--clap-model-id", type=int, default=1,
                        help="Model ID passed to load_ckpt when supported (paper default: 1)")
    parser.add_argument("--no-clap-auto-fallback", action="store_true",
                        help="Disable automatic retries across CLAP amodel/fusion configs")

    parser.add_argument("--cache-path", type=str, default="cache/clap_test_embeddings.npz")
    parser.add_argument("--recompute-cache", action="store_true")

    parser.add_argument("--output-csv", type=str, default="results/clap_iid_ood_test_eval.csv")
    parser.add_argument("--save-episodes-json", type=str, default=None,
                        help="Optional path to dump sampled episode indices")

    return parser.parse_args()


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


def _clap_embed_file(model, wav_path: Path, sr: int) -> np.ndarray:
    import librosa
    import torch

    audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    audio = audio.astype(np.float32)

    if hasattr(model, "get_audio_embedding_from_filelist"):
        emb = model.get_audio_embedding_from_filelist([str(wav_path)], use_tensor=False)
    elif hasattr(model, "get_audio_embedding_from_data"):
        try:
            emb = model.get_audio_embedding_from_data(x=[audio], use_tensor=False)
        except TypeError:
            emb = model.get_audio_embedding_from_data(x=np.expand_dims(audio, axis=0))
    else:
        raise RuntimeError("Unsupported CLAP module API: no embedding method found")

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb)
    if emb.ndim == 1:
        emb = emb[None, :]
    return emb[0]


def _safe_load_ckpt(model, clap_ckpt: str | None, clap_model_id: Optional[int]) -> None:
    kwargs = {}
    if clap_ckpt:
        kwargs["ckpt"] = clap_ckpt
    if clap_model_id is not None:
        kwargs["model_id"] = clap_model_id

    try:
        model.load_ckpt(**kwargs)
        return
    except TypeError:
        kwargs.pop("model_id", None)
        model.load_ckpt(**kwargs)


def _build_clap_model(amodel: str, enable_fusion: bool):
    import laion_clap

    return laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel)


def _resolve_clap_model(
    amodel: str,
    enable_fusion: Optional[bool],
    clap_ckpt: str | None,
    clap_model_id: Optional[int],
    no_auto_fallback: bool,
):
    tried = set()
    candidates: List[Tuple[str, bool]] = []

    if enable_fusion is None:
        candidates.append((amodel, True))
        candidates.append((amodel, False))
    else:
        candidates.append((amodel, enable_fusion))

    if not no_auto_fallback:
        alt_amodels = ["HTSAT-base", "HTSAT-tiny", "PANN-14"]
        for alt in alt_amodels:
            if alt != amodel:
                candidates.append((alt, True))
                candidates.append((alt, False))

    errors: List[str] = []
    for cand_amodel, cand_fusion in candidates:
        key = (cand_amodel, cand_fusion)
        if key in tried:
            continue
        tried.add(key)
        try:
            model = _build_clap_model(amodel=cand_amodel, enable_fusion=cand_fusion)
            _safe_load_ckpt(model=model, clap_ckpt=clap_ckpt, clap_model_id=clap_model_id)
            print(f"[CLAP] Loaded checkpoint with amodel={cand_amodel}, fusion={cand_fusion}")
            return model
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
            errors.append(f"amodel={cand_amodel}, fusion={cand_fusion} -> {msg}")

    attempts = "\n".join(f"  - {e}" for e in errors)
    raise RuntimeError(
        "Failed to load LAION-CLAP checkpoint with all attempted model configs. "
        "Try explicit flags like `--amodel HTSAT-base --enable-fusion --clap-model-id 1` "
        "or pass a matching `--clap-ckpt`. Attempts:\n"
        f"{attempts}"
    )


def resolve_runtime_device(device_arg: str) -> str:
    import torch

    requested = device_arg.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def _cache_get_scalar(cache, key: str):
    if key not in cache.files:
        return None
    return cache[key].item()


def _cache_matches(
    cache,
    *,
    amodel: str,
    enable_fusion: Optional[bool],
    clap_ckpt: Optional[str],
    clap_model_id: Optional[int],
    sr: int,
) -> bool:
    if str(_cache_get_scalar(cache, "backend") or "") not in ("", "clap"):
        return False

    cached_amodel = str(_cache_get_scalar(cache, "amodel") or "")
    if cached_amodel and cached_amodel != amodel:
        return False

    cached_fusion = _cache_get_scalar(cache, "enable_fusion")
    requested_fusion = -1 if enable_fusion is None else (1 if enable_fusion else 0)
    if cached_fusion is not None and int(cached_fusion) != requested_fusion:
        return False

    cached_ckpt = str(_cache_get_scalar(cache, "clap_ckpt") or "")
    requested_ckpt = clap_ckpt or ""
    if cached_ckpt != requested_ckpt:
        return False

    cached_model_id = _cache_get_scalar(cache, "clap_model_id")
    requested_model_id = -1 if clap_model_id is None else int(clap_model_id)
    if cached_model_id is not None and int(cached_model_id) != requested_model_id:
        return False

    cached_sr = _cache_get_scalar(cache, "sr")
    if cached_sr is not None and int(cached_sr) != int(sr):
        return False

    return True


def load_or_build_embeddings(samples: Sequence[Sample], cache_path: Path, recompute: bool,
                             amodel: str, enable_fusion: Optional[bool], clap_ckpt: str | None,
                             clap_model_id: Optional[int],
                             no_clap_auto_fallback: bool,
                             device: str, sr: int) -> np.ndarray:
    sample_ids = np.array([s.sample_id for s in samples], dtype=object)

    if cache_path.exists() and not recompute:
        cache = np.load(cache_path, allow_pickle=True)
        cached_ids = cache["sample_ids"]
        if (
            len(cached_ids) == len(sample_ids)
            and np.all(cached_ids == sample_ids)
            and _cache_matches(
                cache,
                amodel=amodel,
                enable_fusion=enable_fusion,
                clap_ckpt=clap_ckpt,
                clap_model_id=clap_model_id,
                sr=sr,
            )
        ):
            return cache["embeddings"].astype(np.float32)

    import torch

    model = _resolve_clap_model(
        amodel=amodel,
        enable_fusion=enable_fusion,
        clap_ckpt=clap_ckpt,
        clap_model_id=clap_model_id,
        no_auto_fallback=no_clap_auto_fallback,
    )

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    embeddings = []
    for sample in tqdm(samples, desc="CLAP embedding", leave=True):
        emb = _clap_embed_file(model, sample.wav_path, sr=sr)
        norm = np.linalg.norm(emb) + 1e-12
        embeddings.append((emb / norm).astype(np.float32))

    features = np.stack(embeddings, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_ids=sample_ids,
        embeddings=features,
        backend=np.asarray("clap", dtype=object),
        amodel=np.asarray(amodel, dtype=object),
        enable_fusion=np.asarray(-1 if enable_fusion is None else (1 if enable_fusion else 0), dtype=np.int64),
        clap_ckpt=np.asarray(clap_ckpt or "", dtype=object),
        clap_model_id=np.asarray(-1 if clap_model_id is None else int(clap_model_id), dtype=np.int64),
        sr=np.asarray(sr, dtype=np.int64),
    )
    return features


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


def episode_accuracy(
    algorithm: str,
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    device: str,
    args: argparse.Namespace,
) -> float:
    if algorithm == "proto":
        return proto_accuracy(features, support_idx, support_y, query_idx, query_y, n_way)
    if algorithm == "protolp":
        return protolp_accuracy_from_args(
            features, support_idx, support_y, query_idx, query_y, n_way, args, device=device,
        )
    if algorithm == "baseline":
        return baseline_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device,
            args.baseline_optim,
            args.baseline_lr,
            args.baseline_iters,
            args.baseline_batch_size,
            args.baseline_weight_decay,
        )
    if algorithm == "baseline++":
        return baselinepp_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device,
            args.baseline_optim,
            args.baseline_lr,
            args.baseline_iters,
            args.baseline_batch_size,
            args.baseline_weight_decay,
            args.baselinepp_scale,
        )
    if algorithm == "dn4":
        return dn4_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            args.dn4_k,
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

    cache_path = Path(args.cache_path)
    features = load_or_build_embeddings(
        samples=samples,
        cache_path=cache_path,
        recompute=args.recompute_cache,
        amodel=args.amodel,
        enable_fusion=args.enable_fusion,
        clap_ckpt=args.clap_ckpt,
        clap_model_id=args.clap_model_id,
        no_clap_auto_fallback=args.no_clap_auto_fallback,
        device=runtime_device,
        sr=args.sr,
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
                    device=runtime_device,
                    args=args,
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

    print("=== CLAP IID/OOD Episodic Results ===")
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
