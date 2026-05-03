#!/usr/bin/env python3
"""BEATs episodic few-shot evaluation with LibFewShot-style IID/OOD sampling.

Uses the BEATs audio encoder (90M params, 12 transformer layers, 768-D)
to produce embeddings via mean-pooling over encoder hidden states.

Checkpoint must be downloaded manually from:
  https://github.com/microsoft/unilm/tree/master/beats

Evaluation heads: proto, protolp, baseline, baseline++, dn4, mcl.
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
    parser = argparse.ArgumentParser(description="BEATs IID/OOD episodic evaluation on test split")
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
        "--algorithms", nargs="+",
        default=["proto", "baseline", "baseline++", "dn4"],
        choices=extend_algorithm_choices(["proto", "protolp", "baseline", "baseline++", "dn4", "mcl"]),
        help="Episode heads to evaluate on BEATs embeddings",
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
    parser.add_argument("--dn4-k", type=int, default=3)
    parser.add_argument("--mcl-katz-factor", type=float, default=0.5)
    parser.add_argument("--mcl-gamma", type=float, default=20.0)
    parser.add_argument("--mcl-gamma2", type=float, default=20.0)
    parser.add_argument("--mcl-pool-h", type=int, default=16)
    parser.add_argument("--mcl-pool-w", type=int, default=4)
    add_protolp_args(parser)
    add_transductive_head_args(parser)

    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto|cpu|cuda|cuda:N")
    parser.add_argument("--beats-ckpt", type=str, required=True,
                        help="Path to BEATs checkpoint .pt file (e.g., BEATs_iter3_plus_AS2M.pt)")

    parser.add_argument("--cache-path", type=str, default="cache/beats_test_embeddings.npz")
    parser.add_argument("--recompute-cache", action="store_true")

    parser.add_argument("--output-csv", type=str, default="results/beats_iid_ood_test_eval.csv")
    parser.add_argument("--save-episodes-json", type=str, default=None)

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


def _load_beats_model(beats_ckpt: str, device: str):
    """Load BEATs model from checkpoint."""
    import torch
    from beats import BEATs, BEATsConfig

    ckpt_path = Path(beats_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"BEATs checkpoint not found: {ckpt_path}\n"
            "Download from: https://github.com/microsoft/unilm/tree/master/beats"
        )

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except EOFError as exc:
        raise RuntimeError(
            f"BEATs checkpoint appears truncated or corrupted: {ckpt_path}\n"
            "Re-download or re-copy the checkpoint and try again."
        ) from exc
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print(f"[BEATs] Loaded checkpoint: {ckpt_path} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return model


# ── Embedding extraction ───────────────────────────────────────────────────────


def _beats_embed_file(
    model, wav_path: Path, audio_sr: int, device: str,
    need_maps: bool = False, pool_h: int = 16, pool_w: int = 4,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract embedding (and optional MCL map) from a single audio file."""
    import librosa
    import torch
    import torch.nn.functional as F

    audio, _ = librosa.load(str(wav_path), sr=audio_sr, mono=True)
    audio = audio.astype(np.float32)
    # BEATs patch embedding is 16x16; need at least 16 mel frames (~0.25s at 16kHz)
    min_samples = int(0.25 * audio_sr)
    if len(audio) < min_samples:
        padded = np.zeros(min_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    source = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)
    padding_mask = torch.zeros(1, source.shape[1], dtype=torch.bool, device=device)

    with torch.no_grad():
        features, _ = model.extract_features(source, padding_mask=padding_mask)
        # features: (1, time_steps, 768)

        emb = features.mean(dim=1)  # (1, 768)
        emb = emb / (torch.norm(emb, p=2, dim=-1, keepdim=True) + 1e-12)
        emb_np = emb[0].detach().cpu().numpy().astype(np.float32)

        map_np = None
        if need_maps:
            tokens = features.transpose(1, 2)  # (1, 768, T)
            pooled = F.adaptive_avg_pool1d(tokens, output_size=pool_h * pool_w)
            pooled = pooled.view(1, features.shape[-1], pool_h, pool_w)
            map_np = pooled[0].detach().cpu().numpy().astype(np.float32)

    return emb_np, map_np


# ── Cache system ───────────────────────────────────────────────────────────────


def _cache_get_scalar(cache, key: str):
    if key not in cache.files:
        return None
    return cache[key].item()


def _cache_matches(
    cache, *, beats_ckpt: str,
    audio_sr: int, need_maps: bool, mcl_pool_h: int, mcl_pool_w: int,
) -> bool:
    if str(_cache_get_scalar(cache, "backend") or "") != "beats":
        return False
    cached_ckpt = str(_cache_get_scalar(cache, "beats_ckpt") or "")
    if cached_ckpt and cached_ckpt != beats_ckpt:
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
    samples: Sequence[Sample], cache_path: Path, recompute: bool,
    beats_ckpt: str, audio_sr: int,
    device: str, need_maps: bool, mcl_pool_h: int, mcl_pool_w: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sample_ids = np.array([s.sample_id for s in samples], dtype=object)

    if cache_path.exists() and not recompute:
        cache = np.load(cache_path, allow_pickle=True)
        cached_ids = cache["sample_ids"]
        if (
            len(cached_ids) == len(sample_ids)
            and np.all(cached_ids == sample_ids)
            and _cache_matches(
                cache, beats_ckpt=beats_ckpt,
                audio_sr=audio_sr, need_maps=need_maps,
                mcl_pool_h=mcl_pool_h, mcl_pool_w=mcl_pool_w,
            )
        ):
            embeddings = cache["embeddings"].astype(np.float32)
            map_features = None
            if need_maps and "map_features" in cache.files:
                map_features = cache["map_features"].astype(np.float32)
            print(f"[cache] loaded {len(embeddings)} embeddings from {cache_path}")
            return embeddings, map_features

    model = _load_beats_model(beats_ckpt, device)

    embeddings = []
    map_features_list = []
    for sample in tqdm(samples, desc="BEATs embedding", leave=True):
        emb, map_feat = _beats_embed_file(
            model, sample.wav_path, audio_sr, device,
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
        "backend": np.asarray("beats", dtype=object),
        "beats_ckpt": np.asarray(beats_ckpt, dtype=object),
        "audio_sr": np.asarray(audio_sr, dtype=np.int64),
        "maps_cached": np.asarray(1 if need_maps else 0, dtype=np.int64),
        "mcl_pool_h": np.asarray(mcl_pool_h, dtype=np.int64),
        "mcl_pool_w": np.asarray(mcl_pool_w, dtype=np.int64),
    }
    if maps_arr is not None:
        save_dict["map_features"] = maps_arr
    np.savez_compressed(cache_path, **save_dict)
    return features, maps_arr


# ── Episode sampling (same as CLAP/Qwen2Audio) ────────────────────────────────


def sample_iid_episode(
    classes: Sequence[str], class_to_indices: Dict[str, List[int]],
    n_way: int, k_shot: int, q_query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    chosen_classes = random.sample(list(classes), n_way)
    support_idx, support_y, query_idx, query_y = [], [], [], []
    for y, class_name in enumerate(chosen_classes):
        pool = class_to_indices[class_name]
        if len(pool) < k_shot + q_query:
            raise ValueError(f"IID infeasible for class={class_name}: need {k_shot + q_query}, found {len(pool)}")
        picked = random.sample(pool, k_shot + q_query)
        support_idx.extend(picked[:k_shot])
        support_y.extend([y] * k_shot)
        query_idx.extend(picked[k_shot:])
        query_y.extend([y] * q_query)
    return (np.asarray(support_idx, dtype=np.int64), np.asarray(support_y, dtype=np.int64),
            np.asarray(query_idx, dtype=np.int64), np.asarray(query_y, dtype=np.int64), chosen_classes)


def sample_ood_episode(
    classes: Sequence[str], class_bg_groups: Dict[str, Dict[str, List[int]]],
    n_way: int, k_shot: int, q_query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feasible = [c for c in classes if len(class_bg_groups[c]["team_a"]) >= k_shot and len(class_bg_groups[c]["team_b"]) >= q_query]
    if len(feasible) < n_way:
        raise ValueError(f"OOD infeasible: only {len(feasible)} classes have team_a>={k_shot} and team_b>={q_query}")
    chosen_classes = random.sample(feasible, n_way)
    support_idx, support_y, query_idx, query_y = [], [], [], []
    for y, cn in enumerate(chosen_classes):
        support_idx.extend(random.sample(class_bg_groups[cn]["team_a"], k_shot))
        support_y.extend([y] * k_shot)
        query_idx.extend(random.sample(class_bg_groups[cn]["team_b"], q_query))
        query_y.extend([y] * q_query)
    return (np.asarray(support_idx, dtype=np.int64), np.asarray(support_y, dtype=np.int64),
            np.asarray(query_idx, dtype=np.int64), np.asarray(query_y, dtype=np.int64), chosen_classes)


# ── Episode classification heads (imported pattern) ───────────────────────────
# Re-use all heads from qwen2audio_episode_eval (identical implementation)

from qwen2audio_episode_eval import (  # noqa: E402
    proto_accuracy,
    baseline_accuracy,
    baselinepp_accuracy,
    dn4_accuracy,
    mcl_accuracy,
    _episode_tensors,
)


def episode_accuracy(
    algorithm: str, features: np.ndarray,
    support_idx: np.ndarray, support_y: np.ndarray,
    query_idx: np.ndarray, query_y: np.ndarray,
    n_way: int, k_shot: int, device: str,
    args: argparse.Namespace, map_features: Optional[np.ndarray] = None,
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
            raise RuntimeError("MCL requires map features but none were computed")
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
        samples=samples, cache_path=cache_path, recompute=args.recompute_cache,
        beats_ckpt=args.beats_ckpt,
        audio_sr=args.audio_sr, device=runtime_device,
        need_maps=need_maps, mcl_pool_h=args.mcl_pool_h, mcl_pool_w=args.mcl_pool_w,
    )

    rows: List[Dict] = []
    episode_dump: Dict[str, List[Dict]] = {
        f"{scenario}:{algorithm}": [] for scenario in args.scenarios for algorithm in args.algorithms
    }

    for scenario in args.scenarios:
        for algorithm in args.algorithms:
            accs: List[float] = []
            skipped = 0
            for _ in tqdm(range(args.episodes), desc=f"{scenario.upper()} | {algorithm}", leave=True):
                try:
                    if scenario == "iid":
                        s_idx, s_y, q_idx, q_y, chosen = sample_iid_episode(
                            classes_available, class_to_indices, args.n_way, args.k_shot, args.q_query)
                    else:
                        s_idx, s_y, q_idx, q_y, chosen = sample_ood_episode(
                            classes_available, class_bg_groups, args.n_way, args.k_shot, args.q_query)
                except ValueError:
                    if args.strict:
                        raise
                    skipped += 1
                    continue

                acc = episode_accuracy(
                    algorithm, features, s_idx, s_y, q_idx, q_y,
                    args.n_way, args.k_shot, runtime_device, args, map_features,
                )
                accs.append(acc)
                if args.save_episodes_json:
                    episode_dump[f"{scenario}:{algorithm}"].append(
                        {"classes": chosen, "support_indices": s_idx.tolist(),
                         "query_indices": q_idx.tolist(), "accuracy": acc})

            mean, ci = compute_ci95(accs)
            rows.append({
                "scenario": scenario, "algorithm": algorithm,
                "accuracy": f"{mean:.4f}", "ci95": f"{ci:.4f}",
                "episodes_target": args.episodes, "episodes_ran": len(accs),
                "episodes_skipped": skipped, "n_way": args.n_way,
                "k_shot": args.k_shot, "q_query": args.q_query,
                "sampler_seed": args.episode_seed, "cache_path": str(cache_path),
            })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=== BEATs IID/OOD Episodic Results ===")
    for row in rows:
        print(f"{row['scenario']:<4} {row['algorithm']:<10} "
              f"acc={float(row['accuracy'])*100:.2f}% ± {float(row['ci95'])*100:.2f}%  "
              f"ran={row['episodes_ran']}/{row['episodes_target']}")
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
