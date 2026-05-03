#!/usr/bin/env python3
"""AST episodic few-shot evaluation (frozen backbone).

Adds HELA-VFA (Hellinger-distance prototype classifier) to AST evaluation
alongside proto/baseline/baseline++/dn4/mcl and transductive heads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import audiomae_episode_eval as aee  # samplers, episode_accuracy
import clap_episode_eval as ce        # build_samples, build_index
import clap_hela_vfa_episode_eval as ch
from bpa.balanced_pairwise_affinities import BPA
from protolp_head import add_protolp_args
from transductive_heads import add_transductive_head_args, extend_algorithm_choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AST HELA-VFA IID/OOD episodic evaluation")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--sorted-root", type=str, default=None)
    parser.add_argument("--spec-root-name", type=str, default="KOS_1_alpha_spec")
    parser.add_argument("--sorted-root-name", type=str, default="Sorted")

    parser.add_argument("--ast-model-id", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")

    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-seed", type=int, default=42)

    parser.add_argument("--scenarios", nargs="+", default=["iid", "ood"], choices=["iid", "ood"])
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["hela_vfa"],
        choices=extend_algorithm_choices(["hela_vfa", "proto", "protolp", "baseline", "baseline++", "dn4", "mcl"]),
    )
    parser.add_argument(
        "--proto-bpa",
        action="store_true",
        help="Apply BPA transductive transform before proto evaluation",
    )
    parser.add_argument("--strict", action="store_true")

    parser.add_argument("--baseline-optim", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--baseline-lr", type=float, default=0.01)
    parser.add_argument("--baseline-iters", type=int, default=100)
    parser.add_argument("--baseline-batch-size", type=int, default=0)
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

    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--output-csv", type=str, default="results/ast_hela_vfa_iid_ood_test_eval.csv")
    parser.add_argument("--save-episodes-json", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable, using CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def infer_sorted_root(data_root: Path, spec_root_name: str, sorted_root_name: str) -> Path:
    root_str = str(data_root)
    if spec_root_name in root_str:
        return Path(root_str.replace(spec_root_name, sorted_root_name))
    return data_root.parent / sorted_root_name


def _extract_batch_embeddings(model, features_dict, device: str, need_maps: bool, pool_h: int, pool_w: int):
    import torch
    import torch.nn.functional as F

    inputs = {k: v.to(device) for k, v in features_dict.items()}
    with torch.no_grad():
        out = model.audio_spectrogram_transformer(**inputs)
        last_hidden = out.last_hidden_state
        emb = last_hidden[:, 0, :]
        emb = emb / (torch.norm(emb, p=2, dim=-1, keepdim=True) + 1e-12)

        map_np = None
        if need_maps:
            tokens = last_hidden[:, 1:, :]
            if tokens.shape[1] == 0:
                tokens = last_hidden
            token_seq = tokens.transpose(1, 2).contiguous()
            pooled = F.adaptive_avg_pool1d(token_seq, output_size=pool_h * pool_w)
            pooled = pooled.view(pooled.shape[0], pooled.shape[1], pool_h, pool_w)
            map_np = pooled.detach().cpu().numpy().astype(np.float32)

    emb_np = emb.detach().cpu().numpy().astype(np.float32)
    return emb_np, map_np


def _embed_samples_for_eval(model, feature_extractor, samples, sampling_rate: int, max_audio_seconds: float, device: str,
                            batch_size: int = 32, need_maps: bool = False,
                            mcl_pool_h: int = 16, mcl_pool_w: int = 4) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import librosa

    model.eval()
    max_len = int(max_audio_seconds * sampling_rate)

    feats: List[np.ndarray] = []
    maps: List[np.ndarray] = []
    for start in tqdm(range(0, len(samples), batch_size), desc="AST embedding", leave=False):
        chunk = samples[start:start + batch_size]
        waves: List[np.ndarray] = []
        for sample in chunk:
            audio, _ = librosa.load(str(sample.wav_path), sr=sampling_rate, mono=True)
            audio = audio.astype(np.float32)
            if len(audio) < max_len:
                arr = np.zeros((max_len,), dtype=np.float32)
                arr[:len(audio)] = audio
                audio = arr
            else:
                audio = audio[:max_len]
            waves.append(audio)

        batch = feature_extractor(
            waves,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        emb, fmap = _extract_batch_embeddings(
            model,
            batch,
            device=device,
            need_maps=need_maps,
            pool_h=mcl_pool_h,
            pool_w=mcl_pool_w,
        )
        feats.append(emb)
        if need_maps and fmap is not None:
            maps.append(fmap)

    feat_np = np.concatenate(feats, axis=0)
    map_np = np.concatenate(maps, axis=0) if need_maps else None
    return feat_np, map_np


def episode_accuracy(
    algorithm: str,
    features: np.ndarray,
    map_features: Optional[np.ndarray],
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    device: str,
    args,
) -> float:
    if algorithm == "hela_vfa":
        return ch.hela_vfa_accuracy(features, support_idx, support_y, query_idx, query_y, n_way)
    if algorithm == "proto" and getattr(args, "proto_bpa", False):
        bpa_features, s_count, q_count = _bpa_proto_features(
            features=features,
            support_idx=support_idx,
            support_y=support_y,
            query_idx=query_idx,
            args=args,
        )
        s_idx = np.arange(s_count, dtype=np.int64)
        q_idx = np.arange(s_count, s_count + q_count, dtype=np.int64)
        return aee.proto_accuracy(bpa_features, s_idx, support_y, q_idx, query_y, n_way)
    return aee.episode_accuracy(
        algorithm=algorithm, features=features, map_features=map_features,
        support_idx=support_idx, support_y=support_y,
        query_idx=query_idx, query_y=query_y,
        n_way=n_way, device=device, args=args,
    )


def _bpa_proto_features(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int]:
    import torch

    support_feat = features[support_idx]
    query_feat = features[query_idx]
    concat = np.concatenate([support_feat, query_feat], axis=0)
    bpa = BPA(
        distance_metric="cosine",
        ot_reg=0.1,
        num_shot=args.k_shot,
        num_way=args.n_way,
        num_query=args.q_query,
    )
    with torch.no_grad():
        bpa_feat = bpa(
            torch.from_numpy(concat).float(),
            torch.from_numpy(support_y).long(),
        )
    return bpa_feat.cpu().numpy(), support_feat.shape[0], query_feat.shape[0]


def run_eval(args: argparse.Namespace) -> None:
    if not args.data_root or not args.split_file:
        raise ValueError("--data-root and --split-file are required for evaluation")

    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    set_seed(args.episode_seed)
    device = resolve_device(args.device)

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    sorted_root = Path(args.sorted_root) if args.sorted_root else infer_sorted_root(
        data_root, args.spec_root_name, args.sorted_root_name,
    )

    test_classes = ce.read_test_classes(split_file)
    samples = ce.build_samples(data_root=data_root, sorted_root=sorted_root, test_classes=test_classes)
    class_to_indices, class_bg_groups = ce.build_index(samples)

    classes_available = [c for c in test_classes if c in class_to_indices and len(class_to_indices[c]) > 0]
    if len(classes_available) < args.n_way:
        raise ValueError(f"Only {len(classes_available)} non-empty test classes; n_way={args.n_way}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ast_model_id)
    sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
    model = ASTForAudioClassification.from_pretrained(args.ast_model_id, ignore_mismatched_sizes=True)
    model = model.to(device)
    model.eval()

    need_maps = "mcl" in args.algorithms
    features, map_features = _embed_samples_for_eval(
        model, feature_extractor, samples,
        sampling_rate=sampling_rate, max_audio_seconds=args.max_audio_seconds,
        device=str(device), batch_size=max(args.batch_size, 8),
        need_maps=need_maps, mcl_pool_h=args.mcl_pool_h, mcl_pool_w=args.mcl_pool_w,
    )

    head_args = SimpleNamespace(
        baseline_optim=args.baseline_optim, baseline_lr=args.baseline_lr,
        baseline_iters=args.baseline_iters, baseline_batch_size=args.baseline_batch_size,
        baseline_weight_decay=args.baseline_weight_decay, baselinepp_scale=args.baselinepp_scale,
        dn4_k=args.dn4_k, k_shot=args.k_shot, proto_bpa=args.proto_bpa,
        mcl_katz_factor=args.mcl_katz_factor, mcl_gamma=args.mcl_gamma, mcl_gamma2=args.mcl_gamma2,
        protolp_epochs=args.protolp_epochs, protolp_alpha=args.protolp_alpha,
        protolp_lambda=args.protolp_lambda, protolp_gamma=args.protolp_gamma,
        protolp_beta=args.protolp_beta, protolp_svd_dim=args.protolp_svd_dim,
        protolp_power=args.protolp_power, protolp_center=args.protolp_center,
        transductive_normalize=args.transductive_normalize,
        alpha_tim_iters=args.alpha_tim_iters, alpha_tim_lr=args.alpha_tim_lr,
        alpha_tim_temp=args.alpha_tim_temp, alpha_tim_alpha=args.alpha_tim_alpha,
        alpha_tim_ce_weight=args.alpha_tim_ce_weight,
        alpha_tim_marginal_weight=args.alpha_tim_marginal_weight,
        alpha_tim_conditional_weight=args.alpha_tim_conditional_weight,
        laplacian_shot_iters=args.laplacian_shot_iters,
        laplacian_shot_knn=args.laplacian_shot_knn,
        laplacian_shot_lambda=args.laplacian_shot_lambda,
        laplacian_shot_norm_type=args.laplacian_shot_norm_type,
        bdcspn_temp=args.bdcspn_temp, bdcspn_norm_type=args.bdcspn_norm_type,
        paddle_iters=args.paddle_iters, paddle_lambda=args.paddle_lambda,
        paddle_temp=args.paddle_temp,
        ecpe_epochs=args.ecpe_epochs, ecpe_lambda=args.ecpe_lambda,
        ecpe_alpha=args.ecpe_alpha, ecpe_update_rate=args.ecpe_update_rate,
        ecpe_svd_dim=args.ecpe_svd_dim, ecpe_power=args.ecpe_power,
        ecpe_center=args.ecpe_center, ecpe_balance=args.ecpe_balance,
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
                        s_idx, s_y, q_idx, q_y, chosen = ce.sample_iid_episode(
                            classes=classes_available, class_to_indices=class_to_indices,
                            n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
                    else:
                        s_idx, s_y, q_idx, q_y, chosen = ce.sample_ood_episode(
                            classes=classes_available, class_bg_groups=class_bg_groups,
                            n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
                except ValueError:
                    if args.strict:
                        raise
                    skipped += 1
                    continue

                acc = episode_accuracy(
                    algorithm=algorithm, features=features, map_features=map_features,
                    support_idx=s_idx, support_y=s_y, query_idx=q_idx, query_y=q_y,
                    n_way=args.n_way, device=str(device), args=head_args,
                )
                accs.append(acc)
                if args.save_episodes_json:
                    episode_dump[f"{scenario}:{algorithm}"].append(
                        {"classes": chosen, "support_indices": s_idx.tolist(),
                         "query_indices": q_idx.tolist(), "accuracy": acc})

            mean, ci_val = ce.compute_ci95(accs)
            rows.append({
                "scenario": scenario, "algorithm": algorithm,
                "accuracy": f"{mean:.4f}", "ci95": f"{ci_val:.4f}",
                "episodes_target": args.episodes, "episodes_ran": len(accs),
                "episodes_skipped": skipped, "n_way": args.n_way,
                "k_shot": args.k_shot, "q_query": args.q_query,
                "sampler_seed": args.episode_seed,
            })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=== AST HELA-VFA IID/OOD Episodic Results ===")
    for row in rows:
        print(f"{row['scenario']:<4} {row['algorithm']:<10} "
              f"acc={float(row['accuracy']) * 100:.2f}% ± {float(row['ci95']) * 100:.2f}%  "
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
    if args.self_test:
        ch._self_test()
        return
    run_eval(args)


if __name__ == "__main__":
    main()
