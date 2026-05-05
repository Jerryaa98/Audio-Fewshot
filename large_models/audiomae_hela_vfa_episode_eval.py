#!/usr/bin/env python3
"""AudioMAE HELA-VFA episodic few-shot evaluation (frozen backbone).

Adds HELA-VFA (Hellinger-distance prototype classifier) to the AudioMAE eval
pipeline alongside proto/baseline/baseline++/dn4/mcl. Reuses cache and samplers
from ``audiomae_episode_eval``; reuses HELA-VFA core from
``clap_hela_vfa_episode_eval``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

import audiomae_episode_eval as aee
import clap_hela_vfa_episode_eval as ch  # hellinger_dist, hela_vfa_accuracy, hesim_loss
from bpa.balanced_pairwise_affinities import BPA
from protolp_head import add_protolp_args
from transductive_heads import add_transductive_head_args, extend_algorithm_choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioMAE HELA-VFA IID/OOD episodic evaluation")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--sorted-root", type=str, default=None)
    parser.add_argument("--spec-root-name", type=str, default="KOS_1_alpha_spec")
    parser.add_argument("--sorted-root-name", type=str, default="Sorted")

    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=4)
    parser.add_argument("--q-query", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-seed", type=int, default=42)

    parser.add_argument("--scenarios", nargs="+", default=["iid", "ood"], choices=["iid", "ood"])
    parser.add_argument(
        "--algorithms",
        nargs="+",
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

    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--audiomae-model-id",
        type=str,
        default="hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k",
    )
    parser.add_argument("--audiomae-num-classes", type=int, default=0)
    parser.add_argument("--embedding-type", type=str, default="global", choices=["global", "frame-mean"])
    parser.add_argument("--finetuned-ckpt", type=str, default=None,
                        help="Optional checkpoint from audiomae_hela_vfa_finetune.py or audiomae_finetune_lastk.py")

    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--num-mel-bins", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--mean", type=float, default=-4.2677393)
    parser.add_argument("--std", type=float, default=4.5689974)

    parser.add_argument("--cache-path", type=str, default="cache/audiomae_test_embeddings.npz",
                        help="Shared cache format with audiomae_episode_eval.py")
    parser.add_argument("--recompute-cache", action="store_true")

    parser.add_argument("--output-csv", type=str, default="results/audiomae_hela_vfa_iid_ood_test_eval.csv")
    parser.add_argument("--save-episodes-json", type=str, default=None)

    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


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
    args: argparse.Namespace,
) -> float:
    if algorithm == "hela_vfa":
        return ch.hela_vfa_accuracy(features, support_idx, support_y, query_idx, query_y, n_way)
    if algorithm == "proto" and args.proto_bpa:
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
        algorithm=algorithm,
        features=features,
        map_features=map_features,
        support_idx=support_idx,
        support_y=support_y,
        query_idx=query_idx,
        query_y=query_y,
        n_way=n_way,
        device=device,
        args=args,
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
    aee.set_seed(args.episode_seed)
    runtime_device = aee.resolve_runtime_device(args.device)

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    sorted_root = Path(args.sorted_root) if args.sorted_root else aee.infer_sorted_root(
        data_root, args.spec_root_name, args.sorted_root_name,
    )

    test_classes = aee.read_test_classes(split_file)
    samples = aee.build_samples(data_root=data_root, sorted_root=sorted_root, test_classes=test_classes)
    class_to_indices, class_bg_groups = aee.build_index(samples)

    classes_available = [c for c in test_classes if c in class_to_indices and len(class_to_indices[c]) > 0]
    if len(classes_available) < args.n_way:
        raise ValueError(f"Only {len(classes_available)} non-empty test classes found; n_way={args.n_way}")

    cache_path = Path(args.cache_path)
    features, map_features = aee.load_or_build_embeddings(
        samples=samples,
        cache_path=cache_path,
        recompute=args.recompute_cache,
        args=args,
        device=runtime_device,
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

            for _ in tqdm(range(args.episodes), desc=f"{scenario.upper()} | {algorithm}", leave=True):
                try:
                    if scenario == "iid":
                        s_idx, s_y, q_idx, q_y, chosen_classes = aee.sample_iid_episode(
                            classes=classes_available,
                            class_to_indices=class_to_indices,
                            n_way=args.n_way,
                            k_shot=args.k_shot,
                            q_query=args.q_query,
                        )
                    else:
                        s_idx, s_y, q_idx, q_y, chosen_classes = aee.sample_ood_episode(
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
                    map_features=map_features,
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

            mean, ci_val = aee.compute_ci95(accs)
            rows.append(
                {
                    "embedding_backend": "audiomae",
                    "embedding_model": args.audiomae_model_id,
                    "embedding_type": args.embedding_type,
                    "scenario": scenario,
                    "algorithm": algorithm,
                    "accuracy": f"{mean:.4f}",
                    "ci95": f"{ci_val:.4f}",
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

    print("=== AudioMAE HELA-VFA IID/OOD Episodic Results ===")
    for row in rows:
        print(
            f"{row['scenario']:<4} {row['algorithm']:<10} "
            f"acc={float(row['accuracy']) * 100:.2f}% ± {float(row['ci95']) * 100:.2f}%  "
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
    if args.self_test:
        ch._self_test()
        return
    run_eval(args)


if __name__ == "__main__":
    main()
