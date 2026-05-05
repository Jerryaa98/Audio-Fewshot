#!/usr/bin/env python3
"""LAION-CLAP HELA-VFA episodic few-shot evaluation (frozen backbone).

Adds HELA-VFA (Hellinger-distance prototype classifier, WACV 2024) as an episodic
head over CLAP audio embeddings. Reuses caching/sampling logic from
``clap_episode_eval`` so the same .npz cache is interoperable.

HELA-VFA reference: HELA-VFA/HELA_VFA_main.py (Hellinger_dist prototypes -> -dist scores).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

import clap_episode_eval as ce
from bpa.balanced_pairwise_affinities import BPA
from protolp_head import add_protolp_args
from transductive_heads import add_transductive_head_args, extend_algorithm_choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLAP HELA-VFA episodic evaluation on test split")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Required unless --self-test is set")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Required unless --self-test is set")
    parser.add_argument("--sorted-root", type=str, default=None)
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
        default=["hela_vfa"],
        choices=extend_algorithm_choices(["hela_vfa", "proto", "protolp", "baseline", "baseline++", "dn4"]),
        help="Episode heads to evaluate. HELA-VFA is the default; others available for comparison.",
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
    add_protolp_args(parser)
    add_transductive_head_args(parser)

    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amodel", type=str, default="HTSAT-base")
    parser.add_argument("--enable-fusion", dest="enable_fusion", action="store_true")
    parser.add_argument("--disable-fusion", dest="enable_fusion", action="store_false")
    parser.set_defaults(enable_fusion=None)
    parser.add_argument("--clap-ckpt", type=str, default=None)
    parser.add_argument("--clap-model-id", type=int, default=1)
    parser.add_argument("--no-clap-auto-fallback", action="store_true")

    parser.add_argument("--cache-path", type=str, default="cache/clap_test_embeddings.npz",
                        help="Shared with clap_episode_eval.py (same backend='clap')")
    parser.add_argument("--recompute-cache", action="store_true")

    parser.add_argument("--output-csv", type=str, default="results/clap_hela_vfa_iid_ood_test_eval.csv")
    parser.add_argument("--save-episodes-json", type=str, default=None)

    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny self-test of hellinger_dist + hela_vfa scoring and exit")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# HELA-VFA core: Hellinger distance + prototype scoring
# Faithful port of HELA-VFA/Hellinger_dist.py and HELA-VFA/HELA_VFA_main.py.
# -----------------------------------------------------------------------------

def hellinger_dist(x, y):
    """Variational Hellinger distance.

    x: (n, d) tensor, y: (m, d) tensor -> (n, m) distances.
    Mirrors HELA-VFA/Hellinger_dist.py:5-26 (uses scalar mean/std reparam trick).
    """
    import torch

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x1 = x.unsqueeze(1).expand(n, m, d)
    y1 = y.unsqueeze(0).expand(n, m, d)

    sqrt2 = math.sqrt(2.0)
    x_m = torch.mean(x1)
    y_m = torch.mean(y1)
    x_std = torch.std(x1)
    y_std = torch.std(y1)

    P1 = x_m + x_std * (1.0 / torch.sqrt(torch.abs(2 * math.pi * x_std * x_std))) \
        * torch.exp(-((x1 - x_m) * (x1 - x_m)) / (2 * x_std * x_std))
    Q1 = y_m + y_std * (1.0 / torch.sqrt(torch.abs(2 * math.pi * y_std * y_std))) \
        * torch.exp(-((y1 - y_m) * (y1 - y_m)) / (2 * y_std * y_std))
    return torch.pow(torch.sqrt(torch.abs(P1)) - torch.sqrt(torch.abs(Q1)), 2).sum(2) / sqrt2


def hela_vfa_scores(z_support, support_y, z_query, n_way: int):
    """Returns (n_query, n_way) scores = -Hellinger(query, prototype)."""
    import torch

    protos = torch.stack(
        [z_support[support_y == c].mean(dim=0) for c in range(n_way)]
    )
    return -hellinger_dist(z_query, protos)


def hela_vfa_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
) -> float:
    import torch

    z_s = torch.from_numpy(features[support_idx]).float()
    z_q = torch.from_numpy(features[query_idx]).float()
    s_y = torch.from_numpy(support_y).long()
    scores = hela_vfa_scores(z_s, s_y, z_q, n_way)
    preds = scores.argmax(dim=1).numpy()
    return float((preds == query_y).mean())


# -----------------------------------------------------------------------------
# Hesim auxiliary loss (faithful port of HELA-VFA/Hesim/HesimLoss.py).
# -----------------------------------------------------------------------------

def hesim_loss(scores, labels, temperature: float = 0.01, eps: float = 1e-12):
    """Vectorized port of HesimLoss._compute_loss with HellingerSimilarity.

    Treats ``scores`` (B, n_way) as embeddings, L2-normalizes, computes pairwise
    similarity via matmul. For each (anchor, positive) pair (same label, off-
    diagonal), computes:
        -log( exp(sqrt|pos|/T - sqrt|max|/T) /
              (sum_neg exp(sqrt|neg|/T - sqrt|max|/T) + same numerator) )
    where max is detached over (pos, all negs) for stability. Returns mean over
    pos pairs, or zero if no pos/neg pairs exist (matches HesimLoss zero path).
    """
    import torch
    import torch.nn.functional as F

    B = scores.size(0)
    if B < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype, requires_grad=True)

    embs = F.normalize(scores, p=2, dim=1)
    sim_mat = torch.mm(embs, embs.t()) / temperature  # (B, B)

    eye = torch.eye(B, dtype=torch.bool, device=scores.device)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_eq & ~eye
    neg_mask = ~label_eq

    a1, p_idx = pos_mask.nonzero(as_tuple=True)
    if a1.numel() == 0 or not neg_mask.any():
        return torch.zeros((), device=scores.device, dtype=scores.dtype, requires_grad=True)

    pos_pairs = sim_mat[a1, p_idx]  # (n_pos,)

    neg_inf_val = torch.finfo(sim_mat.dtype).min
    neg_sims_full = sim_mat.masked_fill(~neg_mask, neg_inf_val)  # (B, B)
    neg_for_anchor = neg_sims_full[a1]  # (n_pos, B)

    neg_max = neg_for_anchor.max(dim=1, keepdim=True)[0]  # (n_pos, 1)
    max_val = torch.max(pos_pairs.unsqueeze(1), neg_max).detach()  # (n_pos, 1)

    sqrt_max = torch.sqrt(torch.abs(max_val.squeeze(1)))
    num = torch.exp(torch.sqrt(torch.abs(pos_pairs)) - sqrt_max)

    valid_neg = neg_for_anchor != neg_inf_val
    neg_terms = torch.zeros_like(neg_for_anchor)
    safe_neg = torch.where(valid_neg, neg_for_anchor, torch.zeros_like(neg_for_anchor))
    neg_terms = torch.exp(torch.sqrt(torch.abs(safe_neg)) - sqrt_max.unsqueeze(1))
    neg_terms = torch.where(valid_neg, neg_terms, torch.zeros_like(neg_terms))
    den = neg_terms.sum(dim=1) + num

    return -torch.log(num / (den + eps) + eps).mean()


# -----------------------------------------------------------------------------
# Self-test path
# -----------------------------------------------------------------------------

def _self_test() -> None:
    import torch

    torch.manual_seed(0)
    # Sanity: prototype-based score gives correct class for clean separable data.
    n_way, k, q, d = 5, 4, 6, 16
    centers = torch.randn(n_way, d) * 3.0
    z_s = torch.cat([centers[c].unsqueeze(0).expand(k, d) + 0.01 * torch.randn(k, d) for c in range(n_way)])
    s_y = torch.cat([torch.full((k,), c, dtype=torch.long) for c in range(n_way)])
    z_q = torch.cat([centers[c].unsqueeze(0).expand(q, d) + 0.01 * torch.randn(q, d) for c in range(n_way)])
    q_y = torch.cat([torch.full((q,), c, dtype=torch.long) for c in range(n_way)])

    scores = hela_vfa_scores(z_s, s_y, z_q, n_way)
    preds = scores.argmax(dim=1)
    acc = (preds == q_y).float().mean().item()
    print(f"[self-test] hela_vfa_scores accuracy on synthetic clusters: {acc * 100:.2f}%")
    assert acc > 0.9, "HELA-VFA failed sanity test on clean separable data"

    # Hesim loss runs without NaN.
    rand_scores = torch.randn(20, n_way, requires_grad=True)
    rand_labels = torch.randint(0, n_way, (20,))
    loss = hesim_loss(rand_scores, rand_labels, temperature=0.01)
    print(f"[self-test] hesim_loss on random scores: {float(loss):.6f}")
    assert torch.isfinite(loss), "Hesim loss produced non-finite value"
    loss.backward()
    assert rand_scores.grad is not None, "Hesim loss did not produce gradients"
    print("[self-test] OK")


# -----------------------------------------------------------------------------
# Episode dispatch with HELA-VFA support
# -----------------------------------------------------------------------------

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
    if algorithm == "hela_vfa":
        return hela_vfa_accuracy(features, support_idx, support_y, query_idx, query_y, n_way)
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
        return ce.proto_accuracy(bpa_features, s_idx, support_y, q_idx, query_y, n_way)
    return ce.episode_accuracy(
        algorithm, features, support_idx, support_y, query_idx, query_y, n_way, device, args,
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


# -----------------------------------------------------------------------------
# Main eval flow (mirror of ce.run_eval, with HELA-VFA-aware dispatcher)
# -----------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    if not args.data_root or not args.split_file:
        raise ValueError("--data-root and --split-file are required for evaluation")
    ce.set_seed(args.episode_seed)
    runtime_device = ce.resolve_runtime_device(args.device)

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    sorted_root = Path(args.sorted_root) if args.sorted_root else ce.infer_sorted_root(
        data_root, args.spec_root_name, args.sorted_root_name,
    )

    test_classes = ce.read_test_classes(split_file)
    samples = ce.build_samples(data_root=data_root, sorted_root=sorted_root, test_classes=test_classes)
    class_to_indices, class_bg_groups = ce.build_index(samples)

    classes_available = [c for c in test_classes if c in class_to_indices and len(class_to_indices[c]) > 0]
    if len(classes_available) < args.n_way:
        raise ValueError(f"Only {len(classes_available)} non-empty test classes found; n_way={args.n_way}")

    cache_path = Path(args.cache_path)
    features = ce.load_or_build_embeddings(
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

            for _ in tqdm(range(args.episodes), desc=f"{scenario.upper()} | {algorithm}", leave=True):
                try:
                    if scenario == "iid":
                        s_idx, s_y, q_idx, q_y, chosen_classes = ce.sample_iid_episode(
                            classes=classes_available,
                            class_to_indices=class_to_indices,
                            n_way=args.n_way,
                            k_shot=args.k_shot,
                            q_query=args.q_query,
                        )
                    else:
                        s_idx, s_y, q_idx, q_y, chosen_classes = ce.sample_ood_episode(
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

            mean, ci = ce.compute_ci95(accs)
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

    print("=== CLAP HELA-VFA IID/OOD Episodic Results ===")
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
        _self_test()
        return
    run_eval(args)


if __name__ == "__main__":
    main()
