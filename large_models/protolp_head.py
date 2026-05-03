"""ProtoLP transductive prototype-refinement head.

This is a compact, parameterized port of the core logic from the local
``protoLP`` scripts. It operates on one few-shot episode at a time and does not
train model weights; it refines class prototypes at inference time using the
unlabeled query batch.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np


def add_protolp_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ProtoLP")
    group.add_argument(
        "--protolp-epochs",
        type=int,
        default=50,
        help="ProtoLP prototype-refinement iterations per episode",
    )
    group.add_argument(
        "--protolp-alpha",
        type=float,
        default=0.2,
        help="Prototype update step size",
    )
    group.add_argument(
        "--protolp-lambda",
        dest="protolp_lambda",
        type=float,
        default=10.0,
        help="Distance sharpness used by ProtoLP optimal transport",
    )
    group.add_argument(
        "--protolp-gamma",
        type=float,
        default=1.0,
        help="Label-propagation regularization weight",
    )
    group.add_argument(
        "--protolp-beta",
        type=float,
        default=0.6,
        help="Blend weight for propagated labels versus direct prototype assignments",
    )
    group.add_argument(
        "--protolp-svd-dim",
        type=int,
        default=40,
        help="Episode-local SVD projection dimension; set 0 to disable",
    )
    group.add_argument(
        "--protolp-power",
        type=float,
        default=0.0,
        help="Optional signed power transform before normalization; 0 disables it",
    )
    group.add_argument(
        "--protolp-no-center",
        dest="protolp_center",
        action="store_false",
        help="Disable episode mean subtraction after SVD",
    )
    group.set_defaults(protolp_center=True)


def protolp_accuracy_from_args(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    args: argparse.Namespace,
    device: Optional[str] = None,
) -> float:
    return protolp_accuracy(
        features=features,
        support_idx=support_idx,
        support_y=support_y,
        query_idx=query_idx,
        query_y=query_y,
        n_way=n_way,
        epochs=getattr(args, "protolp_epochs", 50),
        alpha=getattr(args, "protolp_alpha", 0.2),
        ot_lambda=getattr(args, "protolp_lambda", 10.0),
        lp_gamma=getattr(args, "protolp_gamma", 1.0),
        lp_beta=getattr(args, "protolp_beta", 0.6),
        svd_dim=getattr(args, "protolp_svd_dim", 40),
        power=getattr(args, "protolp_power", 0.0),
        center=getattr(args, "protolp_center", True),
        device=device,
    )


def protolp_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    *,
    epochs: int = 50,
    alpha: float = 0.2,
    ot_lambda: float = 10.0,
    lp_gamma: float = 1.0,
    lp_beta: float = 0.6,
    svd_dim: int = 40,
    power: float = 0.0,
    center: bool = True,
    device: Optional[str] = None,
) -> float:
    import torch

    torch_device = _resolve_torch_device(device)
    support_y_t = torch.as_tensor(support_y, dtype=torch.long, device=torch_device)
    query_y_np = np.asarray(query_y, dtype=np.int64)

    support = torch.as_tensor(features[support_idx], dtype=torch.float64, device=torch_device)
    query = torch.as_tensor(features[query_idx], dtype=torch.float64, device=torch_device)
    support, query = _preprocess_episode(
        support=support,
        query=query,
        svd_dim=svd_dim,
        power=power,
        center=center,
    )

    x = torch.cat([support, query], dim=0)
    n_support = support.size(0)
    query_target = float(query.size(0)) / float(n_way)
    support_counts = torch.stack([(support_y_t == c).sum() for c in range(n_way)]).to(x.dtype)
    col_targets = support_counts + query_target

    mus = _init_prototypes(support, support_y_t, n_way)
    for _ in range(max(int(epochs), 0)):
        direct_probs = _prototype_assignments(
            x=x,
            mus=mus,
            support_y=support_y_t,
            n_support=n_support,
            n_way=n_way,
            query_target=query_target,
            ot_lambda=ot_lambda,
        )
        propagated = _label_propagate(
            probs=direct_probs,
            support_y=support_y_t,
            n_support=n_support,
            col_targets=col_targets,
            gamma=lp_gamma,
        )
        mask = (lp_beta * propagated + (1.0 - lp_beta) * direct_probs).clamp(0.0, 1.0)
        mus_est = mask.t().matmul(x) / mask.sum(dim=0).clamp_min(1e-12).unsqueeze(1)
        mus = mus + alpha * (mus_est - mus)

    final_probs = _prototype_assignments(
        x=x,
        mus=mus,
        support_y=support_y_t,
        n_support=n_support,
        n_way=n_way,
        query_target=query_target,
        ot_lambda=ot_lambda,
    )
    preds = final_probs[n_support:].argmax(dim=1).detach().cpu().numpy()
    return float((preds == query_y_np).mean())


def _resolve_torch_device(device: Optional[str]):
    import torch

    if not device or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _preprocess_episode(support, query, svd_dim: int, power: float, center: bool):
    import torch

    n_support = support.size(0)
    x = torch.cat([support, query], dim=0)
    if power and power > 0:
        x = torch.sign(x) * torch.abs(x).clamp_min(1e-12).pow(power)

    x = _normalize_rows(x)
    if svd_dim and svd_dim > 0:
        k = min(int(svd_dim), x.size(0), x.size(1))
        if 0 < k < x.size(1):
            _, _, vh = torch.linalg.svd(x, full_matrices=False)
            x = x.matmul(vh[:k].t())

    if center:
        x = x - x.mean(dim=0, keepdim=True)
        x = _normalize_rows(x)

    return x[:n_support], x[n_support:]


def _normalize_rows(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _init_prototypes(support, support_y, n_way: int):
    import torch

    mus = []
    for class_id in range(n_way):
        cls_feat = support[support_y == class_id]
        if cls_feat.numel() == 0:
            raise ValueError(f"ProtoLP received no support examples for class {class_id}")
        mus.append(cls_feat.mean(dim=0))
    return _normalize_rows(torch.stack(mus, dim=0))


def _prototype_assignments(
    x,
    mus,
    support_y,
    n_support: int,
    n_way: int,
    query_target: float,
    ot_lambda: float,
):
    import torch
    import torch.nn.functional as F

    query = x[n_support:]
    dist = (query.unsqueeze(1) - mus.unsqueeze(0)).norm(dim=2).pow(2)
    query_probs = _sinkhorn_from_cost(
        cost=dist,
        col_target=query_target,
        ot_lambda=ot_lambda,
    )

    probs = torch.zeros(x.size(0), n_way, dtype=x.dtype, device=x.device)
    probs[:n_support] = F.one_hot(support_y, num_classes=n_way).to(dtype=x.dtype)
    probs[n_support:] = query_probs
    return probs


def _sinkhorn_from_cost(cost, col_target: float, ot_lambda: float, max_iter: int = 100, tol: float = 1e-4):
    import torch

    if cost.size(0) == 0:
        return torch.empty_like(cost)

    logits = -float(ot_lambda) * cost
    logits = logits - logits.max()
    p = torch.exp(logits).clamp_min(1e-300)
    p = p / p.sum().clamp_min(1e-300)

    row_targets = torch.ones(cost.size(0), dtype=cost.dtype, device=cost.device)
    col_targets = torch.full((cost.size(1),), float(col_target), dtype=cost.dtype, device=cost.device)
    return _balance_matrix(p, row_targets, col_targets, max_iter=max_iter, tol=tol)


def _label_propagate(probs, support_y, n_support: int, col_targets, gamma: float):
    import torch
    import torch.nn.functional as F

    n_way = probs.size(1)
    y = F.one_hot(support_y, num_classes=n_way).to(dtype=probs.dtype)
    delta = probs.sum(dim=0).clamp_min(1e-12)
    w = probs.t().matmul(probs / delta.unsqueeze(0))
    laplacian = torch.eye(n_way, dtype=probs.dtype, device=probs.device) - w
    z_l = probs[:n_support]
    lhs = z_l.t().matmul(z_l) + float(gamma) * laplacian
    rhs = z_l.t().matmul(y)
    a = _solve_stable(lhs, rhs)
    propagated = probs.matmul(a).clamp_min(1e-12)

    row_targets = torch.ones(propagated.size(0), dtype=probs.dtype, device=probs.device)
    fixed = F.one_hot(support_y, num_classes=n_way).to(dtype=probs.dtype)
    return _balance_matrix(
        propagated,
        row_targets=row_targets,
        col_targets=col_targets,
        fixed_prefix=fixed,
        max_iter=100,
        tol=1e-3,
    )


def _solve_stable(lhs, rhs):
    import torch

    eye = torch.eye(lhs.size(0), dtype=lhs.dtype, device=lhs.device)
    for jitter in (1e-9, 1e-7, 1e-5):
        try:
            return torch.linalg.solve(lhs + jitter * eye, rhs)
        except RuntimeError:
            continue
    return torch.linalg.pinv(lhs).matmul(rhs)


def _balance_matrix(p, row_targets, col_targets, fixed_prefix=None, max_iter: int = 100, tol: float = 1e-4):
    import torch

    p = p.clamp_min(1e-300)
    if fixed_prefix is not None:
        n_fixed = fixed_prefix.size(0)
        p[:n_fixed] = fixed_prefix
    else:
        n_fixed = 0

    for _ in range(max_iter):
        p = p * (row_targets / p.sum(dim=1).clamp_min(1e-300)).unsqueeze(1)
        p = p * (col_targets / p.sum(dim=0).clamp_min(1e-300)).unsqueeze(0)
        if fixed_prefix is not None:
            p[:n_fixed] = fixed_prefix

        row_err = torch.abs(p.sum(dim=1) - row_targets).max()
        col_err = torch.abs(p.sum(dim=0) - col_targets).max()
        if float(torch.maximum(row_err, col_err)) <= tol:
            break
    return p


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n_way, k_shot, q_query, dim = 5, 2, 4, 32
    centers = rng.normal(size=(n_way, dim)).astype(np.float32) * 4.0
    support = []
    query = []
    support_y = []
    query_y = []
    for c in range(n_way):
        support.append(centers[c] + 0.02 * rng.normal(size=(k_shot, dim)).astype(np.float32))
        query.append(centers[c] + 0.02 * rng.normal(size=(q_query, dim)).astype(np.float32))
        support_y.extend([c] * k_shot)
        query_y.extend([c] * q_query)
    features = np.concatenate([np.concatenate(support), np.concatenate(query)], axis=0)
    support_idx = np.arange(n_way * k_shot)
    query_idx = np.arange(n_way * k_shot, features.shape[0])
    acc = protolp_accuracy(
        features,
        support_idx,
        np.asarray(support_y),
        query_idx,
        np.asarray(query_y),
        n_way,
        epochs=3,
        svd_dim=0,
        device="cpu",
    )
    print(f"[self-test] ProtoLP synthetic cluster accuracy: {acc * 100:.2f}%")
    assert acc > 0.95, "ProtoLP failed synthetic cluster sanity test"


if __name__ == "__main__":
    _self_test()
