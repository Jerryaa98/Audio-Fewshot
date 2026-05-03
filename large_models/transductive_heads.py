"""Text-free transductive few-shot inference heads.

Compact ports of the support/query-only few-shot heads used in the local
``transductive-CLIP`` checkout:

- Alpha-TIM
- LaplacianShot
- BD-CSPN
- PADDLE
- ECPE

These heads operate on one episode of already-extracted embeddings. They do not
use CLIP/CLAP text features, class names, or zero-shot logits.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np


TRANSDUCTIVE_ALGORITHMS = ("alpha_tim", "laplacian_shot", "bdcspn", "paddle", "ecpe")


def add_transductive_head_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Text-free transductive heads")
    group.add_argument(
        "--transductive-no-normalize",
        dest="transductive_normalize",
        action="store_false",
        help="Disable L2 normalization for Alpha-TIM and PADDLE inputs",
    )
    group.set_defaults(transductive_normalize=True)

    group.add_argument("--alpha-tim-iters", type=int, default=1000)
    group.add_argument("--alpha-tim-lr", type=float, default=1e-4)
    group.add_argument("--alpha-tim-temp", type=float, default=15.0)
    group.add_argument("--alpha-tim-alpha", type=float, default=7.0)
    group.add_argument("--alpha-tim-ce-weight", type=float, default=1.0)
    group.add_argument("--alpha-tim-marginal-weight", type=float, default=1.0)
    group.add_argument("--alpha-tim-conditional-weight", type=float, default=1.0)

    group.add_argument("--laplacian-shot-iters", type=int, default=20)
    group.add_argument("--laplacian-shot-knn", type=int, default=3)
    group.add_argument("--laplacian-shot-lambda", dest="laplacian_shot_lambda", type=float, default=0.7)
    group.add_argument(
        "--laplacian-shot-norm-type",
        choices=["L2N", "CL2N", "UN"],
        default="L2N",
    )

    group.add_argument("--bdcspn-temp", type=float, default=30.0)
    group.add_argument("--bdcspn-norm-type", choices=["L2N", "CL2N", "UN"], default="L2N")

    group.add_argument("--paddle-iters", type=int, default=20)
    group.add_argument("--paddle-lambda", dest="paddle_lambda", type=float, default=0.0)
    group.add_argument("--paddle-temp", type=float, default=1.0)

    group.add_argument("--ecpe-epochs", type=int, default=10)
    group.add_argument("--ecpe-lambda", dest="ecpe_lambda", type=float, default=2.0)
    group.add_argument("--ecpe-alpha", type=float, default=0.7)
    group.add_argument("--ecpe-update-rate", type=float, default=0.6)
    group.add_argument("--ecpe-svd-dim", type=int, default=40)
    group.add_argument("--ecpe-power", type=float, default=0.0)
    group.add_argument("--ecpe-no-center", dest="ecpe_center", action="store_false")
    group.add_argument("--ecpe-unbalanced", dest="ecpe_balance", action="store_false")
    group.set_defaults(ecpe_center=True, ecpe_balance=True)


def extend_algorithm_choices(base_choices):
    """Return ``base_choices`` plus the shared text-free transductive heads."""
    return list(base_choices) + [a for a in TRANSDUCTIVE_ALGORITHMS if a not in base_choices]


def is_transductive_algorithm(algorithm: str) -> bool:
    return algorithm in TRANSDUCTIVE_ALGORITHMS


def transductive_accuracy_from_args(
    algorithm: str,
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    args: argparse.Namespace,
    device: Optional[str] = None,
) -> float:
    if algorithm == "alpha_tim":
        return alpha_tim_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device=device,
            iters=getattr(args, "alpha_tim_iters", 1000),
            lr=getattr(args, "alpha_tim_lr", 1e-4),
            temperature=getattr(args, "alpha_tim_temp", 15.0),
            alpha=getattr(args, "alpha_tim_alpha", 7.0),
            ce_weight=getattr(args, "alpha_tim_ce_weight", 1.0),
            marginal_weight=getattr(args, "alpha_tim_marginal_weight", 1.0),
            conditional_weight=getattr(args, "alpha_tim_conditional_weight", 1.0),
            normalize=getattr(args, "transductive_normalize", True),
        )
    if algorithm == "laplacian_shot":
        return laplacian_shot_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device=device,
            iters=getattr(args, "laplacian_shot_iters", 20),
            knn=getattr(args, "laplacian_shot_knn", 3),
            lmd=getattr(args, "laplacian_shot_lambda", 0.7),
            norm_type=getattr(args, "laplacian_shot_norm_type", "L2N"),
        )
    if algorithm == "bdcspn":
        return bdcspn_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device=device,
            temperature=getattr(args, "bdcspn_temp", 30.0),
            norm_type=getattr(args, "bdcspn_norm_type", "L2N"),
        )
    if algorithm == "paddle":
        return paddle_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device=device,
            iters=getattr(args, "paddle_iters", 20),
            lmd=getattr(args, "paddle_lambda", 0.0),
            temperature=getattr(args, "paddle_temp", 1.0),
            normalize=getattr(args, "transductive_normalize", True),
        )
    if algorithm == "ecpe":
        return ecpe_accuracy(
            features,
            support_idx,
            support_y,
            query_idx,
            query_y,
            n_way,
            device=device,
            epochs=getattr(args, "ecpe_epochs", 10),
            lam=getattr(args, "ecpe_lambda", 2.0),
            alpha=getattr(args, "ecpe_alpha", 0.7),
            update_rate=getattr(args, "ecpe_update_rate", 0.6),
            svd_dim=getattr(args, "ecpe_svd_dim", 40),
            power=getattr(args, "ecpe_power", 0.0),
            center=getattr(args, "ecpe_center", True),
            balance=getattr(args, "ecpe_balance", True),
        )
    raise ValueError(f"Unsupported transductive algorithm: {algorithm}")


def alpha_tim_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    *,
    device: Optional[str] = None,
    iters: int = 1000,
    lr: float = 1e-4,
    temperature: float = 15.0,
    alpha: float = 7.0,
    ce_weight: float = 1.0,
    marginal_weight: float = 1.0,
    conditional_weight: float = 1.0,
    normalize: bool = True,
) -> float:
    import torch
    import torch.nn.functional as F

    torch_device = _resolve_torch_device(device)
    support, support_lbl, query, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, torch_device
    )
    if normalize:
        support = F.normalize(support, p=2, dim=-1)
        query = F.normalize(query, p=2, dim=-1)

    weights = _class_means(support, support_lbl, n_way).detach().clone()
    weights.requires_grad_()
    optimizer = torch.optim.Adam([weights], lr=float(lr))
    y_s_one_hot = F.one_hot(support_lbl, num_classes=n_way).to(dtype=support.dtype)

    for _ in range(max(int(iters), 0)):
        logits_s = _euclidean_logits(support, weights, temperature)
        logits_q = _euclidean_logits(query, weights, temperature)
        p_s = logits_s.softmax(dim=-1).clamp_min(1e-12)
        p_q = logits_q.softmax(dim=-1).clamp_min(1e-12)

        ce = _alpha_cross_entropy(y_s_one_hot, p_s, alpha)
        marginal = _alpha_entropy(p_q.mean(dim=0), alpha)
        conditional = torch.stack([_alpha_entropy(row, alpha) for row in p_q], dim=0).mean()
        loss = float(ce_weight) * ce - (float(marginal_weight) * marginal - float(conditional_weight) * conditional)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = _euclidean_logits(query, weights, temperature).argmax(dim=-1)
        return float((preds == query_lbl).float().mean().item())


def laplacian_shot_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    *,
    device: Optional[str] = None,
    iters: int = 20,
    knn: int = 3,
    lmd: float = 0.7,
    norm_type: str = "L2N",
) -> float:
    import torch

    torch_device = _resolve_torch_device(device)
    support, support_lbl, query, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, torch_device
    )
    support, query = _normalize_episode(support, query, norm_type)
    protos = _class_means(support, support_lbl, n_way)

    unary = torch.cdist(query, protos).pow(2)
    affinity = _query_affinity(query, knn=max(int(knn), 1))
    y = torch.softmax(-unary, dim=-1)
    old_energy = None
    for _ in range(max(int(iters), 0)):
        y = torch.softmax(-unary + float(lmd) * affinity.matmul(y), dim=-1)
        energy = _laplacian_energy(y, unary, affinity, float(lmd))
        if old_energy is not None and abs(float(energy - old_energy)) <= 1e-6 * max(abs(float(old_energy)), 1e-12):
            break
        old_energy = energy

    preds = y.argmax(dim=-1)
    return float((preds == query_lbl).float().mean().item())


def bdcspn_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    *,
    device: Optional[str] = None,
    temperature: float = 30.0,
    norm_type: str = "L2N",
) -> float:
    import torch
    import torch.nn.functional as F

    torch_device = _resolve_torch_device(device)
    support, support_lbl, query, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, torch_device
    )
    support, query = _normalize_episode(support, query, norm_type)
    init_protos = _class_means(support, support_lbl, n_way)

    eta = support.mean(dim=0, keepdim=True) - query.mean(dim=0, keepdim=True)
    query_shifted = query + eta
    query_aug = torch.cat([support, query_shifted], dim=0)

    logits_aug = _negative_half_sqdist(F.normalize(query_aug, p=2, dim=-1), F.normalize(init_protos, p=2, dim=-1))
    assignments = (float(temperature) * logits_aug).softmax(dim=-1)
    query_aug = F.normalize(query_aug, p=2, dim=-1)
    protos = assignments.t().matmul(query_aug) / assignments.sum(dim=0).clamp_min(1e-12).unsqueeze(1)

    logits_q = _negative_half_sqdist(F.normalize(query, p=2, dim=-1), F.normalize(protos, p=2, dim=-1))
    preds = (float(temperature) * logits_q).softmax(dim=-1).argmax(dim=-1)
    return float((preds == query_lbl).float().mean().item())


def paddle_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    *,
    device: Optional[str] = None,
    iters: int = 20,
    lmd: float = 0.0,
    temperature: float = 1.0,
    normalize: bool = True,
) -> float:
    import torch
    import torch.nn.functional as F

    torch_device = _resolve_torch_device(device)
    support, support_lbl, query, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, torch_device
    )
    if normalize:
        support = F.normalize(support, p=2, dim=-1)
        query = F.normalize(query, p=2, dim=-1)

    y_s_one_hot = F.one_hot(support_lbl, num_classes=n_way).to(dtype=support.dtype)
    weights = _class_means(support, support_lbl, n_way)
    v = torch.zeros(n_way, dtype=support.dtype, device=torch_device)
    logits = _negative_half_sqdist(query, weights)
    u = (float(temperature) * logits).softmax(dim=-1)

    n_query = max(query.size(0), 1)
    for _ in range(max(int(iters), 0)):
        logits = _negative_half_sqdist(query, weights)
        u = (float(temperature) * logits + float(lmd) * v.unsqueeze(0) / n_query).softmax(dim=-1)
        v = torch.log(u.sum(dim=0) / n_query + 1e-15) + 1.0

        num = u.t().matmul(query) + y_s_one_hot.t().matmul(support)
        den = u.sum(dim=0) + y_s_one_hot.sum(dim=0)
        weights = num / den.clamp_min(1e-12).unsqueeze(1)

    preds = u.argmax(dim=-1)
    return float((preds == query_lbl).float().mean().item())


def ecpe_accuracy(
    features: np.ndarray,
    support_idx: np.ndarray,
    support_y: np.ndarray,
    query_idx: np.ndarray,
    query_y: np.ndarray,
    n_way: int,
    *,
    device: Optional[str] = None,
    epochs: int = 10,
    lam: float = 2.0,
    alpha: float = 0.7,
    update_rate: float = 0.6,
    svd_dim: int = 40,
    power: float = 0.0,
    center: bool = True,
    balance: bool = True,
) -> float:
    import torch

    torch_device = _resolve_torch_device(device)
    support, support_lbl, query, query_lbl = _episode_tensors(
        features, support_idx, support_y, query_idx, query_y, torch_device
    )
    support, query = _ecpe_preprocess_episode(
        support=support,
        query=query,
        svd_dim=svd_dim,
        power=power,
        center=center,
    )

    base_features = torch.cat([support, query], dim=0)
    n_support = support.size(0)
    n_query = query.size(0)
    protos = _class_means(support, support_lbl, n_way)

    z = _ecpe_initial_probs(
        features=base_features,
        protos=protos,
        support_lbl=support_lbl,
        n_support=n_support,
        n_way=n_way,
        lam=lam,
        balance=balance,
    )

    for _ in range(max(int(epochs), 0)):
        z = _ecpe_initial_probs(
            features=base_features,
            protos=protos,
            support_lbl=support_lbl,
            n_support=n_support,
            n_way=n_way,
            lam=lam,
            balance=balance,
        )

        entropy = _ecpe_entropy(z)
        calibrated_z = z * (1.0 - entropy).unsqueeze(1)
        proto_est = calibrated_z.t().matmul(base_features) / calibrated_z.sum(dim=0).clamp_min(1e-12).unsqueeze(1)
        protos = (1.0 - float(update_rate)) * protos + float(update_rate) * proto_est

        proto_affinity = _ecpe_kernel(protos, protos, lam=lam)
        proto_entropy = _ecpe_entropy(proto_affinity)
        entropy_threshold = entropy.mean()
        proto_mask = (proto_entropy < entropy_threshold).to(dtype=base_features.dtype)
        proto_nodes = protos * proto_mask.unsqueeze(1)
        proto_node_probs = proto_affinity * proto_mask.unsqueeze(1)

        graph_features = torch.cat([base_features, proto_nodes], dim=0)
        graph_z = torch.cat([calibrated_z, proto_node_probs], dim=0)
        graph = _ecpe_graph(graph_features, lam=lam)
        eye = torch.eye(graph.size(0), dtype=graph.dtype, device=graph.device)
        z = torch.linalg.solve(eye - float(alpha) * graph, graph_z)
        z = _ecpe_optimal_transport(
            z,
            n_lsamples=n_support,
            support_lbl=support_lbl,
            epsilon=1e-3,
            class_balance=balance,
        )

    preds = z[: n_support + n_query][n_support:].argmax(dim=-1)
    return float((preds == query_lbl).float().mean().item())


def _resolve_torch_device(device: Optional[str]):
    import torch

    if not device or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _episode_tensors(features, support_idx, support_y, query_idx, query_y, device):
    import torch

    support = torch.as_tensor(features[support_idx], dtype=torch.float32, device=device)
    support_lbl = torch.as_tensor(support_y, dtype=torch.long, device=device)
    query = torch.as_tensor(features[query_idx], dtype=torch.float32, device=device)
    query_lbl = torch.as_tensor(query_y, dtype=torch.long, device=device)
    return support, support_lbl, query, query_lbl


def _normalize_episode(support, query, norm_type: str):
    import torch.nn.functional as F

    norm_type = str(norm_type).upper()
    if norm_type == "UN":
        return support, query
    if norm_type == "CL2N":
        center = support.mean(dim=0, keepdim=True)
        support = support - center
        query = query - center
    if norm_type in ("L2N", "CL2N"):
        support = F.normalize(support, p=2, dim=-1)
        query = F.normalize(query, p=2, dim=-1)
    return support, query


def _ecpe_preprocess_episode(support, query, svd_dim: int, power: float, center: bool):
    import torch

    n_support = support.size(0)
    x = torch.cat([support, query], dim=0)
    if power and power > 0:
        x = torch.sign(x) * x.abs().clamp_min(1e-12).pow(float(power))

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


def _class_means(support, support_lbl, n_way: int):
    import torch

    protos = []
    for class_id in range(n_way):
        cls = support[support_lbl == class_id]
        if cls.numel() == 0:
            raise ValueError(f"No support examples for class {class_id}")
        protos.append(cls.mean(dim=0))
    return torch.stack(protos, dim=0)


def _negative_half_sqdist(samples, weights):
    return -0.5 * (samples.unsqueeze(1) - weights.unsqueeze(0)).pow(2).sum(dim=-1)


def _euclidean_logits(samples, weights, temperature: float):
    return float(temperature) * _negative_half_sqdist(samples, weights)


def _ecpe_initial_probs(features, protos, support_lbl, n_support: int, n_way: int, lam: float, balance: bool):
    import torch
    import torch.nn.functional as F

    query = features[n_support:]
    query_dist = torch.cdist(query, protos).pow(2)
    query_scores = torch.exp(-float(lam) * query_dist).clamp_min(1e-12)
    query_probs = _ecpe_optimal_transport(
        query_scores,
        n_lsamples=0,
        support_lbl=support_lbl[:0],
        epsilon=1e-3,
        class_balance=balance,
    )

    probs = torch.zeros(features.size(0), n_way, dtype=features.dtype, device=features.device)
    probs[:n_support] = F.one_hot(support_lbl, num_classes=n_way).to(dtype=features.dtype)
    probs[n_support:] = query_probs
    return probs


def _ecpe_kernel(features1, features2, lam: float):
    import torch

    return torch.exp(-float(lam) * torch.cdist(features1, features2).pow(2)).clamp_min(1e-12)


def _ecpe_graph(features, lam: float):
    import torch

    samples = features.size(0)
    eye = torch.eye(samples, dtype=features.dtype, device=features.device)
    graph = _ecpe_kernel(features, features, lam=lam) * (1.0 - eye)
    degree = graph.sum(dim=-1).clamp_min(1e-12).pow(-0.5)
    return degree.unsqueeze(0) * graph * degree.unsqueeze(1)


def _ecpe_entropy(prob_matrix, normalize: bool = True):
    import torch

    probs = prob_matrix.clamp_min(1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    if normalize:
        max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=probs.dtype, device=probs.device))
        entropy = entropy / max_entropy.clamp_min(1e-12)
    return entropy


def _ecpe_optimal_transport(matrix, n_lsamples: int, support_lbl, epsilon: float, class_balance: bool):
    import torch
    import torch.nn.functional as F

    if matrix.numel() == 0:
        return matrix

    n_samples, n_way = matrix.shape
    probs = matrix.clamp_min(1e-12).clone()
    row_target = torch.ones(n_samples, dtype=probs.dtype, device=probs.device)
    col_target = torch.ones(n_way, dtype=probs.dtype, device=probs.device) * (float(n_samples) / float(n_way))

    n_lsamples = max(int(n_lsamples), 0)
    support_one_hot = None
    if n_lsamples > 0:
        support_one_hot = F.one_hot(support_lbl[:n_lsamples], num_classes=n_way).to(dtype=probs.dtype)

    for _ in range(1000):
        prev_row_sum = probs.sum(dim=1)
        probs = probs * (row_target / prev_row_sum.clamp_min(1e-12)).unsqueeze(1)
        if class_balance:
            probs = probs * (col_target / probs.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)
        if support_one_hot is not None:
            probs[:n_lsamples] = support_one_hot

        row_error = torch.max(torch.abs(probs.sum(dim=1) - row_target))
        if float(row_error) <= float(epsilon):
            break

    return probs


def _alpha_entropy(probs, alpha: float):
    import torch

    probs = probs.clamp_min(1e-12)
    if abs(float(alpha) - 1.0) < 1e-6:
        return -(probs * torch.log(probs)).sum()
    return (1.0 - probs.pow(float(alpha)).sum()) / (float(alpha) - 1.0)


def _alpha_cross_entropy(one_hot, probs, alpha: float):
    import torch

    probs = probs.clamp_min(1e-12)
    if abs(float(alpha) - 1.0) < 1e-6:
        return -(one_hot * torch.log(probs)).sum(dim=-1).mean()
    term = one_hot.pow(float(alpha)) * probs.pow(1.0 - float(alpha))
    return ((1.0 - term.sum(dim=-1)) / (float(alpha) - 1.0)).mean()


def _query_affinity(query, knn: int):
    import torch

    q = query.size(0)
    affinity = torch.zeros((q, q), dtype=query.dtype, device=query.device)
    if q <= 1:
        return affinity

    k = min(max(int(knn), 2), q)
    dist = torch.cdist(query, query)
    nn_idx = dist.topk(k=k, largest=False).indices[:, 1:]
    rows = torch.arange(q, device=query.device).unsqueeze(1).expand_as(nn_idx)
    affinity[rows.reshape(-1), nn_idx.reshape(-1)] = 1.0
    return affinity


def _laplacian_energy(y, unary, affinity, lmd: float):
    import torch

    pairwise = affinity.matmul(y)
    return (y * torch.log(y.clamp_min(1e-20)) + unary * y - float(lmd) * pairwise * y).sum()


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n_way, k_shot, q_query, dim = 5, 3, 6, 32
    centers = rng.normal(size=(n_way, dim)).astype(np.float32) * 4.0
    support, query, support_y, query_y = [], [], [], []
    for c in range(n_way):
        support.append(centers[c] + 0.05 * rng.normal(size=(k_shot, dim)).astype(np.float32))
        query.append(centers[c] + 0.05 * rng.normal(size=(q_query, dim)).astype(np.float32))
        support_y.extend([c] * k_shot)
        query_y.extend([c] * q_query)

    features = np.concatenate([np.concatenate(support), np.concatenate(query)], axis=0)
    support_idx = np.arange(n_way * k_shot)
    query_idx = np.arange(n_way * k_shot, features.shape[0])
    support_y_arr = np.asarray(support_y)
    query_y_arr = np.asarray(query_y)

    class Args:
        alpha_tim_iters = 3
        laplacian_shot_iters = 3
        paddle_iters = 3

    for algorithm in TRANSDUCTIVE_ALGORITHMS:
        acc = transductive_accuracy_from_args(
            algorithm,
            features,
            support_idx,
            support_y_arr,
            query_idx,
            query_y_arr,
            n_way,
            Args(),
            device="cpu",
        )
        print(f"[self-test] {algorithm}: {acc * 100:.2f}%")
        assert acc > 0.95, f"{algorithm} failed synthetic cluster self-test"


if __name__ == "__main__":
    _self_test()
