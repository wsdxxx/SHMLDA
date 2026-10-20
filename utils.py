
import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import os


def set_seed(seed=2024):
    """固定所有的随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def fix_similarity_matrix(mat, target_size):
    if not isinstance(mat, np.ndarray):
        mat = mat.to_numpy()
    mat = np.asarray(mat)

    if mat.ndim != 2:
        raise ValueError("Similarity matrix must be two-dimensional.")
    if mat.shape[0] > target_size or mat.shape[1] > target_size:
        raise ValueError(
            f"Similarity matrix shape {mat.shape} exceeds target size "
            f"{target_size}."
        )
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    if mat.shape == (target_size, target_size):
        return mat

    fixed = np.zeros((target_size, target_size), dtype=mat.dtype)
    fixed[: mat.shape[0], : mat.shape[1]] = mat
    return fixed


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    sensitivity = TP / max(TP + FN, 1)
    precision = TP / max(TP + FP, 1)
    specificity = TN / max(TN + FP, 1)

    mcc_denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if mcc_denominator > 0:
        mcc = (TP * TN - FP * FN) / np.sqrt(mcc_denominator)
    else:
        mcc = 0.0

    if precision + sensitivity > 0:
        F1_score = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        F1_score = 0.0

    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def select_decision_threshold(y_true, y_score, metric="mcc"):
    """Choose a decision threshold using validation labels only.

    The threshold is selected over the unique validation scores, which avoids
    fixing the classification cutoff to the arbitrary probability value 0.5.
    Ties are resolved in favor of the threshold closest to 0.5.
    """
    metric = metric.lower()
    if metric not in {"mcc", "f1", "youden"}:
        raise ValueError("metric must be one of: mcc, f1, youden.")

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_true.size == 0:
        raise ValueError("Cannot select a threshold from empty predictions.")
    if np.unique(y_true).size < 2:
        return 0.5

    candidates = np.unique(y_score)
    if candidates.size == 0:
        return 0.5

    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(np.int64)
        metrics = calculate_metrics(y_true, y_pred)
        if metric == "mcc":
            score = metrics[5]
        elif metric == "f1":
            score = metrics[4]
        else:
            score = metrics[1] + metrics[3] - 1.0

        closer_to_default = abs(threshold - 0.5) < abs(
            best_threshold - 0.5
        )
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and closer_to_default
        ):
            best_score = score
            best_threshold = float(threshold)

    return float(np.clip(best_threshold, 0.0, 1.0))


def concat_hypergraphs(H_list):
    if not H_list:
        raise ValueError("H_list must contain at least one hypergraph.")
    if any(H.dim() != 2 for H in H_list):
        raise ValueError("Every hypergraph incidence matrix must be 2D.")
    num_nodes = H_list[0].size(0)
    if any(H.size(0) != num_nodes for H in H_list):
        raise ValueError("All hypergraphs must have the same number of nodes.")
    return torch.cat(H_list, dim=1)


def make_undirected_edge_index(edge_index):
    """Return a unique undirected version of a directed edge index."""
    if edge_index is None or edge_index.numel() == 0:
        if edge_index is None:
            return torch.empty((2, 0), dtype=torch.long)
        return edge_index

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    return torch.unique(
        torch.cat((edge_index, edge_index.flip(0)), dim=1),
        dim=1,
    )


def hypergraph_star_expansion(H, node_offset=0):
    """Expand a hypergraph with explicit hypernodes.

    This helper keeps hypernodes in the returned edge list. Use
    ``hypergraph_clique_expansion`` when the downstream model only accepts
    real node indices.
    """
    num_nodes, num_edges = H.shape
    edges = []
    for e in range(num_edges):
        hyper_node = node_offset + num_nodes + e
        nodes = torch.where(H[:, e] > 0)[0]
        for v in nodes:
            v = v.item() + node_offset
            edges.append([v, hyper_node])
            edges.append([hyper_node, v])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def hypergraph_clique_expansion(H, node_offset=0):
    """Convert hyperedges to an undirected node-level adjacency list."""
    if H.dim() != 2:
        raise ValueError("H must have shape [num_nodes, num_hyperedges].")
    if node_offset < 0:
        raise ValueError("node_offset must be non-negative.")

    num_nodes, num_edges = H.shape
    src = []
    dst = []

    for e in range(num_edges):
        nodes = torch.where(H[:, e] > 0)[0]
        if nodes.numel() < 2:
            continue

        for i in range(nodes.numel()):
            u = int(nodes[i].item()) + node_offset
            for j in range(i + 1, nodes.numel()):
                v = int(nodes[j].item()) + node_offset
                src.extend((u, v))
                dst.extend((v, u))

    if not src:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def sample_bipartite_negatives(
    num_lnc,
    num_dis,
    num_samples,
    exclude_edge_index=None,
    generator=None,
    disease_offset=0,
):
    """Sample unique lnc-disease pairs while excluding all known pairs.

    Returned disease indices use the global graph index space, i.e. they are
    shifted by ``disease_offset``.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if num_samples == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if num_lnc <= 0 or num_dis <= 0:
        raise ValueError("num_lnc and num_dis must be positive.")
    if disease_offset < 0:
        raise ValueError("disease_offset must be non-negative.")

    excluded = set()
    if exclude_edge_index is not None and exclude_edge_index.numel() > 0:
        if (
            exclude_edge_index.dim() != 2
            or exclude_edge_index.size(0) != 2
        ):
            raise ValueError(
                "exclude_edge_index must have shape [2, num_edges]."
            )
        excluded_edges = exclude_edge_index.detach().cpu().long()
        valid = (
            (excluded_edges[0] >= 0)
            & (excluded_edges[0] < num_lnc)
            & (excluded_edges[1] >= disease_offset)
            & (excluded_edges[1] < disease_offset + num_dis)
        )
        excluded_keys = (
            excluded_edges[0, valid] * num_dis
            + (excluded_edges[1, valid] - disease_offset)
        )
        excluded.update(excluded_keys.tolist())

    available = num_lnc * num_dis - len(excluded)
    if num_samples > available:
        raise ValueError(
            f"Cannot sample {num_samples} negatives; only {available} valid "
            "lnc-disease pairs remain."
        )

    sampled = torch.empty(0, dtype=torch.long)
    while sampled.numel() < num_samples:
        remaining = num_samples - sampled.numel()
        probe_size = max(1024, remaining * 2)
        src = torch.randint(num_lnc, (probe_size,), generator=generator)
        dst = torch.randint(num_dis, (probe_size,), generator=generator)
        keys = src * num_dis + dst

        accepted = []
        for key in keys.tolist():
            if key in excluded:
                continue
            excluded.add(key)
            accepted.append(key)
            if len(accepted) == remaining:
                break

        if accepted:
            accepted = torch.tensor(accepted, dtype=torch.long)
            sampled = torch.cat((sampled, accepted), dim=0)

    sampled_src = sampled // num_dis
    sampled_dst = sampled % num_dis + disease_offset
    return torch.stack((sampled_src, sampled_dst), dim=0)


def get_data(data_ID, output_dim, k=3, threshold=0.7):
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    base = f".\\dataset\\dataset{data_ID}\\"

    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        else:
            return x.to_numpy(dtype=np.float32)

    LCS = to_numpy(pd.read_csv(base + "LCS.csv", header=None))
    LGS = to_numpy(pd.read_csv(base + "LGS.csv", header=None))
    LFS = to_numpy(pd.read_csv(base + "LFS.csv", header=None))

    DCS = to_numpy(pd.read_csv(base + "DCS.csv", header=None))
    DGS = to_numpy(pd.read_csv(base + "DGS.csv", header=None))
    DSS = to_numpy(pd.read_csv(base + "DSS.csv", header=None))

    interaction = pd.read_csv(base + "LD_adjmat.csv", header=None)

    num_lnc = LCS.shape[0]
    target_size = DCS.shape[0]
    LCS = fix_similarity_matrix(LCS, num_lnc)
    LGS = fix_similarity_matrix(LGS, num_lnc)
    LFS = fix_similarity_matrix(LFS, num_lnc)
    DCS = fix_similarity_matrix(DCS, target_size)
    DGS = fix_similarity_matrix(DGS, target_size)
    DSS = fix_similarity_matrix(DSS, target_size)

    num_dis = DCS.shape[0]
    if interaction.shape != (num_lnc, num_dis):
        raise ValueError(
            f"LD_adjmat.csv has shape {interaction.shape}, expected "
            f"{(num_lnc, num_dis)}."
        )

    lnc_feature = torch.tensor(LCS)
    dis_feature = torch.tensor(DCS)

    if lnc_feature.size(1) < output_dim:
        lnc_feature = torch.cat(
            [
                lnc_feature,
                torch.zeros(num_lnc, output_dim - lnc_feature.size(1)),
            ],
            dim=1,
        )
    else:
        lnc_feature = lnc_feature[:, :output_dim]

    if dis_feature.size(1) < output_dim:
        dis_feature = torch.cat(
            [
                dis_feature,
                torch.zeros(num_dis, output_dim - dis_feature.size(1)),
            ],
            dim=1,
        )
    else:
        dis_feature = dis_feature[:, :output_dim]

    feature = torch.cat([lnc_feature, dis_feature], dim=0)

    def build_knn_hypergraph(sim_matrix, k):
        sim = torch.tensor(sim_matrix, dtype=torch.float)
        N = sim.shape[0]
        if N <= 1:
            return torch.ones((N, N), dtype=torch.float)

        sim_range = sim.max() - sim.min()
        if sim_range > 1e-8:
            sim = (sim - sim.min()) / sim_range
        else:
            sim = torch.zeros_like(sim)

        effective_k = min(k, N - 1)
        H = torch.zeros(N, N, dtype=torch.float)
        for i in range(N):
            sim_i = sim[i].clone()
            sim_i[i] = -1
            knn = torch.topk(sim_i, k=effective_k).indices
            H[i, i] = 1
            for j in knn:
                if sim_i[j] > threshold:
                    H[j, i] = 1
            if H[:, i].sum() == 1:
                H[knn[0], i] = 1
        return H

    # 构建基础超图边结构
    H_lcs = build_knn_hypergraph(LCS, k)
    H_lgs = build_knn_hypergraph(LGS, k)
    H_lfs = build_knn_hypergraph(LFS, k)
    H_lnc = concat_hypergraphs([H_lcs, H_lgs, H_lfs])
    edge_lnc = hypergraph_clique_expansion(H_lnc, node_offset=0)

    H_dcs = build_knn_hypergraph(DCS, k)
    H_dgs = build_knn_hypergraph(DGS, k)
    H_dss = build_knn_hypergraph(DSS, k)
    H_dis = concat_hypergraphs([H_dcs, H_dgs, H_dss])
    disease_offset = num_lnc
    edge_dis = hypergraph_clique_expansion(H_dis, node_offset=disease_offset)

    l, d = interaction.values.nonzero()
    edge_ld = torch.from_numpy(np.vstack([l, d + num_lnc])).long()

    struct_edge = torch.cat([edge_lnc, edge_dis], dim=1)
    struct_edge = torch.unique(struct_edge, dim=1)

    data = Data(x=feature, edge_index=struct_edge).cpu()
    data.num_nodes = feature.shape[0]
    data.feat_dim = output_dim
    data.pos_edge_label_index = edge_ld
    data.num_lnc = num_lnc
    data.num_dis = num_dis

    return data
