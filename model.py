
import inspect

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import (
    calculate_metrics,
    make_undirected_edge_index,
    sample_bipartite_negatives,
)
from SoftHGNN import SoftHGNN


def load_state_dict_compat(model, checkpoint, strict=False, verbose=True):
    """Load a checkpoint while tolerating optional model components.

    Several entry points (case studies, explainability, and older training
    runs) use checkpoints produced by different revisions of this model. A
    strict load makes a missing optional membership/SSM parameter fatal even
    when the rest of the network is compatible. This helper strips common
    wrappers, filters shape-incompatible entries, and reports what was skipped.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state dictionary or a mapping.")

    state_dict = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break

    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        normalized[key] = value

    current = model.state_dict()

    # Older membership decoders used one scalar (raw co-membership) while
    # the current decoder consumes raw and cross-type projected scores. Keep
    # the old scalar branch unchanged and leave the new projected feature at
    # zero when loading those checkpoints.
    membership_prefix = "edge_decoder.membership_mlp."
    for key in list(normalized):
        if not key.startswith(membership_prefix + "3."):
            continue
        legacy_key = membership_prefix + "2." + key[
            len(membership_prefix + "3."):
        ]
        if legacy_key in current and legacy_key not in normalized:
            normalized[legacy_key] = normalized.pop(key)

    input_weight_key = membership_prefix + "0.weight"
    input_weight = normalized.get(input_weight_key)
    target_input_weight = current.get(input_weight_key)
    if (
        hasattr(input_weight, "shape")
        and target_input_weight is not None
        and input_weight.dim() == 2
        and target_input_weight.dim() == 2
        and input_weight.shape[0] == target_input_weight.shape[0]
        and input_weight.shape[1] != target_input_weight.shape[1]
    ):
        migrated = input_weight.new_zeros(target_input_weight.shape)
        input_dim = min(input_weight.size(1), target_input_weight.size(1))
        migrated[:, :input_dim] = input_weight[:, :input_dim]
        normalized[input_weight_key] = migrated

    compatible = {}
    mismatched = []
    unexpected = []
    for key, value in normalized.items():
        target = current.get(key)
        if target is None:
            unexpected.append(key)
            continue
        if hasattr(value, "shape") and tuple(value.shape) != tuple(target.shape):
            mismatched.append((key, tuple(value.shape), tuple(target.shape)))
            continue
        compatible[key] = value

    missing = [key for key in current.keys() if key not in compatible]
    load_result = model.load_state_dict(compatible, strict=False)

    if verbose:
        if missing:
            print(
                "[checkpoint] Initialized missing parameters: "
                + ", ".join(missing[:8])
                + (" ..." if len(missing) > 8 else "")
            )
        if mismatched:
            print(
                "[checkpoint] Skipped shape-incompatible parameters: "
                + ", ".join(item[0] for item in mismatched[:8])
                + (" ..." if len(mismatched) > 8 else "")
            )
        if unexpected:
            print(
                "[checkpoint] Ignored unknown parameters: "
                + ", ".join(unexpected[:8])
                + (" ..." if len(unexpected) > 8 else "")
            )

    if strict and (missing or mismatched or unexpected):
        raise RuntimeError(
            "Checkpoint is not fully compatible with the model. "
            f"missing={len(missing)}, mismatched={len(mismatched)}, "
            f"unexpected={len(unexpected)}"
        )

    return load_result


class FFNBlock(nn.Module):
    def __init__(self, dim):
        super(FFNBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        return x + residual


class SoftHGNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_hyperedges,
        num_heads=4,
        temperature=1.0,
        structural_conditioning=True,
        structure_residual=False,
        use_ssm=True,
        ssm_expand=2,
        hyperedge_residual_scale=1.0,
    ):
        super(SoftHGNNEncoder, self).__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.backbone = SoftHGNN(
            embed_dim=hidden_channels,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            temperature=temperature,
            structural_conditioning=structural_conditioning,
            use_structure_residual=structure_residual,
            use_ssm=use_ssm,
            ssm_expand=ssm_expand,
            hyperedge_residual_scale=hyperedge_residual_scale,
        )
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.ffn = FFNBlock(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.GELU()

    def forward(self, x, edge_index=None, relation_edge_index=None):
        x = self.dropout(x)
        x = self.input_proj(x)
        x = x.unsqueeze(0)
        x = self.backbone(x, edge_index, relation_edge_index)
        x = x.squeeze(0)
        x = self.output_proj(x)
        x = self.activation(x)
        x = self.ffn(x)
        # 统一输出特征做 L2 标准化
        return F.normalize(x, p=2, dim=-1)


class EdgeDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_relations=None,
        use_membership=False,
        membership_dim=2,
    ):
        super(EdgeDecoder, self).__init__()
        if num_relations is not None and num_relations <= 0:
            raise ValueError("num_relations must be positive when provided.")
        if membership_dim <= 0:
            raise ValueError("membership_dim must be positive.")
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, 1))
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ELU()
        self.membership_mlp = None
        if use_membership:
            self.membership_dim = membership_dim
            if membership_dim % 2 != 0:
                raise ValueError(
                    "membership_dim must be even when membership fusion is "
                    "enabled."
                )
            membership_width = membership_dim // 2
            relation_width = max(hidden_channels // 2, 8)
            self.membership_lnc_proj = nn.Sequential(
                nn.Linear(membership_width, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )
            self.membership_dis_proj = nn.Sequential(
                nn.Linear(membership_width, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )
            self.membership_relation = nn.Parameter(
                torch.empty(hidden_channels, hidden_channels)
            )
            nn.init.xavier_uniform_(self.membership_relation)
            self.membership_pair_mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, relation_width),
                nn.GELU(),
            )
            self.membership_score_mlp = nn.Sequential(
                nn.Linear(relation_width + 3, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1),
            )
            self.membership_gate = nn.Sequential(
                nn.Linear(in_channels * 2 + relation_width + 3, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1),
            )
            self.membership_scale = nn.Parameter(torch.tensor(0.1))
            self.last_membership_gate = None
            self.last_membership_logit = None
            self.last_base_logit = None
        else:
            self.membership_dim = None

    def forward(self, z, edge, membership=None):
        x = z[edge[0]] * z[edge[1]]
        for mlp in self.mlps[:-1]:
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        self.last_base_logit = x.detach()
        self.last_membership_gate = None
        self.last_membership_logit = None
        if getattr(self, "membership_lnc_proj", None) is not None:
            if membership is None:
                raise ValueError(
                    "membership scores are required by this edge decoder."
                )
            membership = membership.reshape(-1, self.membership_dim).to(
                dtype=x.dtype
            )
            width = self.membership_dim // 2
            lnc_membership = membership[:, :width]
            dis_membership = membership[:, width:]
            q = self.membership_lnc_proj(lnc_membership)
            k = self.membership_dis_proj(dis_membership)
            bilinear = torch.sum(
                (q @ self.membership_relation) * k,
                dim=-1,
                keepdim=True,
            )
            cosine = F.cosine_similarity(q, k, dim=-1).unsqueeze(-1)
            raw_cosine = F.cosine_similarity(
                lnc_membership,
                dis_membership,
                dim=-1,
            ).unsqueeze(-1)
            pair_feature = self.membership_pair_mlp(
                torch.cat((q * k, torch.abs(q - k)), dim=-1)
            )
            relation_feature = torch.cat(
                (pair_feature, bilinear, cosine, raw_cosine),
                dim=-1,
            )
            membership_logit = self.membership_score_mlp(relation_feature)
            gate_input = torch.cat(
                (z[edge[0]], z[edge[1]], relation_feature),
                dim=-1,
            )
            gate = torch.sigmoid(self.membership_gate(gate_input))
            self.last_membership_gate = gate.detach()
            self.last_membership_logit = membership_logit.detach()
            x = x + self.membership_scale * gate * membership_logit
        return x.squeeze(-1)


class GMAE(nn.Module):
    def __init__(self, encoder, edge_decoder, mask):
        super(GMAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.mask = mask
        self.mask_disabled = False

    def loss_fn(self, pos_out, neg_out):
        pos_label = torch.ones_like(pos_out)
        neg_label = torch.zeros_like(neg_out)
        loss_pos = F.binary_cross_entropy_with_logits(pos_out, pos_label)
        loss_neg = F.binary_cross_entropy_with_logits(neg_out, neg_label)
        return loss_pos + loss_neg

    def _sample_negative_edges(self, data, num_samples, generator):
        negative_pool = getattr(data, "neg_edge_label_index", None)
        if negative_pool is None or negative_pool.numel() == 0:
            # Legacy entry points do not pre-build a negative pool. Keep them
            # working with the same bipartite index space as main.py.
            if hasattr(data, "num_lnc") and hasattr(data, "num_dis"):
                return sample_bipartite_negatives(
                    num_lnc=int(data.num_lnc),
                    num_dis=int(data.num_dis),
                    num_samples=num_samples,
                    exclude_edge_index=data.pos_edge_label_index,
                    generator=generator,
                    disease_offset=int(data.num_lnc),
                ).to(data.pos_edge_label_index.device)

            raise ValueError(
                "data.neg_edge_label_index is required, or data must provide "
                "num_lnc and num_dis for fallback negative sampling."
            )

        pool_size = negative_pool.size(1)
        if pool_size >= num_samples:
            indices = torch.randperm(
                pool_size,
                generator=generator,
                device=negative_pool.device,
            )[:num_samples]
        else:
            indices = []
            remaining = num_samples
            while remaining > 0:
                take = min(pool_size, remaining)
                indices.append(
                    torch.randperm(
                        pool_size,
                        generator=generator,
                        device=negative_pool.device,
                    )[:take]
                )
                remaining -= take
            indices = torch.cat(indices, dim=0)

        return negative_pool[:, indices]

    def _edge_membership(self, edge_index):
        decoder = self.edge_decoder
        if getattr(decoder, "membership_lnc_proj", None) is None:
            return None
        if hasattr(self.encoder.backbone, "edge_membership_pair"):
            source, target = self.encoder.backbone.edge_membership_pair(
                edge_index
            )
            return torch.cat((source, target), dim=-1)
        if hasattr(self.encoder.backbone, "edge_membership_features"):
            return self.encoder.backbone.edge_membership_features(edge_index)
        return self.encoder.backbone.edge_membership_score(edge_index)

    def train_epoch(
        self,
        data,
        optimizer,
        batch_size=8192,
        grad_norm=1.0,
        generator=None,
        seed=None,
    ):
        x, edge_index = data.x, data.edge_index
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(2024 if seed is None else seed)

        relation_edges = getattr(data, "relation_edge_index", None)
        mask_relation_structure = getattr(
            data,
            "mask_relation_structure",
            False,
        )

        if getattr(self, "mask_disabled", False):
            masked_edges = data.pos_edge_label_index
            visible_relation_edges = (
                relation_edges if mask_relation_structure else None
            )
        else:
            mask_callable = (
                self.mask.forward
                if isinstance(self.mask, nn.Module)
                else self.mask
            )
            mask_parameters = inspect.signature(mask_callable).parameters.values()
            accepts_generator = any(
                parameter.name == "generator"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in mask_parameters
            )
            if accepts_generator:
                remaining_edges, masked_edges = self.mask(
                    data.pos_edge_label_index,
                    generator=generator,
                )
            else:
                remaining_edges, masked_edges = self.mask(
                    data.pos_edge_label_index
                )
            visible_relation_edges = (
                remaining_edges if mask_relation_structure else None
            )

        if visible_relation_edges is not None:
            visible_relation_edges = make_undirected_edge_index(
                visible_relation_edges
            )

        neg_edges = self._sample_negative_edges(
            data,
            masked_edges.size(1),
            generator,
        )

        # PyTorch's DataLoader requires a CPU generator, while CUDA masking
        # requires a CUDA generator. Keep the explicit generator for masking
        # and let DataLoader use its CPU-side sampler when training on CUDA.
        loader_generator = (
            None
            if generator is not None and generator.device.type != "cpu"
            else generator
        )
        loader = DataLoader(
            range(masked_edges.size(1)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=loader_generator,
        )

        total_loss = 0
        num_batches = 0

        for perm in loader:
            optimizer.zero_grad()
            z = self.encoder(x, edge_index, visible_relation_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            batch_pos_membership = self._edge_membership(batch_masked_edges)
            batch_neg_membership = self._edge_membership(batch_neg_edges)

            pos_out = self.edge_decoder(
                z,
                batch_masked_edges,
                batch_pos_membership,
            )
            neg_out = self.edge_decoder(
                z,
                batch_neg_edges,
                batch_neg_membership,
            )

            loss = self.loss_fn(pos_out, neg_out)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def batch_predict(
        self,
        z,
        edges,
        memberships=None,
        batch_size=2 ** 16,
    ):
        if edges.numel() == 0:
            return z.new_zeros((0,))

        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            if memberships is None and (
                getattr(self.edge_decoder, "membership_lnc_proj", None)
                is not None
            ):
                # Keep inference helpers compatible with membership-aware
                # decoders when callers do not precompute the scores.
                membership = self._edge_membership(edge)
            else:
                membership = (
                    None
                    if memberships is None
                    else memberships[perm]
                )
            preds.append(
                self.edge_decoder(
                    z,
                    edge,
                    membership,
                ).reshape(-1).cpu()
            )
        return torch.cat(preds, dim=0)

    @torch.no_grad()
    def predict_scores(self, z, pos_edge_index, neg_edge_index):
        if pos_edge_index.numel() == 0 or neg_edge_index.numel() == 0:
            raise ValueError("Both positive and negative edges are required.")

        pos_pred = self.batch_predict(
            z,
            pos_edge_index,
            self._edge_membership(pos_edge_index),
        )
        neg_pred = self.batch_predict(
            z,
            neg_edge_index,
            self._edge_membership(neg_edge_index),
        )

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pred = torch.sigmoid(pred)

        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)

        y = y.cpu().numpy()
        pred = pred.cpu().numpy()

        return y, pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index, threshold=0.5):
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        y, pred = self.predict_scores(z, pos_edge_index, neg_edge_index)

        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)

        temp = (pred >= threshold).astype(int)

        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp)

        return auc, ap, acc, sen, pre, spe, F1, mcc
