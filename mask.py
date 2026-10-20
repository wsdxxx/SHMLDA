import torch
import torch.nn as nn


def mask_edge(edge_index, p, generator=None):
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if not 0.0 < p < 1.0:
        raise ValueError("Mask ratio p must be strictly between 0 and 1.")
    if edge_index.size(1) == 0:
        raise ValueError("Cannot mask an empty edge index.")

    probabilities = torch.full(
        (edge_index.size(1),),
        p,
        dtype=torch.float32,
        device=edge_index.device,
    )
    mask = torch.bernoulli(probabilities, generator=generator).to(torch.bool)

    if not mask.any():
        index = torch.randint(
            edge_index.size(1),
            (1,),
            generator=generator,
            device=edge_index.device,
        )
        mask[index] = True
    elif mask.all():
        index = torch.randint(
            edge_index.size(1),
            (1,),
            generator=generator,
            device=edge_index.device,
        )
        mask[index] = False

    return edge_index[:, ~mask], edge_index[:, mask]


class Mask(nn.Module):
    def __init__(self, p):
        super(Mask, self).__init__()
        if not 0.0 < p < 1.0:
            raise ValueError("Mask ratio p must be strictly between 0 and 1.")
        self.p = p

    def forward(self, edge_index, generator=None):
        remaining_edges, masked_edges = mask_edge(
            edge_index,
            p=self.p,
            generator=generator,
        )
        return remaining_edges, masked_edges
