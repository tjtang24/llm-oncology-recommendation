"""Shared Hybrid Neural Collaborative Filtering model architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class HybridNCF(nn.Module):
    """GMF + MLP recommender with side features for users and items."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_feature_dim: int,
        item_feature_dim: int,
        gmf_dim: int,
        hidden_units: Sequence[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if not hidden_units:
            raise ValueError("hidden_units must contain at least one layer size.")

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.gmf_user_proj = nn.Linear(embedding_dim + user_feature_dim, gmf_dim)
        self.gmf_item_proj = nn.Linear(embedding_dim + item_feature_dim, gmf_dim)

        input_dim = (
            embedding_dim
            + user_feature_dim
            + embedding_dim
            + item_feature_dim
        )
        layers: list[nn.Module] = []
        for units in hidden_units:
            layers.extend(
                (
                    nn.Linear(input_dim, int(units)),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                )
            )
            input_dim = int(units)
        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(gmf_dim + int(hidden_units[-1]), 1)

    def forward(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)

        user_vector = torch.cat((user_embedding, user_features), dim=-1)
        item_vector = torch.cat((item_embedding, item_features), dim=-1)

        gmf_output = self.gmf_user_proj(user_vector) * self.gmf_item_proj(
            item_vector
        )
        mlp_output = self.mlp_layers(torch.cat((user_vector, item_vector), dim=-1))
        output = self.output_layer(torch.cat((gmf_output, mlp_output), dim=-1))
        return output.view(-1)
