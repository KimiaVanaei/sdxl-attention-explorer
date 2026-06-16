from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F

from sdxl_attention.store import (
    AttentionKey,
    AttentionStore,
    Resolution,
)


@dataclass(frozen=True)
class AttentionMapResult:
    """
    Result of aggregating attention for one or more token positions.

    Attributes
    ----------
    raw:
        Image-sized attention map before visualization normalization.

    normalized:
        Image-sized attention map scaled to the interval [0, 1].

    components:
        Individually resized maps used to construct the aggregate.
        Keys identify the U-Net region and original resolution.

    token_positions:
        Token positions included in this result.

    output_size:
        Final map size as (height, width).
    """

    raw: torch.Tensor
    normalized: torch.Tensor
    components: dict[AttentionKey, torch.Tensor]
    token_positions: tuple[int, ...]
    output_size: Resolution


def normalize_attention_map(
    attention_map: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Min-max normalize a two-dimensional map to [0, 1].

    Constant maps are returned as zeros.
    """
    if attention_map.ndim != 2:
        raise ValueError(
            "attention_map must be two-dimensional, "
            f"but received shape {tuple(attention_map.shape)}."
        )

    if eps <= 0:
        raise ValueError("eps must be positive.")

    minimum = attention_map.min()
    maximum = attention_map.max()
    value_range = maximum - minimum

    if value_range.item() < eps:
        return torch.zeros_like(attention_map)

    return (attention_map - minimum) / value_range


def resize_attention_map(
    attention_map: torch.Tensor,
    output_size: Resolution,
) -> torch.Tensor:
    """
    Resize a two-dimensional attention map using bilinear interpolation.
    """
    if attention_map.ndim != 2:
        raise ValueError(
            "attention_map must be two-dimensional, "
            f"but received shape {tuple(attention_map.shape)}."
        )

    output_height, output_width = output_size

    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "Output height and width must be positive."
        )

    map_4d = attention_map[
        None,
        None,
        :,
        :,
    ]

    resized = F.interpolate(
        map_4d,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )

    return resized[0, 0]


def _normalize_token_positions(
    token_positions: int | Iterable[int],
) -> tuple[int, ...]:
    if isinstance(token_positions, int):
        positions = (token_positions,)
    else:
        positions = tuple(token_positions)

    if not positions:
        raise ValueError(
            "At least one token position is required."
        )

    if any(position < 0 for position in positions):
        raise ValueError(
            "Token positions must be non-negative."
        )

    # Remove duplicates while preserving order.
    return tuple(dict.fromkeys(positions))


def aggregate_token_attention(
    store: AttentionStore,
    token_positions: int | Iterable[int],
    output_size: Resolution,
    *,
    block_groups: Iterable[str] | None = None,
    resolutions: Iterable[Resolution] | None = None,
    token_reduction: Literal["mean", "sum"] = "mean",
) -> AttentionMapResult:
    """
    Aggregate stored attention maps for one word or token sequence.

    Parameters
    ----------
    store:
        A populated ``AttentionStore``.

    token_positions:
        One token position or multiple subword-token positions.

        For example:

            3

        or:

            [3, 4]

    output_size:
        Final map size as ``(height, width)``.

    block_groups:
        Optional U-Net regions to include, such as ``["up"]`` or
        ``["down", "up"]``. By default, all recorded regions are used.

    resolutions:
        Optional spatial resolutions to include. By default, all
        recorded resolutions are used.

    token_reduction:
        How maps from multiple subword tokens are combined:

        - ``"mean"`` gives every subword equal average weight.
        - ``"sum"`` adds their attention values.

    Notes
    -----
    Each selected block-group/resolution component is resized to the
    requested output size. Components are then averaged with equal
    weight.

    Min-max normalization is applied only after component aggregation.
    """
    positions = _normalize_token_positions(
        token_positions
    )

    if token_reduction not in {"mean", "sum"}:
        raise ValueError(
            "token_reduction must be 'mean' or 'sum'."
        )

    selected_groups = (
        None
        if block_groups is None
        else set(block_groups)
    )

    selected_resolutions = (
        None
        if resolutions is None
        else {
            store.normalize_resolution(resolution)
            for resolution in resolutions
        }
    )

    matching_keys = [
        key
        for key in store.keys()
        if (
            selected_groups is None
            or key.block_group in selected_groups
        )
        and (
            selected_resolutions is None
            or key.resolution in selected_resolutions
        )
    ]

    if not matching_keys:
        raise ValueError(
            "No stored attention maps match the requested "
            "block groups and resolutions."
        )

    components: dict[
        AttentionKey,
        torch.Tensor,
    ] = {}

    for key in matching_keys:
        attention = store.get_average(
            block_group=key.block_group,
            resolution=key.resolution,
        )

        number_of_tokens = attention.shape[-1]

        invalid_positions = [
            position
            for position in positions
            if position >= number_of_tokens
        ]

        if invalid_positions:
            raise IndexError(
                f"Token positions {invalid_positions} are invalid "
                f"for an attention tensor with "
                f"{number_of_tokens} token positions."
            )

        selected = attention[
            ...,
            list(positions),
        ]

        if token_reduction == "mean":
            token_map = selected.mean(dim=-1)
        else:
            token_map = selected.sum(dim=-1)

        components[key] = resize_attention_map(
            attention_map=token_map,
            output_size=output_size,
        )

    raw_map = torch.stack(
        list(components.values()),
        dim=0,
    ).mean(dim=0)

    normalized_map = normalize_attention_map(
        raw_map
    )

    return AttentionMapResult(
        raw=raw_map,
        normalized=normalized_map,
        components=components,
        token_positions=positions,
        output_size=output_size,
    )
