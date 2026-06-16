import pytest
import torch

from sdxl_attention.aggregation import (
    aggregate_token_attention,
    normalize_attention_map,
    resize_attention_map,
)
from sdxl_attention.store import (
    AttentionKey,
    AttentionStore,
)


def test_normalize_attention_map() -> None:
    attention_map = torch.tensor(
        [
            [2.0, 4.0],
            [6.0, 10.0],
        ]
    )

    normalized = normalize_attention_map(
        attention_map
    )

    assert normalized.min().item() == 0.0
    assert normalized.max().item() == 1.0

    assert torch.allclose(
        normalized,
        torch.tensor(
            [
                [0.0, 0.25],
                [0.5, 1.0],
            ]
        ),
    )


def test_normalize_constant_map_returns_zeros() -> None:
    attention_map = torch.full(
        (4, 4),
        0.5,
    )

    normalized = normalize_attention_map(
        attention_map
    )

    assert torch.equal(
        normalized,
        torch.zeros_like(attention_map),
    )


def test_resize_attention_map() -> None:
    attention_map = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    resized = resize_attention_map(
        attention_map,
        output_size=(8, 6),
    )

    assert resized.shape == (8, 6)


def test_aggregate_token_attention() -> None:
    down_probabilities = torch.tensor(
        [0.1, 0.7, 0.2],
        dtype=torch.float32,
    )

    up_probabilities = torch.tensor(
        [0.2, 0.3, 0.5],
        dtype=torch.float32,
    )

    down_attention = (
        down_probabilities
        .reshape(1, 1, 1, 3)
        .repeat(1, 2, 16, 1)
    )

    up_attention = (
        up_probabilities
        .reshape(1, 1, 1, 3)
        .repeat(1, 2, 16, 1)
    )

    store = AttentionStore(
        image_size=(8, 8),
        use_classifier_free_guidance=False,
        storage_device="cpu",
    )

    store.add(
        down_attention,
        (
            "down_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    store.add(
        up_attention,
        (
            "up_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    result = aggregate_token_attention(
        store=store,
        token_positions=1,
        output_size=(8, 8),
    )

    assert result.raw.shape == (8, 8)
    assert result.normalized.shape == (8, 8)
    assert result.token_positions == (1,)

    # Token 1 has attention 0.7 in the down component
    # and 0.3 in the up component. Their equal average is 0.5.
    assert torch.allclose(
        result.raw,
        torch.full((8, 8), 0.5),
        atol=1e-6,
    )

    assert set(result.components) == {
        AttentionKey("down", (4, 4)),
        AttentionKey("up", (4, 4)),
    }


def test_aggregate_multiple_token_positions() -> None:
    probabilities = torch.tensor(
        [0.2, 0.3, 0.5],
        dtype=torch.float32,
    )

    attention = (
        probabilities
        .reshape(1, 1, 1, 3)
        .repeat(1, 2, 16, 1)
    )

    store = AttentionStore(
        image_size=(8, 8),
        use_classifier_free_guidance=False,
    )

    store.add(
        attention,
        (
            "up_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    result = aggregate_token_attention(
        store=store,
        token_positions=[0, 2],
        output_size=(8, 8),
        token_reduction="mean",
    )

    # Mean of token values 0.2 and 0.5.
    assert torch.allclose(
        result.raw,
        torch.full((8, 8), 0.35),
        atol=1e-6,
    )


def test_aggregate_rejects_invalid_token_position() -> None:
    attention = torch.softmax(
        torch.randn(1, 2, 16, 3),
        dim=-1,
    )

    store = AttentionStore(
        image_size=(8, 8),
        use_classifier_free_guidance=False,
    )

    store.add(
        attention,
        (
            "up_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    with pytest.raises(
        IndexError,
        match="invalid",
    ):
        aggregate_token_attention(
            store=store,
            token_positions=10,
            output_size=(8, 8),
        )
