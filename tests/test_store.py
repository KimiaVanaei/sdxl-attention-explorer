import torch

from sdxl_attention.store import (
    AttentionKey,
    AttentionStore,
)


def test_store_selects_conditional_cfg_branch() -> None:
    token_pattern = torch.tensor(
        [0.05, 0.15, 0.50, 0.20, 0.10],
        dtype=torch.float32,
    )

    conditional_attention = (
        token_pattern
        .reshape(1, 1, 1, 5)
        .repeat(
            1,   # batch
            3,   # heads
            16,  # spatial positions: 4x4
            1,   # tokens
        )
    )

    unconditional_attention = torch.full_like(
        conditional_attention,
        fill_value=999.0,
    )

    attention_probs = torch.cat(
        [
            unconditional_attention,
            conditional_attention,
        ],
        dim=0,
    )

    store = AttentionStore(
        image_size=(512, 512),
        use_classifier_free_guidance=True,
        allowed_resolutions={(4, 4)},
        storage_device="cpu",
    )

    store.add(
        attention_probs=attention_probs,
        layer_name=(
            "up_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    average = store.get_average(
        block_group="up",
        resolution=(4, 4),
    )

    assert average.shape == (4, 4, 5)

    assert torch.allclose(
        average[0, 0],
        token_pattern,
        atol=1e-6,
    )

    assert store.keys() == [
        AttentionKey(
            block_group="up",
            resolution=(4, 4),
        )
    ]


def test_store_groups_layers_by_unet_region() -> None:
    attention_probs = torch.softmax(
        torch.randn(1, 2, 16, 5),
        dim=-1,
    )

    store = AttentionStore(
        image_size=(512, 512),
        use_classifier_free_guidance=False,
    )

    store.add(
        attention_probs,
        (
            "down_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    store.add(
        attention_probs,
        (
            "mid_block.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    store.add(
        attention_probs,
        (
            "up_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    assert store.block_groups() == [
        "down",
        "mid",
        "up",
    ]

    assert store.resolutions("down") == [
        (4, 4)
    ]

    for block_group in (
        "down",
        "mid",
        "up",
    ):
        average = store.get_average(
            block_group,
            (4, 4),
        )

        token_sums = average.sum(dim=-1)

        assert torch.allclose(
            token_sums,
            torch.ones_like(token_sums),
            atol=1e-5,
        )


def test_store_reset_removes_all_maps() -> None:
    attention_probs = torch.softmax(
        torch.randn(1, 2, 16, 5),
        dim=-1,
    )

    store = AttentionStore(
        image_size=(512, 512),
        use_classifier_free_guidance=False,
    )

    store.add(
        attention_probs,
        (
            "up_blocks.1.attentions.0."
            "transformer_blocks.0.attn2.processor"
        ),
    )

    assert store.keys()

    store.reset()

    assert store.keys() == []
    assert store.summary() == "AttentionStore is empty."
