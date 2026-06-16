import torch
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
)

from sdxl_attention.processor import (
    RecordingCrossAttnProcessor,
)
from sdxl_attention.store import AttentionStore


def test_recording_processor_preserves_output() -> None:
    torch.manual_seed(1234)

    attention = Attention(
        query_dim=8,
        cross_attention_dim=6,
        heads=2,
        dim_head=4,
        dropout=0.0,
    )

    attention.eval()

    hidden_states = torch.randn(
        1,
        6,
        8,
    )

    encoder_hidden_states = torch.randn(
        1,
        5,
        6,
    )

    attention.set_processor(
        AttnProcessor()
    )

    with torch.no_grad():
        reference_output = attention(
            hidden_states=hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states
            ),
        )

    store = AttentionStore(
        image_size=(2, 3),
        use_classifier_free_guidance=False,
        allowed_resolutions={(2, 3)},
        storage_device="cpu",
    )

    attention.set_processor(
        RecordingCrossAttnProcessor(
            store=store,
            layer_name=(
                "up_blocks.1.attentions.0."
                "transformer_blocks.0."
                "attn2.processor"
            ),
        )
    )

    with torch.no_grad():
        recorded_output = attention(
            hidden_states=hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states
            ),
        )

    assert reference_output.shape == (
        recorded_output.shape
    )

    assert torch.allclose(
        reference_output,
        recorded_output,
        atol=1e-6,
        rtol=1e-5,
    )

    stored_map = store.get_average(
        block_group="up",
        resolution=(2, 3),
    )

    assert stored_map.shape == (2, 3, 5)

    token_sums = stored_map.sum(dim=-1)

    assert torch.allclose(
        token_sums,
        torch.ones_like(token_sums),
        atol=1e-5,
    )
