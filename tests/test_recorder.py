from __future__ import annotations

from typing import Any

import pytest

from sdxl_attention.processor import (
    RecordingCrossAttnProcessor,
)
from sdxl_attention.recorder import AttentionRecorder
from sdxl_attention.store import AttentionStore


class FakeUNet:
    """
    Minimal stand-in for Diffusers' attention-processor interface.
    """

    def __init__(
        self,
        processors: dict[str, Any],
    ) -> None:
        self._processors = dict(processors)

    @property
    def attn_processors(self) -> dict[str, Any]:
        return dict(self._processors)

    def set_attn_processor(
        self,
        processors: dict[str, Any],
    ) -> None:
        if set(processors) != set(self._processors):
            raise ValueError(
                "Processor names do not match."
            )

        self._processors = dict(processors)


def create_fake_unet() -> tuple[
    FakeUNet,
    object,
    object,
]:
    self_attention_processor = object()
    cross_attention_processor = object()

    unet = FakeUNet(
        {
            (
                "down_blocks.0.attentions.0."
                "transformer_blocks.0."
                "attn1.processor"
            ): self_attention_processor,
            (
                "down_blocks.0.attentions.0."
                "transformer_blocks.0."
                "attn2.processor"
            ): cross_attention_processor,
        }
    )

    return (
        unet,
        self_attention_processor,
        cross_attention_processor,
    )


def create_store() -> AttentionStore:
    return AttentionStore(
        image_size=(512, 512),
        storage_device="cpu",
    )


def test_recorder_replaces_only_cross_attention() -> None:
    (
        unet,
        original_self_attention,
        original_cross_attention,
    ) = create_fake_unet()

    recorder = AttentionRecorder(
        unet=unet,
        store=create_store(),
    )

    with recorder:
        current = unet.attn_processors

        self_name = (
            "down_blocks.0.attentions.0."
            "transformer_blocks.0."
            "attn1.processor"
        )

        cross_name = (
            "down_blocks.0.attentions.0."
            "transformer_blocks.0."
            "attn2.processor"
        )

        assert recorder.is_active
        assert recorder.recording_count == 1

        assert (
            current[self_name]
            is original_self_attention
        )

        assert isinstance(
            current[cross_name],
            RecordingCrossAttnProcessor,
        )

    restored = unet.attn_processors

    assert (
        restored[self_name]
        is original_self_attention
    )

    assert (
        restored[cross_name]
        is original_cross_attention
    )

    assert not recorder.is_active


def test_recorder_restores_after_exception() -> None:
    (
        unet,
        original_self_attention,
        original_cross_attention,
    ) = create_fake_unet()

    recorder = AttentionRecorder(
        unet=unet,
        store=create_store(),
    )

    with pytest.raises(
        RuntimeError,
        match="generation failed",
    ):
        with recorder:
            raise RuntimeError(
                "generation failed"
            )

    processors = unet.attn_processors

    assert list(processors.values()) == [
        original_self_attention,
        original_cross_attention,
    ]

    assert not recorder.is_active


def test_recorder_rejects_double_installation() -> None:
    unet, _, _ = create_fake_unet()

    recorder = AttentionRecorder(
        unet=unet,
        store=create_store(),
    )

    recorder.install()

    try:
        with pytest.raises(
            RuntimeError,
            match="already active",
        ):
            recorder.install()
    finally:
        recorder.restore()
