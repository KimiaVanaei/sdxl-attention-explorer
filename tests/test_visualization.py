import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from sdxl_attention.aggregation import (
    AttentionMapResult,
)
from sdxl_attention.visualization import (
    create_rgba_heatmap,
    image_to_numpy,
    overlay_attention_on_image,
    plot_attention_grid,
    plot_attention_result,
)


def create_test_result() -> AttentionMapResult:
    raw_map = torch.tensor(
        [
            [0.0, 0.25],
            [0.5, 1.0],
        ],
        dtype=torch.float32,
    )

    return AttentionMapResult(
        raw=raw_map,
        normalized=raw_map,
        components={},
        token_positions=(3,),
        output_size=(2, 2),
    )


def test_image_to_numpy_converts_pil_image() -> None:
    image = Image.new(
        "RGB",
        size=(2, 2),
        color=(255, 255, 255),
    )

    image_array = image_to_numpy(image)

    assert image_array.shape == (2, 2, 3)
    assert image_array.dtype == np.uint8


def test_create_rgba_heatmap_uses_attention_alpha() -> None:
    normalized_map = torch.tensor(
        [
            [0.0, 0.25],
            [0.5, 1.0],
        ]
    )

    rgba = create_rgba_heatmap(
        normalized_map,
        alpha=0.8,
        alpha_mode="attention",
    )

    assert rgba.shape == (2, 2, 4)

    assert rgba[0, 0, 3] == pytest.approx(
        0.0
    )

    assert rgba[1, 1, 3] == pytest.approx(
        0.8
    )


def test_overlay_attention_on_image() -> None:
    image = Image.new(
        "RGB",
        size=(2, 2),
        color=(255, 255, 255),
    )

    normalized_map = torch.tensor(
        [
            [0.0, 0.25],
            [0.5, 1.0],
        ]
    )

    overlay = overlay_attention_on_image(
        image=image,
        normalized_map=normalized_map,
    )

    assert overlay.shape == (2, 2, 3)
    assert overlay.dtype == np.uint8

    # A zero-attention location remains unchanged.
    assert np.array_equal(
        overlay[0, 0],
        np.array([255, 255, 255]),
    )

    # A high-attention location receives the heatmap color.
    assert not np.array_equal(
        overlay[1, 1],
        np.array([255, 255, 255]),
    )


def test_overlay_rejects_size_mismatch() -> None:
    image = Image.new(
        "RGB",
        size=(4, 4),
        color=(255, 255, 255),
    )

    normalized_map = torch.zeros(2, 2)

    with pytest.raises(
        ValueError,
        match="sizes must match",
    ):
        overlay_attention_on_image(
            image=image,
            normalized_map=normalized_map,
        )


def test_plot_attention_result_saves_file(
    tmp_path,
) -> None:
    image = Image.new(
        "RGB",
        size=(2, 2),
        color=(255, 255, 255),
    )

    output_path = (
        tmp_path / "single_attention.png"
    )

    figure = plot_attention_result(
        image=image,
        result=create_test_result(),
        token_label="cat",
        save_path=output_path,
        show=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_plot_attention_grid_saves_file(
    tmp_path,
) -> None:
    image = Image.new(
        "RGB",
        size=(2, 2),
        color=(255, 255, 255),
    )

    output_path = (
        tmp_path / "attention_grid.png"
    )

    result = create_test_result()

    figure = plot_attention_grid(
        image=image,
        results={
            "cat": result,
            "bicycle": result,
        },
        save_path=output_path,
        show=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)
