from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure
from PIL import Image

from sdxl_attention.aggregation import AttentionMapResult


ImageInput = Image.Image | np.ndarray
AlphaMode = Literal["attention", "constant"]


def attention_map_to_numpy(
    attention_map: torch.Tensor,
) -> np.ndarray:
    """
    Convert a two-dimensional attention tensor to a NumPy array.

    Values are clipped to the interval [0, 1].
    """
    if attention_map.ndim != 2:
        raise ValueError(
            "attention_map must be two-dimensional, "
            f"but received {tuple(attention_map.shape)}."
        )

    return (
        attention_map
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
    )


def image_to_numpy(
    image: ImageInput,
) -> np.ndarray:
    """
    Convert a PIL image or NumPy image to an RGB uint8 array.
    """
    if isinstance(image, Image.Image):
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

    image_array = np.asarray(image)

    if image_array.ndim == 2:
        image_array = np.repeat(
            image_array[..., None],
            repeats=3,
            axis=-1,
        )

    if image_array.ndim != 3:
        raise ValueError(
            "image must be a two-dimensional grayscale image "
            "or a three-dimensional color image."
        )

    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    if image_array.shape[-1] != 3:
        raise ValueError(
            "A color image must contain three RGB channels "
            "or four RGBA channels."
        )

    if np.issubdtype(
        image_array.dtype,
        np.floating,
    ):
        maximum = float(image_array.max())

        if maximum <= 1.0:
            image_array = image_array * 255.0

    return np.clip(
        image_array,
        0,
        255,
    ).astype(np.uint8)


def create_rgba_heatmap(
    normalized_map: torch.Tensor,
    *,
    alpha: float = 0.75,
    cmap: str = "magma",
    alpha_mode: AlphaMode = "attention",
) -> np.ndarray:
    """
    Convert a normalized attention map into an RGBA heatmap.

    Parameters
    ----------
    normalized_map:
        Two-dimensional map whose values are expected in [0, 1].

    alpha:
        Maximum opacity of the generated heatmap.

    cmap:
        Name of a Matplotlib colormap.

    alpha_mode:
        ``"attention"``:
            Opacity is proportional to the attention value. Areas with
            low attention become transparent.

        ``"constant"``:
            All heatmap pixels use the same opacity.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if alpha_mode not in {
        "attention",
        "constant",
    }:
        raise ValueError(
            "alpha_mode must be 'attention' or 'constant'."
        )

    map_array = attention_map_to_numpy(
        normalized_map
    )

    try:
        colormap = colormaps.get_cmap(cmap)
    except ValueError as error:
        raise ValueError(
            f"Unknown Matplotlib colormap: {cmap!r}."
        ) from error

    rgba = np.asarray(
        colormap(map_array),
        dtype=np.float32,
    )

    if alpha_mode == "attention":
        rgba[..., 3] = alpha * map_array
    else:
        rgba[..., 3] = alpha

    return rgba


def overlay_attention_on_image(
    image: ImageInput,
    normalized_map: torch.Tensor,
    *,
    alpha: float = 0.75,
    cmap: str = "magma",
    alpha_mode: AlphaMode = "attention",
) -> np.ndarray:
    """
    Alpha-composite a normalized attention map over an image.

    Returns
    -------
    np.ndarray
        RGB uint8 image shaped ``[height, width, 3]``.
    """
    image_array = image_to_numpy(image)

    map_array = attention_map_to_numpy(
        normalized_map
    )

    if image_array.shape[:2] != map_array.shape:
        raise ValueError(
            "Image and attention map sizes must match. "
            f"Image size: {image_array.shape[:2]}, "
            f"map size: {map_array.shape}."
        )

    rgba_heatmap = create_rgba_heatmap(
        normalized_map,
        alpha=alpha,
        cmap=cmap,
        alpha_mode=alpha_mode,
    )

    image_float = (
        image_array.astype(np.float32) / 255.0
    )

    heatmap_rgb = rgba_heatmap[..., :3]
    heatmap_alpha = rgba_heatmap[..., 3:4]

    composite = (
        image_float * (1.0 - heatmap_alpha)
        + heatmap_rgb * heatmap_alpha
    )

    return (
        np.clip(composite, 0.0, 1.0)
        * 255.0
    ).round().astype(np.uint8)


def _save_figure(
    figure: Figure,
    save_path: str | Path | None,
    *,
    dpi: int,
) -> None:
    if save_path is None:
        return

    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    output_path = Path(save_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )


def plot_attention_result(
    image: ImageInput,
    result: AttentionMapResult,
    *,
    token_label: str,
    alpha: float = 0.75,
    cmap: str = "magma",
    alpha_mode: AlphaMode = "attention",
    save_path: str | Path | None = None,
    dpi: int = 150,
    show: bool = True,
) -> Figure:
    """
    Plot the generated image, heatmap, and attention overlay.
    """
    if not token_label.strip():
        raise ValueError(
            "token_label cannot be empty."
        )

    image_array = image_to_numpy(image)

    if image_array.shape[:2] != result.output_size:
        raise ValueError(
            "The image size does not match the attention result. "
            f"Image size: {image_array.shape[:2]}, "
            f"result size: {result.output_size}."
        )

    normalized_array = attention_map_to_numpy(
        result.normalized
    )

    overlay = overlay_attention_on_image(
        image=image_array,
        normalized_map=result.normalized,
        alpha=alpha,
        cmap=cmap,
        alpha_mode=alpha_mode,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(image_array)
    axes[0].set_title("Generated image")

    heatmap_artist = axes[1].imshow(
        normalized_array,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    token_positions = ", ".join(
        str(position)
        for position in result.token_positions
    )

    axes[1].set_title(
        f'Attention for "{token_label}"\n'
        f"token position(s): {token_positions}"
    )

    axes[2].imshow(overlay)
    axes[2].set_title("Attention overlay")

    for axis in axes:
        axis.axis("off")

    figure.colorbar(
        heatmap_artist,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Normalized attention",
    )

    figure.tight_layout()

    _save_figure(
        figure,
        save_path,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure


def plot_attention_grid(
    image: ImageInput,
    results: Mapping[str, AttentionMapResult],
    *,
    alpha: float = 0.75,
    cmap: str = "magma",
    alpha_mode: AlphaMode = "attention",
    save_path: str | Path | None = None,
    dpi: int = 150,
    show: bool = True,
) -> Figure:
    """
    Compare attention maps for several token labels.

    The first row contains normalized heatmaps.
    The second row contains image overlays.
    """
    if not results:
        raise ValueError(
            "results cannot be empty."
        )

    image_array = image_to_numpy(image)
    number_of_results = len(results)

    figure, axes = plt.subplots(
        2,
        number_of_results,
        figsize=(4.5 * number_of_results, 8),
        squeeze=False,
    )

    for column, (
        token_label,
        result,
    ) in enumerate(results.items()):
        if image_array.shape[:2] != result.output_size:
            raise ValueError(
                f"The result for {token_label!r} has output size "
                f"{result.output_size}, but the image size is "
                f"{image_array.shape[:2]}."
            )

        normalized_array = attention_map_to_numpy(
            result.normalized
        )

        overlay = overlay_attention_on_image(
            image=image_array,
            normalized_map=result.normalized,
            alpha=alpha,
            cmap=cmap,
            alpha_mode=alpha_mode,
        )

        token_positions = ", ".join(
            str(position)
            for position in result.token_positions
        )

        axes[0, column].imshow(
            normalized_array,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )

        axes[0, column].set_title(
            f'"{token_label}"\n'
            f"token position(s): {token_positions}"
        )

        axes[1, column].imshow(overlay)
        axes[1, column].set_title(
            f'"{token_label}" overlay'
        )

        axes[0, column].axis("off")
        axes[1, column].axis("off")

    figure.suptitle(
        "Prompt-token cross-attention",
        fontsize=16,
    )

    figure.tight_layout()

    _save_figure(
        figure,
        save_path,
        dpi=dpi,
    )

    if show:
        plt.show()

    return figure
