from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import get_token

from sdxl_attention import (
    AttentionRecorder,
    AttentionStore,
    aggregate_token_attention,
    find_token_positions,
    format_prompt_tokens,
    inspect_sdxl_prompt,
    plot_attention_grid,
    plot_attention_result,
)


DEFAULT_MODEL_ID = (
    "stabilityai/stable-diffusion-xl-base-1.0"
)

DEFAULT_PROMPT = (
    "a red cat sitting beside a blue bicycle"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an SDXL image and visualize "
            "prompt-token cross-attention maps."
        )
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt used for image generation.",
    )

    parser.add_argument(
        "--tokens",
        nargs="+",
        default=[
            "red",
            "cat",
            "blue",
            "bicycle",
        ],
        help=(
            "Readable prompt tokens to visualize. "
            "Each item must match decoded tokenizer text."
        ),
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model repository identifier.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Generated image height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Generated image width.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of denoising inference steps.",
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[16, 32],
        help=(
            "Square attention-grid resolutions to store. "
            "Default: 16 32."
        ),
    )

    parser.add_argument(
        "--block-groups",
        nargs="+",
        choices=[
            "down",
            "mid",
            "up",
        ],
        default=[
            "down",
            "mid",
            "up",
        ],
        help="U-Net block groups to include.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory in which generated files are saved.",
    )

    return parser.parse_args()


def require_huggingface_token() -> str:
    """
    Retrieve the configured Hugging Face access token.

    The token may come from:

    - the HF_TOKEN environment variable;
    - a previous `hf auth login`;
    - a previous `huggingface_hub.login()` call.
    """
    token = get_token()

    if token is None:
        raise RuntimeError(
            "No Hugging Face token was found.\n\n"
            "Authenticate using one of these methods:\n"
            "  1. Set the HF_TOKEN environment variable.\n"
            "  2. Run `hf auth login` in a terminal.\n"
            "  3. In Colab, load HF_TOKEN from Colab Secrets.\n\n"
            "Do not hardcode the token in this script."
        )

    return token


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.height <= 0:
        raise ValueError(
            "--height must be positive."
        )

    if arguments.width <= 0:
        raise ValueError(
            "--width must be positive."
        )

    if arguments.steps <= 0:
        raise ValueError(
            "--steps must be positive."
        )

    if arguments.guidance_scale < 0:
        raise ValueError(
            "--guidance-scale cannot be negative."
        )

    if any(
        resolution <= 0
        for resolution in arguments.resolutions
    ):
        raise ValueError(
            "All attention resolutions must be positive."
        )


def safe_filename(value: str) -> str:
    """
    Convert a token label into a safe filename component.
    """
    filename = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value.strip(),
    ).strip("_")

    return filename or "token"


def load_pipeline(
    model_id: str,
    token: str,
) -> StableDiffusionXLPipeline:
    """
    Load SDXL using authenticated Hugging Face access.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for this example."
        )

    pipeline = (
        StableDiffusionXLPipeline
        .from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            token=token,
        )
    )

    # Move modules between CPU and GPU as needed.
    # This is more suitable for common Colab GPUs than
    # permanently placing the whole SDXL pipeline on CUDA.
    pipeline.enable_model_cpu_offload()

    pipeline.set_progress_bar_config(
        disable=False
    )

    return pipeline


def resolve_token_positions(
    pipeline: StableDiffusionXLPipeline,
    prompt: str,
    requested_tokens: list[str],
) -> tuple[dict[str, list[int]], str]:
    """
    Match readable token labels to SDXL tokenizer positions.
    """
    prompt_tokens = inspect_sdxl_prompt(
        pipeline,
        prompt,
        include_special_tokens=True,
    )

    tokenizer_1_tokens = prompt_tokens[
        "tokenizer_1"
    ]

    formatted_tokens = format_prompt_tokens(
        tokenizer_1_tokens
    )

    print("\nTokenizer positions:\n")
    print(formatted_tokens)

    positions_by_label: dict[
        str,
        list[int],
    ] = {}

    for token_label in requested_tokens:
        positions = find_token_positions(
            tokenizer_1_tokens,
            token_label,
            match="exact",
            case_sensitive=False,
        )

        if not positions:
            raise ValueError(
                f"Could not find token {token_label!r} "
                "in the prompt.\n\n"
                "Inspect the printed tokenizer table and use "
                "the decoded token text shown there."
            )

        positions_by_label[token_label] = positions

        print(
            f"{token_label!r}: "
            f"position(s) {positions}"
        )

    return positions_by_label, formatted_tokens


def create_attention_results(
    store: AttentionStore,
    positions_by_label: dict[str, list[int]],
    output_size: tuple[int, int],
    block_groups: list[str],
    resolutions: list[int],
) -> dict[str, Any]:
    """
    Aggregate an image-sized attention map for every requested token.
    """
    selected_resolutions = [
        (resolution, resolution)
        for resolution in resolutions
    ]

    return {
        token_label: aggregate_token_attention(
            store=store,
            token_positions=positions,
            output_size=output_size,
            block_groups=block_groups,
            resolutions=selected_resolutions,
            token_reduction="mean",
        )
        for token_label, positions
        in positions_by_label.items()
    }


def save_outputs(
    *,
    output_directory: Path,
    generated_image: Any,
    attention_results: dict[str, Any],
    prompt: str,
    model_id: str,
    token_positions: dict[str, list[int]],
    tokenizer_table: str,
    store: AttentionStore,
    arguments: argparse.Namespace,
) -> None:
    """
    Save the image, plots, tokenizer table, and generation metadata.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_image_path = (
        output_directory
        / "generated_image.png"
    )

    generated_image.save(
        generated_image_path
    )

    tokenizer_path = (
        output_directory
        / "prompt_tokens.txt"
    )

    tokenizer_path.write_text(
        tokenizer_table,
        encoding="utf-8",
    )

    for token_label, result in (
        attention_results.items()
    ):
        token_filename = safe_filename(
            token_label
        )

        figure = plot_attention_result(
            image=generated_image,
            result=result,
            token_label=token_label,
            save_path=(
                output_directory
                / f"attention_{token_filename}.png"
            ),
            show=False,
        )

        plt.close(figure)

    grid_figure = plot_attention_grid(
        image=generated_image,
        results=attention_results,
        save_path=(
            output_directory
            / "attention_grid.png"
        ),
        show=False,
    )

    plt.close(grid_figure)

    metadata = {
        "model_id": model_id,
        "prompt": prompt,
        "tokens": token_positions,
        "height": arguments.height,
        "width": arguments.width,
        "steps": arguments.steps,
        "guidance_scale": (
            arguments.guidance_scale
        ),
        "seed": arguments.seed,
        "block_groups": (
            arguments.block_groups
        ),
        "resolutions": (
            arguments.resolutions
        ),
        "store_summary": store.summary(),
    }

    metadata_path = (
        output_directory
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")

    for output_path in sorted(
        output_directory.iterdir()
    ):
        print(f"  {output_path}")


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    hf_token = require_huggingface_token()

    print(
        "Hugging Face authentication found."
    )

    print(
        f"Loading model: {arguments.model_id}"
    )

    pipeline = None
    pipeline_output = None

    try:
        pipeline = load_pipeline(
            model_id=arguments.model_id,
            token=hf_token,
        )

        (
            positions_by_label,
            tokenizer_table,
        ) = resolve_token_positions(
            pipeline=pipeline,
            prompt=arguments.prompt,
            requested_tokens=arguments.tokens,
        )

        use_cfg = (
            arguments.guidance_scale > 1.0
        )

        store = AttentionStore(
            image_size=(
                arguments.height,
                arguments.width,
            ),
            use_classifier_free_guidance=(
                use_cfg
            ),
            batch_index=0,
            allowed_resolutions=(
                arguments.resolutions
            ),
            allowed_block_groups=(
                arguments.block_groups
            ),
            storage_device="cpu",
        )

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(arguments.seed)

        print("\nGenerating image and recording attention...")

        with AttentionRecorder(
            pipeline.unet,
            store,
        ) as recorder:
            print(
                "Recording cross-attention from "
                f"{recorder.recording_count} layers."
            )

            with torch.inference_mode():
                pipeline_output = pipeline(
                    prompt=arguments.prompt,
                    height=arguments.height,
                    width=arguments.width,
                    num_inference_steps=(
                        arguments.steps
                    ),
                    guidance_scale=(
                        arguments.guidance_scale
                    ),
                    generator=generator,
                )

        generated_image = (
            pipeline_output.images[0]
        )

        print()
        print(store.summary())

        if not store.keys():
            raise RuntimeError(
                "No attention maps were stored. "
                "Check --resolutions and --block-groups."
            )

        output_size = (
            generated_image.height,
            generated_image.width,
        )

        attention_results = (
            create_attention_results(
                store=store,
                positions_by_label=(
                    positions_by_label
                ),
                output_size=output_size,
                block_groups=(
                    arguments.block_groups
                ),
                resolutions=(
                    arguments.resolutions
                ),
            )
        )

        save_outputs(
            output_directory=(
                arguments.output_dir
            ),
            generated_image=generated_image,
            attention_results=(
                attention_results
            ),
            prompt=arguments.prompt,
            model_id=arguments.model_id,
            token_positions=(
                positions_by_label
            ),
            tokenizer_table=tokenizer_table,
            store=store,
            arguments=arguments,
        )

    finally:
        del pipeline_output
        del pipeline

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
