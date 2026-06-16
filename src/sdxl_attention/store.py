from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch


Resolution = tuple[int, int]
ResolutionInput = int | Resolution


@dataclass(frozen=True)
class AttentionKey:
    """
    Identifies one group of compatible attention maps.

    Maps can only be averaged directly when they come from the same
    U-Net region and have the same spatial resolution.
    """

    block_group: str
    resolution: Resolution


class AttentionStore:
    """
    Memory-conscious storage for cross-attention probabilities.

    Expected input shape
    --------------------
    [effective_batch, heads, spatial_positions, tokens]

    Stored aggregate shape
    ----------------------
    [height, width, tokens]

    The store:

    1. Selects the conditional classifier-free-guidance branch.
    2. Selects one generated image from the conditional batch.
    3. Averages over attention heads.
    4. Groups maps by U-Net region and spatial resolution.
    5. Maintains running sums instead of retaining every layer output.
    """

    BLOCK_ORDER = ("down", "mid", "up", "other")

    def __init__(
        self,
        image_size: Resolution,
        *,
        use_classifier_free_guidance: bool = True,
        batch_index: int = 0,
        allowed_resolutions: Iterable[ResolutionInput] | None = None,
        allowed_block_groups: Iterable[str] | None = None,
        storage_device: str | torch.device = "cpu",
    ) -> None:
        image_height, image_width = image_size

        if image_height <= 0 or image_width <= 0:
            raise ValueError("Image dimensions must be positive.")

        if batch_index < 0:
            raise ValueError("batch_index must be non-negative.")

        self.image_size = (
            int(image_height),
            int(image_width),
        )

        self.use_classifier_free_guidance = (
            use_classifier_free_guidance
        )
        self.batch_index = batch_index
        self.storage_device = torch.device(storage_device)

        self.allowed_resolutions = self._normalize_resolution_set(
            allowed_resolutions
        )

        self.allowed_block_groups = (
            None
            if allowed_block_groups is None
            else set(allowed_block_groups)
        )

        self.reset()

    @staticmethod
    def normalize_resolution(
        resolution: ResolutionInput,
    ) -> Resolution:
        """
        Convert a square integer or an explicit tuple to (height, width).

        Examples
        --------
        16       -> (16, 16)
        (16, 24) -> (16, 24)
        """
        if isinstance(resolution, int):
            if resolution <= 0:
                raise ValueError(
                    "Resolution values must be positive."
                )

            return resolution, resolution

        if len(resolution) != 2:
            raise ValueError(
                "A resolution tuple must contain two values."
            )

        height, width = resolution

        if height <= 0 or width <= 0:
            raise ValueError(
                "Resolution values must be positive."
            )

        return int(height), int(width)

    @classmethod
    def _normalize_resolution_set(
        cls,
        resolutions: Iterable[ResolutionInput] | None,
    ) -> set[Resolution] | None:
        if resolutions is None:
            return None

        normalized = {
            cls.normalize_resolution(resolution)
            for resolution in resolutions
        }

        if not normalized:
            raise ValueError(
                "allowed_resolutions cannot be empty."
            )

        return normalized

    @staticmethod
    def infer_block_group(layer_name: str) -> str:
        """
        Infer a broad U-Net region from a Diffusers processor path.
        """
        if layer_name.startswith("down_blocks."):
            return "down"

        if layer_name.startswith("mid_block."):
            return "mid"

        if layer_name.startswith("up_blocks."):
            return "up"

        return "other"

    def infer_spatial_shape(
        self,
        spatial_positions: int,
    ) -> Resolution:
        """
        Infer a spatial grid whose product equals spatial_positions.

        For rectangular images, the factor pair whose aspect ratio is
        closest to the generated image is selected.
        """
        if spatial_positions <= 0:
            raise ValueError(
                "spatial_positions must be positive."
            )

        image_height, image_width = self.image_size
        target_ratio = image_width / image_height

        candidates: list[
            tuple[float, int, int]
        ] = []

        for divisor in range(
            1,
            math.isqrt(spatial_positions) + 1,
        ):
            if spatial_positions % divisor != 0:
                continue

            other = spatial_positions // divisor

            for height, width in {
                (divisor, other),
                (other, divisor),
            }:
                candidate_ratio = width / height

                score = abs(
                    math.log(
                        candidate_ratio / target_ratio
                    )
                )

                candidates.append(
                    (score, height, width)
                )

        if not candidates:
            raise ValueError(
                f"Could not infer a spatial grid for "
                f"{spatial_positions} positions."
            )

        _, best_height, best_width = min(candidates)

        return best_height, best_width

    @torch.no_grad()
    def add(
        self,
        attention_probs: torch.Tensor,
        layer_name: str,
    ) -> None:
        """
        Add one cross-attention probability tensor.

        Parameters
        ----------
        attention_probs:
            Tensor shaped:
                [batch, heads, spatial_positions, tokens]

        layer_name:
            Full Diffusers attention-processor path.
        """
        if attention_probs.ndim != 4:
            raise ValueError(
                "attention_probs must have shape "
                "[batch, heads, spatial_positions, tokens], "
                f"but received {tuple(attention_probs.shape)}."
            )

        block_group = self.infer_block_group(
            layer_name
        )

        if (
            self.allowed_block_groups is not None
            and block_group not in self.allowed_block_groups
        ):
            return

        (
            effective_batch_size,
            number_of_heads,
            spatial_positions,
            number_of_tokens,
        ) = attention_probs.shape

        if number_of_heads <= 0:
            raise ValueError(
                "The number of attention heads must be positive."
            )

        if number_of_tokens <= 0:
            raise ValueError(
                "The number of token positions must be positive."
            )

        resolution = self.infer_spatial_shape(
            spatial_positions
        )

        if (
            self.allowed_resolutions is not None
            and resolution not in self.allowed_resolutions
        ):
            return

        if self.use_classifier_free_guidance:
            if effective_batch_size % 2 != 0:
                raise ValueError(
                    "Classifier-free guidance requires an even "
                    "effective attention batch size."
                )

            conditional_start = (
                effective_batch_size // 2
            )

            attention_probs = attention_probs[
                conditional_start:
            ]

        conditional_batch_size = (
            attention_probs.shape[0]
        )

        if self.batch_index >= conditional_batch_size:
            raise IndexError(
                f"batch_index={self.batch_index} is invalid for "
                f"a conditional batch of size "
                f"{conditional_batch_size}."
            )

        # [heads, spatial_positions, tokens]
        selected_attention = attention_probs[
            self.batch_index
        ]

        # Average heads before storing:
        #
        # [heads, spatial_positions, tokens]
        #                    ->
        # [spatial_positions, tokens]
        mean_attention = (
            selected_attention
            .detach()
            .to(dtype=torch.float32)
            .mean(dim=0)
        )

        # [spatial_positions, tokens]
        #                    ->
        # [height, width, tokens]
        mean_attention = mean_attention.reshape(
            resolution[0],
            resolution[1],
            number_of_tokens,
        )

        # Stored aggregates do not need to remain on the GPU.
        mean_attention = mean_attention.to(
            device=self.storage_device
        )

        key = AttentionKey(
            block_group=block_group,
            resolution=resolution,
        )

        if key not in self._attention_sums:
            self._attention_sums[key] = (
                mean_attention.clone()
            )

            self._attention_counts[key] = 1
        else:
            existing = self._attention_sums[key]

            if existing.shape != mean_attention.shape:
                raise ValueError(
                    "The incoming map does not match the stored "
                    "attention shape. "
                    f"Stored: {tuple(existing.shape)}, "
                    f"incoming: {tuple(mean_attention.shape)}."
                )

            existing.add_(mean_attention)
            self._attention_counts[key] += 1

        self.layer_call_counts[layer_name] += 1

    def get_average(
        self,
        block_group: str,
        resolution: ResolutionInput,
    ) -> torch.Tensor:
        """
        Return the average stored map on the CPU.

        Returned shape:
            [height, width, tokens]
        """
        key = AttentionKey(
            block_group=block_group,
            resolution=self.normalize_resolution(
                resolution
            ),
        )

        if key not in self._attention_sums:
            raise KeyError(
                f"No attention map was stored for {key}. "
                f"Available keys: {self.keys()}"
            )

        average = (
            self._attention_sums[key]
            / float(self._attention_counts[key])
        )

        return average.detach().cpu()

    def keys(self) -> list[AttentionKey]:
        """
        Return all stored keys in a predictable order.
        """
        block_rank = {
            name: index
            for index, name in enumerate(
                self.BLOCK_ORDER
            )
        }

        return sorted(
            self._attention_sums,
            key=lambda key: (
                block_rank.get(
                    key.block_group,
                    len(block_rank),
                ),
                key.resolution[0]
                * key.resolution[1],
                key.resolution,
            ),
        )

    def block_groups(self) -> list[str]:
        """
        Return the U-Net groups currently present in the store.
        """
        present = {
            key.block_group
            for key in self._attention_sums
        }

        return [
            block_group
            for block_group in self.BLOCK_ORDER
            if block_group in present
        ]

    def resolutions(
        self,
        block_group: str | None = None,
    ) -> list[Resolution]:
        """
        Return stored resolutions, optionally restricted to one group.
        """
        resolutions = {
            key.resolution
            for key in self._attention_sums
            if (
                block_group is None
                or key.block_group == block_group
            )
        }

        return sorted(
            resolutions,
            key=lambda resolution: (
                resolution[0] * resolution[1],
                resolution,
            ),
        )

    def summary(self) -> str:
        if not self._attention_sums:
            return "AttentionStore is empty."

        lines = ["Stored cross-attention maps:"]

        for block_group in self.block_groups():
            lines.append(f"  {block_group}:")

            for resolution in self.resolutions(
                block_group
            ):
                key = AttentionKey(
                    block_group,
                    resolution,
                )

                count = self._attention_counts[key]

                lines.append(
                    f"    {resolution[0]}x"
                    f"{resolution[1]}: "
                    f"{count} recorded call(s)"
                )

        lines.append(
            "  Unique recorded layers: "
            f"{len(self.layer_call_counts)}"
        )

        return "\n".join(lines)

    def reset(self) -> None:
        """
        Delete all stored attention aggregates and counters.
        """
        self._attention_sums: dict[
            AttentionKey,
            torch.Tensor,
        ] = {}

        self._attention_counts: Counter[
            AttentionKey
        ] = Counter()

        self.layer_call_counts: Counter[
            str
        ] = Counter()
