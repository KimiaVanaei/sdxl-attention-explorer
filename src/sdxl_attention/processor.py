from __future__ import annotations

from typing import Any

import torch

from sdxl_attention.store import AttentionStore


class RecordingCrossAttnProcessor:
    """
    Cross-attention processor that records attention probabilities.

    This processor is intended for ``attn2`` layers, where:

    - queries come from latent image features;
    - keys and values come from text embeddings.

    The processor preserves the ordinary attention output while passing
    a detached view of the probability matrix to ``AttentionStore``.
    """

    def __init__(
        self,
        store: AttentionStore,
        layer_name: str,
    ) -> None:
        self.store = store
        self.layer_name = layer_name

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply cross-attention and record its probability matrix.

        Parameters
        ----------
        attn:
            Diffusers ``Attention`` module containing the query, key,
            value, and output projections.

        hidden_states:
            Query-side latent features. Usually shaped:

                [batch, spatial_positions, query_dimension]

            Four-dimensional inputs are also supported:

                [batch, channels, height, width]

        encoder_hidden_states:
            Text-conditioning embeddings shaped:

                [batch, token_positions, cross_attention_dimension]
        """
        del args, kwargs

        if encoder_hidden_states is None:
            raise ValueError(
                "RecordingCrossAttnProcessor must only be installed "
                "on cross-attention layers. encoder_hidden_states "
                "was None."
            )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(
                hidden_states,
                temb,
            )

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            (
                batch_size,
                channels,
                height,
                width,
            ) = hidden_states.shape

            hidden_states = (
                hidden_states
                .reshape(
                    batch_size,
                    channels,
                    height * width,
                )
                .transpose(1, 2)
            )

        elif input_ndim == 3:
            batch_size = hidden_states.shape[0]

        else:
            raise ValueError(
                "hidden_states must be a 3D or 4D tensor, "
                f"but received {tuple(hidden_states.shape)}."
            )

        query_length = hidden_states.shape[1]
        key_length = encoder_hidden_states.shape[1]

        if encoder_hidden_states.shape[0] != batch_size:
            raise ValueError(
                "hidden_states and encoder_hidden_states must have "
                "the same batch size. "
                f"Received {batch_size} and "
                f"{encoder_hidden_states.shape[0]}."
            )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask,
                target_length=key_length,
                batch_size=batch_size,
            )

        if attn.group_norm is not None:
            hidden_states = (
                attn.group_norm(
                    hidden_states.transpose(1, 2)
                )
                .transpose(1, 2)
            )

        # Queries come from latent image features.
        query = attn.to_q(hidden_states)

        if attn.norm_cross is not None:
            encoder_hidden_states = (
                attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            )

        # Keys and values come from prompt embeddings.
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # SDXL base does not use Q/K normalization. Explicitly reject
        # unsupported variants instead of silently producing wrong maps.
        if (
            getattr(attn, "norm_q", None) is not None
            or getattr(attn, "norm_k", None) is not None
        ):
            raise NotImplementedError(
                "Q/K-normalized attention is not supported by this "
                "SDXL-focused processor."
            )

        # [batch, sequence, heads * head_dimension]
        #                         ->
        # [batch * heads, sequence, head_dimension]
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # softmax(scale * Q @ K^T + optional mask)
        #
        # Shape:
        # [batch * heads, query_positions, token_positions]
        attention_probs = attn.get_attention_scores(
            query=query,
            key=key,
            attention_mask=attention_mask,
        )

        expected_shape = (
            batch_size * attn.heads,
            query_length,
            key_length,
        )

        if tuple(attention_probs.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected attention-probability shape. "
                f"Expected {expected_shape}, received "
                f"{tuple(attention_probs.shape)}."
            )

        # Restore an explicit head dimension for AttentionStore:
        #
        # [batch * heads, query_positions, tokens]
        #                         ->
        # [batch, heads, query_positions, tokens]
        attention_probs_for_store = attention_probs.reshape(
            batch_size,
            attn.heads,
            query_length,
            key_length,
        )

        self.store.add(
            attention_probs=attention_probs_for_store,
            layer_name=self.layer_name,
        )

        # Apply attention probabilities to values.
        hidden_states = torch.bmm(
            attention_probs,
            value,
        )

        hidden_states = attn.batch_to_head_dim(
            hidden_states
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = (
                hidden_states
                .transpose(-1, -2)
                .reshape(
                    batch_size,
                    channels,
                    height,
                    width,
                )
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = (
            hidden_states
            / attn.rescale_output_factor
        )

        return hidden_states
