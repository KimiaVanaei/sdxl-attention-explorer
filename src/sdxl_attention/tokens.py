from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class PromptToken:
    """
    Information about one tokenizer position in a prompt.
    """

    position: int
    token_id: int
    raw_token: str
    text: str
    is_special: bool


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    include_special_tokens: bool = True,
) -> list[PromptToken]:
    """
    Tokenize a prompt and return readable token-position information.

    Only active positions are returned. Padding positions after the
    end-of-text token are excluded.
    """
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    token_ids = encoded.input_ids[0].tolist()
    attention_mask = encoded.attention_mask[0].tolist()

    special_ids = set(tokenizer.all_special_ids)
    tokens: list[PromptToken] = []

    for position, (token_id, is_active) in enumerate(
        zip(token_ids, attention_mask)
    ):
        if not is_active:
            continue

        is_special = token_id in special_ids

        if is_special and not include_special_tokens:
            continue

        raw_token = tokenizer.convert_ids_to_tokens(token_id)

        decoded_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        tokens.append(
            PromptToken(
                position=position,
                token_id=token_id,
                raw_token=raw_token,
                text=decoded_text,
                is_special=is_special,
            )
        )

    return tokens


def inspect_sdxl_prompt(
    pipeline: Any,
    prompt: str,
    *,
    include_special_tokens: bool = True,
) -> dict[str, list[PromptToken]]:
    """
    Tokenize an SDXL prompt using both text tokenizers.
    """
    if not hasattr(pipeline, "tokenizer"):
        raise TypeError(
            "The pipeline does not expose tokenizer."
        )

    if not hasattr(pipeline, "tokenizer_2"):
        raise TypeError(
            "The pipeline does not expose tokenizer_2."
        )

    return {
        "tokenizer_1": tokenize_prompt(
            pipeline.tokenizer,
            prompt,
            include_special_tokens=include_special_tokens,
        ),
        "tokenizer_2": tokenize_prompt(
            pipeline.tokenizer_2,
            prompt,
            include_special_tokens=include_special_tokens,
        ),
    }


def find_token_positions(
    tokens: list[PromptToken],
    query: str,
    *,
    match: Literal["exact", "contains"] = "exact",
    case_sensitive: bool = False,
) -> list[int]:
    """
    Find tokenizer positions matching readable token text.

    Examples
    --------
    ``find_token_positions(tokens, "cat")`` returns every position
    whose decoded text is exactly ``"cat"``.
    """
    if not query.strip():
        raise ValueError("query cannot be empty.")

    if match not in {"exact", "contains"}:
        raise ValueError(
            "match must be either 'exact' or 'contains'."
        )

    normalized_query = query.strip()

    if not case_sensitive:
        normalized_query = normalized_query.casefold()

    positions: list[int] = []

    for token in tokens:
        if token.is_special:
            continue

        candidate = token.text

        if not case_sensitive:
            candidate = candidate.casefold()

        if match == "exact":
            matched = candidate == normalized_query
        else:
            matched = normalized_query in candidate

        if matched:
            positions.append(token.position)

    return positions


def format_prompt_tokens(
    tokens: list[PromptToken],
) -> str:
    """
    Return a readable plain-text table of prompt tokens.
    """
    header = (
        f"{'position':>8} | "
        f"{'token id':>8} | "
        f"{'raw token':<24} | "
        "decoded"
    )

    separator = "-" * len(header)

    rows = [header, separator]

    for token in tokens:
        decoded = token.text or "<special>"

        rows.append(
            f"{token.position:8d} | "
            f"{token.token_id:8d} | "
            f"{token.raw_token:<24} | "
            f"{decoded}"
        )

    return "\n".join(rows)
