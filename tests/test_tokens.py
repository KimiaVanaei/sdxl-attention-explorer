from types import SimpleNamespace

import torch

from sdxl_attention.tokens import (
    find_token_positions,
    format_prompt_tokens,
    tokenize_prompt,
)


class FakeTokenizer:
    model_max_length = 8
    all_special_ids = [100, 101]

    def __call__(
        self,
        prompt,
        *,
        padding,
        max_length,
        truncation,
        return_attention_mask,
        return_tensors,
    ):
        del (
            prompt,
            padding,
            max_length,
            truncation,
            return_attention_mask,
            return_tensors,
        )

        return SimpleNamespace(
            input_ids=torch.tensor(
                [[100, 10, 11, 12, 101, 101, 101, 101]]
            ),
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 0, 0, 0]]
            ),
        )

    def convert_ids_to_tokens(self, token_id):
        mapping = {
            100: "<start>",
            10: "red</w>",
            11: "cat</w>",
            12: "cat</w>",
            101: "<end>",
        }

        return mapping[token_id]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        del clean_up_tokenization_spaces

        mapping = {
            100: "",
            10: "red",
            11: "cat",
            12: "cat",
            101: "",
        }

        token_id = token_ids[0]

        if skip_special_tokens and token_id in self.all_special_ids:
            return ""

        return mapping[token_id]


def test_tokenize_prompt_removes_padding() -> None:
    tokens = tokenize_prompt(
        FakeTokenizer(),
        "a test prompt",
    )

    assert [token.position for token in tokens] == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert tokens[2].text == "cat"
    assert tokens[0].is_special
    assert tokens[-1].is_special


def test_find_token_positions_returns_repeated_tokens() -> None:
    tokens = tokenize_prompt(
        FakeTokenizer(),
        "a test prompt",
    )

    positions = find_token_positions(
        tokens,
        "cat",
    )

    assert positions == [2, 3]


def test_special_tokens_can_be_excluded() -> None:
    tokens = tokenize_prompt(
        FakeTokenizer(),
        "a test prompt",
        include_special_tokens=False,
    )

    assert [token.text for token in tokens] == [
        "red",
        "cat",
        "cat",
    ]


def test_format_prompt_tokens_creates_table() -> None:
    tokens = tokenize_prompt(
        FakeTokenizer(),
        "a test prompt",
    )

    table = format_prompt_tokens(tokens)

    assert "position" in table
    assert "token id" in table
    assert "red</w>" in table
    assert "cat" in table
