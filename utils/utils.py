"""
Utility functions
"""

import torch
import torch.nn as nn
from torch import Tensor


def prepare_target(labels: Tensor, attention_mask: Tensor, decoder_start_token_id: int):
    """
    Prepare decoder target by shifting to the right and adding the start token.
    """

    shifted_labels = labels.new_zeros(labels.shape)
    shifted_labels[..., 1:] = labels[..., :-1].clone()
    shifted_labels[..., 0] = decoder_start_token_id

    shifted_attn_mask = attention_mask.new_zeros(attention_mask.shape)
    shifted_attn_mask[..., 1:] = attention_mask[..., :-1].clone()
    shifted_attn_mask[..., 0] = False

    return shifted_labels, shifted_attn_mask


def translate_tokens(
    input_ids: Tensor,
    model: nn.Module,
    tokenizer,
    device: str = "cpu",
    trim: bool = True,
):
    """
    Translate tokens.
    """

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    if trim:
        # remove trailing eos tokens (if any)
        for last_id in range(input_ids.shape[1] - 1, -1, -1):
            if input_ids[0, last_id] != tokenizer.eos_token_id:
                break
        last_id += 1

    with torch.no_grad():
        out = (
            model.generate(
                input_ids[0, :last_id],
                eos_token_id=tokenizer.eos_token_id,
                min_length=2,
                max_length=100,
            )
            .squeeze()
            .detach()
            .cpu()
        )

    return out


def translate_text(source: str, model: nn.Module, tokenizer, device: str = "cpu"):
    """
    Translate a text.
    """
    input_ids = tokenizer(
        source,
        truncation=True,
        max_length=model.max_seq_len,
        return_tensors="pt",
    )["input_ids"]
    input_ids = input_ids.to(device)

    with torch.no_grad():
        out = (
            model.generate(
                input_ids,
                eos_token_id=tokenizer.eos_token_id,
                min_length=2,
                max_length=100,
            )
            .squeeze()
            .detach()
            .cpu()
        )

    out = tokenizer.decode(out, skip_special_tokens=True)
    return out
