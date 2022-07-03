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


def translate_text(source: str, model: nn.Module, tokenizer, device: str = "cpu"):
    """
    Translate a text.
    """
    # add EOS token if not already present
    if not source.endswith(tokenizer.eos_token):
        source += tokenizer.eos_token
    input_ids = tokenizer(
        source,
        truncation=True,
        max_length=model.max_seq_len,
        return_tensors="pt",
    )["input_ids"]
    input_ids = input_ids.to(device)

    model.eval()

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


def get_translation_examples(model, tokenizer, device, dataset, count: int = 10):
    translation_examples = []
    for i in range(count):
        example = dataset[i]
        src_txt = example["translation"]["en"]
        tgt_txt = example["translation"]["de"]
        translated = translate_text(src_txt, model, tokenizer, device=device)
        translation_examples.append(f"S: {src_txt}\nT: {tgt_txt}\nO: {translated}")
    return translation_examples
