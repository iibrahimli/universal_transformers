"""
Load a model and generate output
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import UniversalTransformer
import utils


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "input_text",
        type=str,
        help="Input text",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device to use. None to use GPU if available else CPU",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Provided device {device}, but cuda is not available. Exiting.")
            exit(1)
    print(f"Using device: {device}")

    # Load tokenizer (GPT-2 uses BPE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = UniversalTransformer(
        source_vocab_size=tokenizer.vocab_size,
        target_vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        d_feedforward=2048,
        max_seq_len=100,
        max_time_step=10,
        halting_thresh=0.8,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate output
    input_text = args.input_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    output_ids = model.generate(input_ids, eos_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids.squeeze(0))

    # Print output
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")
