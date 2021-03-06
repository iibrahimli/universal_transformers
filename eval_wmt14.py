"""
Evaluate UT on the WMT14 en-de translation task.
"""

import argparse

import evaluate
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


from model import UniversalTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="model.pt",
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model dimension",
    )
    parser.add_argument(
        "--d_feedforward",
        type=int,
        default=2048,
        help="Feedforward dimension",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=100,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_time_step",
        type=int,
        default=10,
        help="Maximum time step",
    )
    parser.add_argument(
        "--halting_thresh",
        type=float,
        default=0.8,
        help="Halting threshold",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Top-k",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p",
    )
    args = parser.parse_args()

    # Load tokenizer (GPT-2 uses BPE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    test_ds = load_dataset("wmt14", "de-en", split="test")

    # Initialize model
    model = UniversalTransformer(
        source_vocab_size=tokenizer.vocab_size,
        target_vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_feedforward=args.d_feedforward,
        max_seq_len=args.max_seq_len,
        max_time_step=args.max_time_step,
        halting_thresh=args.halting_thresh,
    ).to(device)

    # load weights
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # metrics for evaluation
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    model.eval()

    sources = []
    refs = []
    preds = []

    ponders = {
        "enc": {
            "min": [],
            "mean": [],
            "max": [],
        },
        "dec": {
            "min": [],
            "mean": [],
            "max": [],
        },
    }

    for sample in tqdm(test_ds):
        source = sample["translation"]["en"]
        target = sample["translation"]["de"]

        # tokenize
        model_input = tokenizer(
            source + tokenizer.eos_token,
            max_length=args.max_seq_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # forward
        with torch.no_grad():
            generated, enc_ponder, dec_ponders = model.generate(
                model_input,
                tokenizer.eos_token_id,
                n_beams=0,
                use_sampling=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        output = tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        sources.append(source)
        refs.append([target])
        preds.append(output)

        ponders["enc"]["min"].append(enc_ponder.min().item())
        ponders["enc"]["mean"].append(enc_ponder.mean().item())
        ponders["enc"]["max"].append(enc_ponder.max().item())
        ponders["dec"]["min"].append(dec_ponders.min().item())
        enc_ponder_times.append(enc_ponder)
        dec_ponder_times.append(np.mean([p.mean().item() for p in dec_ponders]))

    bleu_score = bleu.compute(predictions=preds, references=refs)["bleu"]
    bertscore_f1 = np.mean(
        bertscore.compute(predictions=preds, references=refs, lang="de")["f1"]
    )

    # save results
    results = pd.DataFrame(
        {
            "source": sources,
            "target": [x[0] for x in refs],
            "prediction": preds,
            "enc_ponder_min": enc_ponder_min,
            "enc_ponder_mean": enc_ponder_mean,
            "enc_ponder_max": enc_ponder_max,
            "dec_ponder_min": dec_ponder_min,
            "dec_ponder_mean": dec_ponder_mean,
            "dec_ponder_max": dec_ponder_max,
        }
    )
    results.to_csv("results.csv", index=False)
    print("Saved results to results.csv")

    # save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"Samples: {len(refs)}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Top-k: {args.top_k}\n")
        f.write(f"Top-p: {args.top_p}\n")
        f.write("-------------------------\n")
        f.write(f"BLEU: {bleu_score:.4f}\n")
        f.write(f"BERT-score: {bertscore_f1:.4f}\n")
    print("Saved metrics to metrics.txt")

    # print contents of metrics.txt
    with open("metrics.txt", "r") as f:
        print(f.read())