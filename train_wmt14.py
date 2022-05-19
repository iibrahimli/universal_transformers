"""
Train UT on the WMT14 de-en translation task.

TODO:
 - custom LR schedule
 - recheck shapes in docstrings
 - halting_layer no gradient
 - check what else has no gradient
 - make it faster
 - train & val log interval
 - take config from cmd args

"""

import sys
from itertools import cycle
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric

from model import UniversalTransformer
import utils


# configure logger
logger.remove()
log_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}"
logger.add(sys.stderr, format=log_fmt)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MAX_SEQ_LENGTH = 50


def encode(examples):
    """Encode examples from dataset"""
    src_texts = [e["de"] for e in examples["translation"]]
    tgt_texts = [e["en"] for e in examples["translation"]]
    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        tgt_texts,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    res = {}
    res["input_ids"] = model_inputs["input_ids"]
    res["attention_mask"] = ~model_inputs["attention_mask"].bool()

    labels, attn_mask = utils.prepare_target(
        labels["input_ids"], labels["attention_mask"], tokenizer.pad_token_id
    )
    res["labels"] = labels
    res["labels_attention_mask"] = ~attn_mask.bool()
    return res


def get_dataloaders(batch_size, map_batch_size: int = 500):
    """Get train, val, and test dataloaders"""

    def _get_dataloader_from_ds(ds):
        # TODO: batchsize
        # ds = ds.map(encode, batched=True, batch_size=map_batch_size, remove_columns=["translation"])
        ds = ds.map(encode, batched=True, batch_size=map_batch_size)
        ds = ds.with_format(type="torch")
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        return dl

    # TODO: dataset sizes (take)
    # streaming to avoid downloading the whole dataset
    train_ds = load_dataset("wmt14", "de-en", split="train", streaming=True)
    validation_ds = load_dataset(
        "wmt14", "de-en", split="validation", streaming=True
    ).take(100)
    test_ds = load_dataset("wmt14", "de-en", split="test", streaming=True)

    train_dl = _get_dataloader_from_ds(train_ds)
    validation_dl = _get_dataloader_from_ds(validation_ds)
    test_dl = _get_dataloader_from_ds(test_ds)

    return train_dl, validation_dl, test_dl


def unpack_batch(batch):
    source = batch["input_ids"]
    target = batch["labels"]
    src_pad_mask = batch["attention_mask"]
    tgt_pad_mask = batch["labels_attention_mask"]
    return source, target, src_pad_mask, tgt_pad_mask


def batch_loss_step(model, batch, loss_fn, device):
    """Compute loss for a batch"""
    source, target, src_pad_mask, tgt_pad_mask = unpack_batch(batch)
    source = source.to(device)
    target = target.to(device)
    src_pad_mask = src_pad_mask.to(device)
    tgt_pad_mask = tgt_pad_mask.to(device)
    out = model(
        source,
        target,
        source_padding_mask=src_pad_mask,
        target_padding_mask=tgt_pad_mask,
    )
    loss_value = loss_fn(out.view(-1, model.target_vocab_size), target.view(-1))
    return out, loss_value


if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
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
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing",
    )
    parser.add_argument(
        "--tr_log_interval",
        type=int,
        default=5,
        help="Log training loss every N steps"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=200,
        help="Run validation (& log) every N steps"
    )
    args = parser.parse_args(args=[])

    logger.info(f"Using args: {args}")

    # Load tokenizer (GPT-2 uses BPE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataloaders
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
        args.batch_size, map_batch_size=10 * args.batch_size
    )

    # Demo sentence to try to translate throughout training
    demo_sample = next(iter(validation_dataloader))
    demo_source_txt = demo_sample["translation"]["de"][2]
    demo_target_txt = demo_sample["translation"]["en"][2]
    demo_source_txt, demo_target_txt

    # Initialize model
    model = UniversalTransformer(
        source_vocab_size=tokenizer.vocab_size,
        target_vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_head=args.n_heads,
        d_feedforward=args.d_feedforward,
        max_seq_len=args.max_seq_len,
        max_time_step=args.max_time_step,
        halting_thresh=args.halting_thresh,
    ).to(DEVICE)

    # Training extras
    loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = utils.CustomLRScheduler(
        optimizer,
        d_model=args.d_model,
        warmup_steps=5000,
    )

    # Initialize W&B
    wandb.init(project="universal_transformer_wmt14_test", config=args)
    wandb.watch(model, log_freq=100)

    # Training loop
    for i, batch in cycle(enumerate(train_dataloader)):
        model.train()
        optimizer.zero_grad()

        # Weight update
        out, tr_loss = batch_loss_step(model, batch, loss, DEVICE)
        tr_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr = scheduler.step()

        if i % args.tr_log_interval == 0:
            wandb.log({"tr": {"loss": tr_loss.item()}, "lr": lr})

        # validate & log
        if i % args.val_interval == 0:
            model.eval()
            val_losses = []
            bleu = load_metric("bleu")

            for batch in validation_dataloader:
                with torch.no_grad():
                    out, val_loss = batch_loss_step(model, batch, loss, DEVICE)
                    val_losses.append(val_loss.item())

                # compute BLEU
                source_texts = batch["translation"]["de"]
                target_texts = batch["translation"]["en"]
                for src_txt, tgt_txt in zip(source_texts, target_texts):
                    translated = utils.translate_text(
                        src_txt, model, tokenizer, device=DEVICE
                    )
                    if len(translated) == 0:
                        # to prevent division by zero in BLEU with empty string
                        translated = "0"
                    bleu.add(
                        predictions=translated.split(), references=[tgt_txt.split()]
                    )

            tr_loss_value = tr_loss.item()
            val_loss_value = torch.mean(torch.tensor(val_losses)).item()
            bleu_score = bleu.compute()["bleu"]
            demo_trans_text = utils.translate_text(
                demo_source_txt, model, tokenizer, device=DEVICE
            )

            # log to W&B and console
            wandb.log(
                {
                    "tr": {"loss": tr_loss_value},
                    "val": {"loss": val_loss_value, "bleu": bleu_score},
                    "demo_translated": wandb.Html(demo_trans_text),
                },
                step=i,
            )
            logger.info(
                f"[{i}] tr_loss: {tr_loss_value:.4f}  val_loss: {val_loss_value:.4f}  val_bleu: {bleu_score:.4f}"
            )
            # logger.info(f"DE: {demo_source_txt}")
            # logger.info(f"EN: {demo_target_txt}")
            logger.info(f"output: {demo_trans_text}")
            logger.info("")
