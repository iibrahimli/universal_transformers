"""
Train UT on the WMT14 de-en translation task.

TODO:
 - recheck shapes in docstrings
 - early stopping ?
 - make it faster
"""

import sys
from pathlib import Path
from itertools import cycle
from functools import partial
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


def encode(examples, max_seq_len=100):
    """Encode examples from dataset"""
    src_texts = [e["de"] for e in examples["translation"]]
    tgt_texts = [e["en"] for e in examples["translation"]]
    model_inputs = tokenizer(
        src_texts,
        max_length=max_seq_len,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        tgt_texts,
        max_length=max_seq_len,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    res = {}
    res["input_ids"] = model_inputs["input_ids"]
    res["attention_mask"] = ~model_inputs["attention_mask"].bool()

    labels, attn_mask = labels["input_ids"], labels["attention_mask"]
    res["labels"] = labels
    res["labels_attention_mask"] = ~attn_mask.bool()
    return res


def get_dataloaders(batch_size: int, val_size: int, max_seq_len: int = 100):
    """Get train, val, and test dataloaders"""

    def _get_dataloader_from_ds(ds):
        ds = ds.map(
            partial(encode, max_seq_len=max_seq_len),
            batched=True,
            batch_size=batch_size * 10,
        )
        ds = ds.with_format(type="torch")
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=device.type == "cuda"
        )
        return dl

    # TODO: dataset sizes (take)
    # streaming to avoid downloading the whole dataset
    train_ds = load_dataset("wmt14", "de-en", split="train", streaming=True)
    validation_ds = load_dataset("wmt14", "de-en", split="validation", streaming=True)
    test_ds = load_dataset("wmt14", "de-en", split="test", streaming=True).take(
        val_size
    )
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
    shifted_target, shifted_tgt_pad_mask = utils.prepare_target(
        target, tgt_pad_mask, tokenizer.eos_token_id
    )
    source = source.to(device)
    target = target.to(device)
    shifted_target = shifted_target.to(device)
    src_pad_mask = src_pad_mask.to(device)
    shifted_tgt_pad_mask = shifted_tgt_pad_mask.to(device)
    out = model(
        source,
        shifted_target,
        source_padding_mask=src_pad_mask,
        target_padding_mask=shifted_tgt_pad_mask,
    )
    loss_value = loss_fn(out.view(-1, model.target_vocab_size), target.view(-1))
    return out, loss_value


if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume training from",
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="Path to directory where model checkpoints will be saved",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="universal_transformer_wmt14",
        help="Name of the W&B project",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (None for Vaswani schedule)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=500,
        help="Number of samples in the validation set",
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
        "--tr_log_interval", type=int, default=5, help="Log training loss every N steps"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=500,
        help="Run validation (& log) every N steps",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device to use. None to use GPU if available else CPU",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"Provided device {device}, but it is not available. Exiting."
            )
            exit(1)
    logger.info(f"Using device: {device}")

    # Create checkpoints directory if it doesn't exist
    Path(args.checkpoints_path).mkdir(parents=True, exist_ok=True)

    # Load tokenizer (GPT-2 uses BPE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataloaders
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
        args.batch_size,
        val_size=args.val_size,
        max_seq_len=args.max_seq_len,
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
        n_heads=args.n_heads,
        d_feedforward=args.d_feedforward,
        max_seq_len=args.max_seq_len,
        max_time_step=args.max_time_step,
        halting_thresh=args.halting_thresh,
    ).to(device)

    # Training extras
    loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = utils.CustomLRScheduler(
        optimizer, d_model=args.d_model, warmup_steps=5000, lr_mul=2.0
    )

    # Step is incremented at the start of iteration, becomes 0
    step = -1
    wandb_run_id = None

    # Resume from checkpoint if needed
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint)
        step = checkpoint["step"]
        wandb_run_id = checkpoint["wandb_run_id"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.set_step(step)
        logger.info(f"Resumed from checkpoint {args.resume_checkpoint} (step {step})")

    # Initialize W&B
    if wandb_run_id is None:
        # Not resuming from checkpoint
        wandb.init(project=args.wandb_project, config=args)
    else:
        # Resume run
        wandb.init(
            project=args.wandb_project, id=wandb_run_id, config=args, resume="must"
        )
    wandb.watch(model, log_freq=100)

    logger.info("Using args: {")
    for k, v in wandb.config.items():
        logger.info(f"    {k}: {v}")
    logger.info("}\n")

    # Training loop
    for batch in cycle(train_dataloader):
        step += 1
        model.train()
        optimizer.zero_grad()

        # Weight update
        out, tr_loss = batch_loss_step(model, batch, loss, device)
        tr_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.lr is None:
            lr = scheduler.step()
        else:
            lr = args.lr

        if step % args.tr_log_interval == 0:
            wandb.log({"tr": {"loss": tr_loss.item()}, "lr": lr}, step=step)
        
        # save checkpoint
        if step % args.save_interval == 0:
            cp_path = Path(args.checkpoints_path) / f"latest.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "wandb_run_id": wandb.run.id,
                },
                cp_path,
            )
            logger.info(f"Saved checkpoint to {cp_path}")

        # validate & log
        if step % args.val_interval == 0:
            model.eval()
            val_losses = []
            bleu = load_metric("bleu")

            for batch in validation_dataloader:
                with torch.no_grad():
                    out, val_loss = batch_loss_step(model, batch, loss, device)
                    val_losses.append(val_loss.item())

                # compute BLEU
                source_texts = batch["translation"]["de"]
                target_texts = batch["translation"]["en"]
                for src_txt, tgt_txt in zip(source_texts, target_texts):
                    translated = utils.translate_text(
                        src_txt, model, tokenizer, device=device
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
                demo_source_txt, model, tokenizer, device=device
            )

            # log to W&B and console
            wandb.log(
                {
                    "tr": {"loss": tr_loss_value},
                    "val": {"loss": val_loss_value, "bleu": bleu_score},
                    "demo_translated": wandb.Html(demo_trans_text),
                },
                step=step,
            )
            logger.info(
                f"[{step}] tr_loss: {tr_loss_value:.4f}  val_loss: {val_loss_value:.4f}  val_bleu: {bleu_score:.4f}"
            )
            logger.info("")
            logger.info(f"DE: {demo_source_txt}")
            logger.info(f"EN: {demo_target_txt}")
            logger.info(f"output: {demo_trans_text}")
            logger.info("")
