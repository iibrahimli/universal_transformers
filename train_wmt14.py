"""
Train UT on the WMT14 de-en translation task.
"""

import os
from pathlib import Path
from itertools import cycle
from functools import partial
import argparse

import wandb
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from model import UniversalTransformer
import utils


def tokenize(examples, max_seq_len=100):
    """Tokenize examples from dataset"""
    src = [ex["en"] + tokenizer.eos_token for ex in examples["translation"]]
    tgt = [ex["de"] + tokenizer.eos_token for ex in examples["translation"]]
    model_inputs = tokenizer(src, max_length=max_seq_len, truncation=True)
    labels = tokenizer(tgt, max_length=max_seq_len, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_dataloaders(
    train_ds, val_ds, batch_size: int, max_seq_len: int, local_rank: int
):
    """Get train and val dataloaders"""

    def _get_dataloader_from_ds(ds, dist=False):
        # wait for the main process to do mapping
        if local_rank != 0:
            torch.distributed.barrier()

        ds = ds.map(
            partial(tokenize, max_seq_len=max_seq_len),
            batched=True,
            remove_columns=ds.column_names,
        )

        # load results from main process
        if local_rank == 0:
            torch.distributed.barrier()

        sampler = DistributedSampler(ds) if dist else None
        data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt")
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=device.type == "cuda",
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=8,
        )
        return dl

    train_dl = _get_dataloader_from_ds(train_ds, dist=True)
    validation_dl = _get_dataloader_from_ds(val_ds)

    return train_dl, validation_dl


def unpack_batch(batch):
    source = batch["input_ids"]
    target = batch["labels"]
    src_pad_mask = ~batch["attention_mask"].bool()
    # extract target padding mask where it is -100
    tgt_pad_mask = target == -100
    # replace -100 in target with 0
    target[tgt_pad_mask] = 0
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
    loss_value = loss_fn(out.view(-1, out.shape[-1]), target.view(-1))

    # delete tensors to free memory
    del (
        source,
        target,
        shifted_target,
        src_pad_mask,
        tgt_pad_mask,
        shifted_tgt_pad_mask,
    )

    return out, loss_value


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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

    torch.distributed.init_process_group(backend="nccl")

    # get local rank
    local_rank = int(os.environ["LOCAL_RANK"])
    print("Started process with local_rank:", local_rank)

    # get logger
    L = utils.Logger(local_rank)

    if args.device is None:
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            L.log(f"Provided device {device}, but it is not available. Exiting.")
            exit(1)
    L.log(f"Using device: {device}")

    # Create checkpoints directory if it doesn't exist
    if local_rank == 0:
        Path(args.checkpoints_path).mkdir(parents=True, exist_ok=True)

    # Load tokenizer (GPT-2 uses BPE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_ds = load_dataset("wmt14", "de-en", split="train")
    validation_ds = load_dataset(
        "wmt14", "de-en", split=f"validation[:{args.val_size}]"
    )
    if local_rank == 0:
        L.log(
            f"Dataset loaded. Train size: {len(train_ds)}, Validation size: {len(validation_ds)}"
        )

    # Prepare dataloaders
    train_dataloader, validation_dataloader = get_dataloaders(
        train_ds,
        validation_ds,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        local_rank=local_rank,
    )

    # Demo sentence to try to translate throughout training
    demo_sample = validation_ds[2]
    demo_source_txt = demo_sample["translation"]["en"]
    demo_target_txt = demo_sample["translation"]["de"]

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    scheduler = utils.CustomLRScheduler(
        optimizer, d_model=args.d_model, warmup_steps=5000, lr_mul=1.0
    )

    # DDP
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # Step is incremented at the start of iteration, becomes 0
    step = -1
    wandb_run_id = None

    # Resume from checkpoint if needed
    if args.resume_checkpoint is not None:
        map_location = {"cuda:0": f"cuda:{local_rank}"}
        checkpoint = torch.load(args.resume_checkpoint)
        step = checkpoint["step"]
        wandb_run_id = checkpoint["wandb_run_id"]
        ddp_model.load_state_dict(
            checkpoint["model_state_dict"], map_location=map_location
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.set_step(step)
        L.log(f"Resumed from checkpoint {args.resume_checkpoint} (step {step})")

    # Initialize W&B
    os.environ["WANDB_START_METHOD"] = "thread"
    if local_rank == 0:
        wandb_entity = "universal-transformer"
        if wandb_run_id is None:
            # Not resuming from checkpoint
            wandb.init(project=args.wandb_project, entity=wandb_entity, config=args)
        else:
            # Resume run
            wandb.init(
                project=args.wandb_project,
                entity=wandb_entity,
                id=wandb_run_id,
                config=args,
                resume="must",
            )
        wandb.watch(ddp_model, log_freq=100)

    if local_rank == 0:
        L.log("Using args: {")
        for k, v in wandb.config.items():
            L.log(f"    {k}: {v}")
        L.log("}\n")

    # Training loop
    for batch in cycle(train_dataloader):
        step += 1
        ddp_model.train()
        optimizer.zero_grad()

        # Weight update
        out, tr_loss = batch_loss_step(ddp_model, batch, loss, device)
        tr_loss.backward()
        nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
        optimizer.step()
        lr = scheduler.step()

        # Logging, validation, saving checkpoints
        if local_rank == 0:

            # log training loss
            if step % args.tr_log_interval == 0:
                wandb.log({"tr": {"loss": tr_loss.item()}, "lr": lr}, step=step)
                L.log(f"[{step}] tr_loss: {tr_loss.item():.4f}")

            # save checkpoint
            if step % args.save_interval == 0:
                cp_path = Path(args.checkpoints_path) / f"latest.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "wandb_run_id": wandb.run.id,
                        "config": vars(args),
                    },
                    cp_path,
                )
                L.log(f"Saved checkpoint to {cp_path}")

            # validate & log
            if step % args.val_interval == 0:
                model.eval()
                val_losses = []
                bleu = load_metric("bleu")

                # BLEU
                for i_ex in range(10):
                    example = validation_ds[i_ex]
                    src_txt = example["translation"]["de"]
                    tgt_txt = example["translation"]["en"]
                    translated = utils.translate_text(
                        src_txt, model, tokenizer, device=device
                    )
                    if len(translated) == 0:
                        # to prevent division by zero in BLEU with empty string
                        translated = "0"
                    bleu.add(
                        predictions=translated.split(), references=[tgt_txt.split()]
                    )
                bleu_score = bleu.compute()["bleu"]

                # validation loss
                with torch.no_grad():
                    for batch in validation_dataloader:
                        out, val_loss = batch_loss_step(model, batch, loss, device)
                        val_losses.append(val_loss.item())
                val_loss_value = torch.mean(torch.tensor(val_losses)).item()

                # translate demo text
                demo_trans_text = utils.translate_text(
                    demo_source_txt, model, tokenizer, device=device
                )

                # log to W&B and console
                wandb.log(
                    {
                        "val": {"loss": val_loss_value, "bleu": bleu_score},
                        "demo_translated": wandb.Html(demo_trans_text),
                    },
                    step=step,
                )
                L.log(
                    f"[{step}] val_loss: {val_loss_value:.4f}  val_bleu: {bleu_score:.4f}"
                )
                L.log("")
                L.log(f"SRC: {demo_source_txt}")
                L.log(f"TGT: {demo_target_txt}")
                L.log(f"OUTPUT: {demo_trans_text}")
                L.log("")
