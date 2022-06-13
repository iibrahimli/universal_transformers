"""Train UT on Algorithmic Tasks"""

import os
from pathlib import Path
from itertools import cycle
from functools import partial

import wandb
import torch
import torch.nn as nn
import utils

# from loguru import logger
import numpy as np

from argparse import ArgumentParser

from algorithmic_generators import generators
from model import UniversalTransformer


DEVICE = 1
print(f"Using device: {DEVICE}")


def calc_seq_acc(outputs, targets, tgt_padding_mask):
    """Calculate accuracy for a batch"""
    if outputs.shape[-1] == 1:
        outputs = np.round(outputs.detach().numpy())
    else:
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
    targets = targets.detach().numpy()
    tgt_padding_mask = tgt_padding_mask.detach().numpy()
    tp = np.all((outputs == targets) | tgt_padding_mask, axis=-1).sum()

    return tp / len(outputs)


def calc_char_acc(outputs, targets, tgt_padding_mask):
    """Calculate accuracy for a batch"""

    if outputs.shape[-1] == 1:
        outputs = np.round(outputs.detach().numpy())
    else:
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
    valid_pos = ~tgt_padding_mask.detach().numpy()
    targets = targets.detach().numpy()
    outputs = outputs[valid_pos]
    targets = targets[valid_pos]
    tp = (outputs == targets).sum()

    return tp / len(outputs)


def batch_loss_step(model, batch, loss_fn, device, pad_val):
    """Compute loss for a batch"""
    source, target, src_pad_mask, tgt_pad_mask = batch
    shifted_target, shifted_tgt_pad_mask = utils.prepare_target(target, tgt_pad_mask, pad_val)
    source = source.to(device)
    target = target.to(device)
    shifted_target = shifted_target.to(device)
    src_pad_mask = src_pad_mask.to(device)
    tgt_pad_mask = tgt_pad_mask.to(device)
    shifted_tgt_pad_mask = shifted_tgt_pad_mask.to(device)
    out = model(
        source,
        shifted_target,
        source_padding_mask=src_pad_mask,
        target_padding_mask=shifted_tgt_pad_mask,
    )
    loss_value = loss_fn(out[tgt_pad_mask].view(-1, model.target_vocab_size), target[tgt_pad_mask].view(-1))
    return out, loss_value


def batch_loss_step_val(model, batch, loss_fn, device):
    """Compute loss for a batch"""
    source, target, src_pad_mask, tgt_pad_mask = batch
    source = source.to(device)
    src_pad_mask = src_pad_mask.to(device)
    target = target.to(device)

    out = model.generate_algorithmic(source, src_pad_mask)
    loss_value = loss_fn(out[tgt_pad_mask].flatten(0, 1), target[tgt_pad_mask].view(-1))
    return out, loss_value


def train_for_a_step(model, length, batch_size, data_generator, step, tr_log_interval, pad_val):
    batch = data_generator.get_batch(length, batch_size)
    model.train()
    optimizer.zero_grad()

    out, tr_loss = batch_loss_step(model, batch, loss, DEVICE, pad_val)
    tr_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr = scheduler.step()

    targets = batch[1]
    tgt_padding_maks = batch[3]
    seg_acc = calc_seq_acc(out, targets, tgt_padding_maks)
    char_acc = calc_char_acc(out, targets, tgt_padding_maks)

    if step % tr_log_interval == 0:
        wandb.log({"tr": {"loss": tr_loss.item(), "seg_acc": seg_acc, 'char_acc': char_acc}, "lr": lr}, step=step)


def infer_for_a_step(model, batch):
    model.eval()
    with torch.no_grad():
        out, eval_loss = batch_loss_step_val(model, batch, loss, DEVICE)
    return out, eval_loss


def run_evaluation(model, l, batch_size, data_generator, val_steps, step=0):
    """Test model on test data of length l"""
    seq_accuracy = []
    char_accuracy = []
    for step in range(val_steps):
        batch = data_generator.get_batch(l, batch_size)
        out, eval_loss = infer_for_a_step(model, batch)
        targets = batch[1]
        tgt_padding_maks = batch[3]
        seq_acc = calc_seq_acc(out, targets, tgt_padding_maks)
        char_acc = calc_char_acc(out, targets, tgt_padding_maks)
        seq_accuracy.append(seq_acc)
        char_accuracy.append(char_acc)
    wandb.log({"val": {"sequence accuracy": np.mean(seq_accuracy), "charcater accuracy": np.mean(char_acc)}})

    return seq_accuracy, char_accuracy


def train_loop(
    model,
    train_length,
    val_length,
    data_generator,
    batch_size,
    train_steps,
    val_steps,
    tr_log_interval,
    val_interval,
    pad_val,
    save_path=Path('None'),
):
    # Main training loop.
    for step in range(train_steps):
        train_for_a_step(model, train_length, batch_size, data_generator, step, tr_log_interval, pad_val)
        # Run evaluation.
        if step > 0 and step % val_interval == 0:
            seq_accuracy, char_accuracy = run_evaluation(model, val_length, batch_size, data_generator, val_steps, step)
            if save_path.name != 'None':
                torch.save(model.state_dict(), save_path/f'{step}')


if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
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
        default=400,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_time_step",
        type=int,
        default=10,
        help="Maximum time step",
    )
    parser.add_argument("--train_steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--val_steps", type=int, default=10, help="Number of validation steps")
    parser.add_argument("--val_interval", type=int, default=100, help="Run validation (& log) every N steps")
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
    parser.add_argument("--tr_log_interval", type=int, default=1, help="Log training loss every N steps")
    parser.add_argument("--train_length", type=int, default=40, help="Length of input sequence for training")
    parser.add_argument("--val_length", type=int, default=40, help="Length of input sequence for validation")
    parser.add_argument(
        "--source_vocab_size",
        type=int,
        default=1,  # for algorithmic tasks just one integer as input
        help="Vocab size of input",
    )
    parser.add_argument("--nclass", type=int, default=33, help="Number of classes (0 is padding)")
    parser.add_argument("--pad_val", type=int, default=0, help="Value used for padding")
    parser.add_argument(
        "--task",
        type=str,
        default="badd, scopy, rev",
        help="List of algorithmic tasks to be processed",  # rev: reverse input sequence, scopy: copy input sequence, badd: integer addition
    )

    parser.add_argument(
        "--save_weights",
        type=str,
        default="None",
        help="path where to save weights to",  # rev: reverse input sequence, scopy: copy input sequence, badd: integer addition
    )

    parser.add_argument(
        "--model_weights",
        type=str,
        default="None",
        help="path to pretrained model weights",  # rev: reverse input sequence, scopy: copy input sequence, badd: integer addition
    )
    args = parser.parse_args()

    train_length = args.train_length
    val_length = args.val_length
    batch_size = args.batch_size
    pad_val = args.pad_val
    train_steps = args.train_steps
    val_steps = args.val_steps
    val_interval = args.val_interval
    tr_log_interval = args.tr_log_interval
    task_list = args.task.replace(" ", "").split(",")
    model_weights = Path(args.model_weights)
    root_save_path = Path(args.save_weights)

    # Iterate over tasks
    for task in task_list:

        # Initialize Generator
        data_generator = generators[task]

        # Initialize model
        model = UniversalTransformer(
            source_vocab_size=args.source_vocab_size,
            target_vocab_size=args.nclass + 1,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_feedforward=args.d_feedforward,
            max_seq_len=args.max_seq_len,
            max_time_step=args.max_time_step,
            halting_thresh=args.halting_thresh,
            target_input_size=1,
            embedding_method="linear",
        ).to(DEVICE)
        if model_weights.name != 'None':
            model.load_state_dict(torch.load(model_weights, map_location='cuda'))


        # Training extras
        loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = utils.CustomLRScheduler(optimizer, d_model=args.d_model, warmup_steps=4000, lr_mul=0.25)

        # Initialize W&B
        wandb.init(project="universal_transformer_algorithmic_task", group=f"{task}", config=args)
        wandb.watch(model, log_freq=1)
        if root_save_path.name != 'None':
            save_path = root_save_path / f'{wandb.run.id}'
            save_path.mkdir(parents=True, exist_ok=True)

        # logger.info("Using args: {")
        # for k, v in wandb.config.items():
        #    logger.info(f"    {k}: {v}")
        # logger.info("}\n")

        # start training loop
        train_loop(
            model,
            train_length,
            val_length,
            data_generator,
            batch_size,
            train_steps,
            val_steps,
            tr_log_interval,
            val_interval,
            pad_val,
            save_path
        )
