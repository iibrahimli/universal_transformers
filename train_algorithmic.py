"""Neural GPU for Learning Algorithms."""


import wandb
import torch
import torch.nn as nn
import utils
from loguru import logger
import numpy as np

from argparse import ArgumentParser

from algorithmic_generators import generators
from model import UniversalTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def calc_acc(outputs, targets, tgt_padding_mask):
    if outputs.shape[-1] == 1:
        outputs = np.round(outputs.detach().numpy())
    else:
        outputs = np.argmax(outputs.detach().numpy(), axis=-1)
    targets = targets.detach().numpy()
    tgt_padding_mask = tgt_padding_mask.detach().numpy()
    tp = np.all((outputs == targets) | tgt_padding_mask, axis=-1).sum()

    return tp/len(outputs)

def batch_loss_step(model, batch, loss_fn, device):
    """Compute loss for a batch"""
    source, target, src_pad_mask, tgt_pad_mask = batch
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


def train_for_a_step(model, length, batch_size, data_generator, step):
    batch = data_generator.get_batch(length, batch_size)
    model.train()
    optimizer.zero_grad()

    # Weight update
    out, tr_loss = batch_loss_step(model, batch, loss, DEVICE)
    tr_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr = scheduler.step()

    targets = batch[1]
    tgt_padding_maks = batch[3]
    acc = calc_acc(out, targets, tgt_padding_maks)

    if step % args.tr_log_interval == 0:
         wandb.log({"tr": {"loss": tr_loss.item(), "accuracy": acc}, "lr": lr}, step=step)


def infer_for_a_step(model, batch):
    model.eval()
    with torch.no_grad():
        out, eval_loss = batch_loss_step(model, batch, loss, DEVICE)

    return out, eval_loss

def single_test(model, batch):
    """Test model on test data of length l using the given session."""
    out, eval_loss = infer_for_a_step(model, batch)
    targets = batch[1]
    tgt_padding_maks = batch[3]
    acc = calc_acc(out, targets, tgt_padding_maks)
    return acc


def run_evaluation(model, length, batch_size, data_generator, test_steps, step=0):
    accuracy = []
    for step in range(test_steps):
        batch = data_generator.get_batch(length, batch_size)
        acc = single_test(model, batch)
        accuracy.append(acc)
    wandb.log({"test": {"accuracy": np.mean(accuracy)}})

    return accuracy



def train_loop(model, train_length, test_length, data_generator, batch_size, train_steps, test_steps):
    # Main training loop.
    for step in range(train_steps):
        train_for_a_step(model, train_length, batch_size, data_generator, step)
        print(step)
        # Run evaluation.
        if step > 0 and step % 10 == 0:
            accuracy = run_evaluation(model, test_length, batch_size, data_generator, test_steps, step)




if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
    parser.add_argument(
        "--train_steps",
        type=int,
        default=1000,
        help="number of training steps"
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=100,
        help="number of test steps"
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
        default=1,
        help="Log training loss every N steps"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Run validation (& log) every N steps"
    )
    parser.add_argument(
        "--train_length",
        type=int,
        default=40,
        help="length of input sequence for training"
    )
    parser.add_argument(
        "--test_length",
        type=int,
        default=400,
        help="length of input sequence for testing"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="rev, scopy, badd",
        help="current algorithmic task"
    )
    args = parser.parse_args()


    train_length = args.train_length
    test_length = args.test_length
    nclass = 33
    batch_size = args.batch_size
    train_steps = args.train_steps
    test_steps = args.test_steps
    task_list = args.task.replace(" ", "").split(",")

    for task in task_list:

        data_generator = generators[task]

        # Initialize model
        model = UniversalTransformer(
            source_vocab_size=1,#data_generator.base+1,
            target_vocab_size=35,#data_generator.base+1,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_feedforward=args.d_feedforward,
            max_seq_len=args.max_seq_len,
            max_time_step=args.max_time_step,
            halting_thresh=args.halting_thresh,
            target_input_size=1,
            embedding_method="linear"
        ).to(DEVICE)

        # Training extras
        loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(
            DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = utils.CustomLRScheduler(
            optimizer,
            d_model=args.d_model,
            warmup_steps=5000,
            lr_mul=2.
        )

        #Initialize W&B
        wandb.init(project="universal_transformer_algorithmic_task", group=f"{task}", config=args)
        wandb.watch(model, log_freq=1)

        logger.info("Using args: {")
        for k, v in wandb.config.items():
           logger.info(f"    {k}: {v}")
        logger.info("}\n")

        #start training
        train_loop(model, train_length, test_length, data_generator, batch_size, train_steps, test_steps)
