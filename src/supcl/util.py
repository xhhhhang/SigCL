import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import datasets as hf_datasets


class TwoCropTransform:
    """Create two crops of the same image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch, logit_optimizer=False):
    if logit_optimizer:
        lr = args.logit_learning_rate
    else:
        lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer, logit_optimizer=False):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        if not logit_optimizer:
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = args.warmup_from + p * (args.warmup_to_logit - args.warmup_from)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


def create_optimizer(name, parameters, lr, momentum=0.9, weight_decay=1e-4):
    """Create optimizer based on name."""
    name = name.lower()
    if name == "sgd":
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif name == "rmsprop":
        print("Using RMSprop optimizer")
        return optim.RMSprop(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif name == "lars":
        from torchlars import LARS
        print("Using LARS optimizer")
        return LARS(
            optim.SGD(
                parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        )
    else:
        raise ValueError(f"Optimizer {name} not supported")


def set_optimizer(opt, model):
    if hasattr(model, "logit_scale") and opt.logit_learning_rate != -1:
        # Separate parameters into two groups
        logit_params = []
        logit_names = []
        other_params = []
        for name, param in model.named_parameters():
            if "logit_" in name:
                logit_names.append(name)
                logit_params.append(param)
            else:
                other_params.append(param)

        print(
            f"SigCL model detected, setting up {opt.optimizer} optimizer for main({opt.learning_rate}) "
            f"and {opt.logit_optimizer} for logit({opt.logit_learning_rate}){logit_names} parameters."
        )
        
        # Create two optimizers with different types and learning rates
        optimizers = {
            "main": create_optimizer(
                opt.optimizer,
                other_params,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                weight_decay=opt.weight_decay,
            ),
            "logit": create_optimizer(
                opt.logit_optimizer,
                logit_params,
                lr=opt.logit_learning_rate,
                momentum=opt.momentum,
                weight_decay=opt.weight_decay if opt.logit_optimizer != "sgd" else 0,
            ),
        }
        return optimizers
    else:
        # Original optimizer setup for non-SigCL models
        return create_optimizer(
            opt.optimizer,
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )


def save_model(model, optimizer, opt, epoch, save_file):
    print("==> Saving...")
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "epoch": epoch,
    }
    if opt.logit_learning_rate != -1:
        state["optimizer"] = optimizer["main"].state_dict()
        state["optimizer_logit"] = optimizer["logit"].state_dict()
    else:
        state["optimizer"] = optimizer.state_dict()

    torch.save(state, save_file)
    del state


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_linear_eval(ckpt_path, opt):
    """Run linear evaluation on a checkpoint and return the validation accuracy."""
    # Create logs directory if it doesn't exist
    log_dir = Path(opt.save_folder) / "linear_eval_logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file name based on checkpoint
    ckpt_name = Path(ckpt_path).stem
    log_file = log_dir / f"{ckpt_name}_linear_eval.log"
    
    linear_cmd = [
        "python",
        "src/supcl/main_linear.py",
        "--model",
        opt.model,
        "--dataset",
        opt.dataset,
        "--method",
        opt.method,
        "--ckpt",
        ckpt_path,
        "--batch_size",
        str(opt.linear_batch_size),  # Using standard linear eval batch size
        "--epochs",
        str(opt.linear_epochs),  # Standard number of epochs for linear eval
        "--learning_rate",
        str(opt.linear_learning_rate),  # Standard learning rate for linear eval
        "--optimizer",
        opt.linear_optimizer,  # Pass the optimizer choice
        "--disable_progress",  # Disable progress to avoid cluttering logs
    ]

    # Run the command and capture output
    try:
        result = subprocess.run(linear_cmd, capture_output=True, text=True, check=True)
        
        # Save stdout to log file
        with open(log_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        # Extract accuracy from the output
        output = result.stdout.strip()
        acc = float(output.split("best accuracy: ")[-1])
        return acc
    except subprocess.CalledProcessError as e:
        # Save error information to log file
        with open(log_file, 'w') as f:
            f.write(f"=== ERROR: Process failed with return code {e.returncode} ===\n")
            f.write("\n=== STDOUT ===\n")
            f.write(e.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(e.stderr)
            f.write(f"\n=== COMMAND THAT FAILED ===\n")
            f.write(' '.join(e.cmd))
        
        print(f"Linear evaluation failed with error code {e.returncode}")
        print(f"Error logs saved to: {log_file}")
        return None


def run_linear_eval_on_saved_checkpoints(opt, fabric):
    print("Running linear evaluation on saved checkpoints...")
    checkpoint_dir = Path(opt.save_folder)
    checkpoints = sorted(
        checkpoint_dir.glob("ckpt_epoch_*.pth"),
        key=lambda x: int(x.stem.split("_")[-1]),
        reverse=True,
    )

    # Check if last checkpoint exists and get its epoch
    last_checkpoint = checkpoint_dir / "last.pth"
    if last_checkpoint.exists():
        # Only add last checkpoint if its epoch doesn't match any existing checkpoint
        last_epoch = opt.epochs
        if not any(int(ckpt.stem.split("_")[-1]) == last_epoch for ckpt in checkpoints):
            checkpoints.append(last_checkpoint)

    # Store results to log all at once
    results = []

    # Run linear eval on each checkpoint
    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[-1]) if "last" not in str(ckpt_path) else opt.epochs

        # Run linear evaluation
        val_acc = run_linear_eval(str(ckpt_path), opt)

        if val_acc is not None:
            results.append((epoch, val_acc))
            print(f"Checkpoint {ckpt_path.name}: Linear evaluation accuracy = {val_acc:.2f}")

    # Sort results by epoch and log to wandb
    results.sort(key=lambda x: x[0])  # Sort by epoch
    for epoch, val_acc in results:
        log_data = {
            "linear_eval/val_accuracy": val_acc,
            "linear_eval/epoch": epoch,
        }
        fabric.log_dict(log_data, step=epoch)


def load_imagenet_hf(opt, transform):
    imagenet = hf_datasets.load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
    imagenet = imagenet.cast_column("image", hf_datasets.Image(mode="RGB"))

    def transform_hf(examples):
        transformed_images = [transform(image) for image in examples["image"]]
        examples["image"] = transformed_images
        return examples

    imagenet.set_transform(transform_hf)
    return imagenet
