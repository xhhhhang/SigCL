import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rootutils
import torch
import torch.backends.cudnn as cudnn
import torch.cuda
from dotenv import load_dotenv
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from wandb.integration.lightning.fabric import WandbLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from networks.resnet_big import SigCLResNet, SupConResNet
from util import (
    AverageMeter,
    TwoCropTransform,
    adjust_learning_rate,
    load_imagenet_hf,
    run_linear_eval,
    run_linear_eval_on_saved_checkpoints,
    save_model,
    seed_everything,
    set_optimizer,
    warmup_learning_rate,
)

from losses import SupConLoss
from src.losses.loss import (
    ExpAverage,
    FocalAverage,
    FocalBase,
    SigCLossAverageV2,
    SigCLossBase,
    SigCLossNegWeight,
    SigCLossRatio,
)


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=9, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="700,800,900",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    # model dataset
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "imagenet", "path"],
        help="dataset",
    )
    parser.add_argument("--mean", type=str, help="mean of dataset in path in form of str tuple")
    parser.add_argument("--std", type=str, help="std of dataset in path in form of str tuple")
    parser.add_argument("--data_folder", type=str, default=None, help="path to custom dataset")
    parser.add_argument("--size", type=int, default=32, help="parameter for RandomResizedCrop")

    # method
    parser.add_argument(
        "--method",
        type=str,
        default="SupCon",
        help="choose method",
    )

    # temperature
    parser.add_argument("--temp", type=float, default=0.07, help="temperature for loss function")

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--syncBN", action="store_true", help="using synchronized batch normalization"
    )
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    parser.add_argument("--trial", type=str, default="0", help="id for recording multiple runs")
    parser.add_argument(
        "--use_sigcl",
        action="store_true",
        help="use SigCL loss instead of default contrastive loss",
    )
    parser.add_argument(
        "--neg_weight_step", type=float, default=1.02, help="step size for negative weight"
    )
    parser.add_argument("--max_neg_weight", type=int, default=16, help="maximum negative weight")
    parser.add_argument("--neg_weight", type=float, default=1, help="negative weight")
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument("--log_tensorboard", action="store_true", help="log to tensorboard")
    parser.add_argument(
        "--init_logit_scale", type=float, default=np.log(10), help="initial logit scale"
    )
    parser.add_argument("--init_logit_bias", type=float, default=0, help="initial logit bias")
    parser.add_argument(
        "--overfit_batch", action="store_true", help="train on a single batch for all epochs"
    )
    parser.add_argument("--bidir", action="store_true", help="use bidirectional exchange")
    parser.add_argument("--compile", action="store_true", help="use torch.compile")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--logit_learning_rate", type=float, default=-1, help="learning rate for logit parameters"
    )
    parser.add_argument(
        "--linear_epochs", type=int, default=25, help="number of epochs for linear eval"
    )
    parser.add_argument(
        "--skip_train", action="store_true", help="skip training and only run linear eval"
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma for focal loss")
    parser.add_argument("--normalize_focal", action="store_true", help="normalize focal loss")
    parser.add_argument("--cosine_logit", action="store_true", help="use cosine annealing for logit scale and bias")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "lars", "rmsprop"],
        help="optimizer for main network parameters",
    )
    parser.add_argument(
        "--logit_optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "lars", "rmsprop"],
        help="optimizer for logit parameters",
    )
    parser.add_argument(
        "--linear_optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "lars", "rmsprop"],
        help="optimizer for linear eval",
    )
    parser.add_argument(
        "--linear_learning_rate", type=float, default=5, help="learning rate for linear eval"
    )
    parser.add_argument(
        "--linear_batch_size", type=int, default=256, help="batch size for linear eval"
    )
    parser.add_argument(
        "--gather_loss", action="store_true", help="gather positive and negative count and loss across GPUs"
    )
    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == "path":
        assert opt.data_folder is not None and opt.mean is not None and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = "./datasets/"
    opt.model_path = f"./save/SupCon/{opt.dataset}_models"
    opt.tb_path = f"./save/SupCon/{opt.dataset}_tensorboard"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # Shorten method name (base/Avg/Sup), model name (just layer number), dataset name
    method = opt.method.replace('SigCLBaseAvg', 'Avg').replace('SigCLBase', 'Dec').replace('SupCon', 'Sup')
    model = opt.model.replace('resnet', 'rs')
    dataset = opt.dataset.replace('cifar', 'cf')
    opt.model_name = "{}_{}_{}_{}_{}_lr_{}_wd_{}_bs_{}*{}_{}".format(
        method,
        dataset, 
        model,
        opt.optimizer[:3],
        opt.linear_optimizer[:3],
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        torch.cuda.device_count(),
        opt.trial,
    )

    if not opt.cosine:
        opt.model_name = f"{opt.model_name}_nc"
    if opt.method.startswith("SigCLBase"):
        opt.model_name = f"{opt.model_name}_{int(opt.neg_weight)}"

    if opt.gather_loss:
        opt.model_name = f"{opt.model_name}_gather"

    if opt.cosine_logit:
        opt.model_name = f"{opt.model_name}_cl"

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if not opt.warm:
        opt.model_name = f"{opt.model_name}_nw"

    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
            if opt.logit_learning_rate != -1:
                eta_min_logit = opt.logit_learning_rate * (opt.lr_decay_rate**3)
                opt.warmup_to_logit = (
                    eta_min_logit
                    + (opt.logit_learning_rate - eta_min_logit)
                    * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                    / 2
                )
        else:
            opt.warmup_to = opt.learning_rate
            if opt.logit_learning_rate != -1:
                opt.warmup_to_logit = opt.logit_learning_rate

    # Load environment variables from .env file
    load_dotenv()

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt, fabric):
    # construct data loader
    if opt.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == "path":
        mean = float(opt.mean)
        std = float(opt.std)
    else:
        raise ValueError(f"dataset not supported: {opt.dataset}")
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Ensure only local rank 0 downloads the dataset
    if fabric.local_rank == 0:
        if opt.dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True
            )
        elif opt.dataset == "cifar100":
            train_dataset = datasets.CIFAR100(
                root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True
            )
        elif opt.dataset == "imagenet":
            train_dataset = load_imagenet_hf(opt, TwoCropTransform(train_transform))["train"]
        elif opt.dataset == "path":
            train_dataset = datasets.ImageFolder(
                root=opt.data_folder, transform=TwoCropTransform(train_transform)
            )

    # Make all processes wait for rank 0 to finish downloading
    if fabric.world_size > 1:
        fabric.barrier()

    # Now all processes can load the dataset without downloading
    if fabric.local_rank != 0:
        if opt.dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                root=opt.data_folder, transform=TwoCropTransform(train_transform), download=False
            )
        elif opt.dataset == "cifar100":
            train_dataset = datasets.CIFAR100(
                root=opt.data_folder, transform=TwoCropTransform(train_transform), download=False
            )
        elif opt.dataset == "imagenet":
            train_dataset = load_imagenet_hf(opt, TwoCropTransform(train_transform))["train"]
        elif opt.dataset == "path":
            train_dataset = datasets.ImageFolder(
                root=opt.data_folder, transform=TwoCropTransform(train_transform)
            )

    # # Make all processes start the rest operations at the same time
    # if fabric.world_size > 1:
    #     fabric.barrier()

    if opt.overfit_batch:
        # Create a subset with only one batch
        subset_indices = list(range(opt.batch_size))
        train_dataset = Subset(train_dataset, subset_indices)

    train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None and not opt.overfit_batch),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
    )

    return train_loader


def set_model(opt, fabric):
    if opt.method == "SupCon":
        model = SupConResNet(name=opt.model)
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == "SigCL":
        model = SigCLResNet(
            name=opt.model,
            init_logit_scale=opt.init_logit_scale,
            init_logit_bias=opt.init_logit_bias,
        )
        criterion = SigCLossNegWeight(
            max_neg_weight=opt.max_neg_weight,
            neg_weight_step=opt.neg_weight_step,
            fabric=fabric,
        )
    elif opt.method.startswith("SigCLBase"):
        model = SigCLResNet(
            name=opt.model,
            init_logit_scale=opt.init_logit_scale,
            init_logit_bias=opt.init_logit_bias,
        )
        if opt.method == "SigCLBaseRatio":
            print(f"Using SigCLBaseRatio: {opt.neg_weight}")
            criterion_class = SigCLossRatio
        elif opt.method == "SigCLBaseAvgV2" or opt.method == "SigCLBaseAvg":
            print(f"Using SigCLBaseAvg: {opt.neg_weight}")
            criterion_class = SigCLossAverageV2
        elif opt.method.startswith("SigCLBaseExp"):
            print(f"Using SigCLBaseExp: {opt.neg_weight}")
            criterion_class = ExpAverage
        else:
            print(f"Using SigCLBase: {opt.neg_weight}")
            criterion_class = SigCLossBase

        criterion = criterion_class(neg_weight=opt.neg_weight, fabric=fabric, bidir=opt.bidir)

    elif opt.method.startswith("Focal"):
        model = SigCLResNet(
            name=opt.model,
            init_logit_scale=opt.init_logit_scale,
            init_logit_bias=opt.init_logit_bias,
        )
        if opt.method == "FocalBase":
            print(f"Using FocalBase: {opt.neg_weight}")
            criterion_class = FocalBase
        elif opt.method == "FocalAverage":
            print(f"Using FocalAverage: {opt.neg_weight}")
            criterion_class = FocalAverage
        else:
            raise ValueError(f"Focal method not supported: {opt.method}")

        criterion = criterion_class(
            gamma=opt.gamma,
            normalize=opt.normalize_focal,
            neg_weight=opt.neg_weight,
            fabric=fabric,
            bidir=opt.bidir,
        )

    # Compile the model
    if opt.compile:
        model = torch.compile(model)

    return model, criterion


def train(fabric, train_loader, model, criterion, optimizer, epoch, opt):
    """One epoch training."""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, examples in enumerate(train_loader):
        if opt.dataset == "imagenet":
            images, labels = examples["image"], examples["label"]
        else:
            images, labels = examples
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.shape[0]

        # warm-up learning rate
        if isinstance(optimizer, dict):
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer["main"])
            warmup_learning_rate(
                opt, epoch, idx, len(train_loader), optimizer["logit"], logit_optimizer=True
            )
        else:
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.method.startswith("SigCL") or opt.method.startswith("Focal"):
            model_out = model(images)

            features = model_out["features"]
            logit_scale = model_out["logit_scale"]
            logit_bias = model_out["logit_bias"]

            labels = labels.repeat(2)  # Repeat labels to match the dimension [2*bsz]
            loss = criterion(
                first_features=features,
                second_features=features,
                first_label=labels,
                second_label=labels,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                mask_diagonal=True,
            )
            if opt.method == "SigCL":
                criterion.step_neg_weight()
        else:
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == "SupCon":
                loss = criterion(features, labels, fabric=fabric)
            elif opt.method == "SimCLR":
                loss = criterion(features, fabric=fabric)
            else:
                raise ValueError(f"contrastive method not supported: {opt.method}")

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        fabric.backward(loss)
        if isinstance(optimizer, dict):
            optimizer["main"].step()
            optimizer["logit"].step()
            optimizer["main"].zero_grad()
            optimizer["logit"].zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if fabric.is_global_zero and (idx + 1) % opt.print_freq == 0:
            if opt.method.startswith("SigCLBase") or opt.method.startswith("Focal"):
                log_str = (
                    "Base {neg_weight}\t"
                    "Train: [{0}][{1}/{2}]\t"
                    "logit_scale {logit_scale:.3f}\t"
                    "logit_bias {logit_bias:.3f}\t"
                    "loss {loss.val:.5f} ({loss.avg:.5f})"
                )
                print(
                    log_str.format(
                        epoch,
                        idx + 1,
                        len(train_loader),
                        logit_scale=logit_scale.item(),
                        logit_bias=logit_bias.item(),
                        loss=losses,
                        neg_weight=opt.neg_weight
                        if opt.method.startswith("SigCLBase") or opt.method.startswith("Focal")
                        else criterion.neg_weight,
                    )
                )
            else:
                print(
                    "{method}\t"
                    "Train: [{0}][{1}/{2}]\t"
                    "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "loss {loss.val:.5f} ({loss.avg:.5f})".format(
                        epoch,
                        idx + 1,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        method=opt.method,
                    )
                )
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()
    seed_everything(opt.seed)
    # torch.set_float32_matmul_precision("high")

    # Initialize logger
    loggers = []
    if opt.log_wandb:
        loggers.append(
            WandbLogger(
                project=os.getenv("WANDB_PROJECT"),
                name=opt.model_name,
                config=vars(opt),
                entity=os.getenv("WANDB_ENTITY"),
            )
        )
    if opt.log_tensorboard:
        loggers.append(TensorBoardLogger(opt.tb_path, name=opt.model_name))

    # default csv logger
    csv_save_path = os.path.join("./save/SupCon", opt.dataset + "_csv_logs")
    loggers.append(CSVLogger(csv_save_path, name=opt.model_name))

    # Initialize Fabric with logger
    fabric = Fabric(accelerator="auto", devices="auto", precision="16-mixed", loggers=loggers)
    fabric.launch()

    # build data loader
    train_loader = set_loader(opt, fabric)

    # neg_weight as real distribution
    if opt.method.startswith("SigCLBase") and opt.neg_weight == -1:
        pass

    # build model and criterion
    model, criterion = set_model(opt, fabric)
    # model = torch.compile(model)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # Setup model, optimizer, and dataloader with Fabric
    if isinstance(optimizer, dict):
        model, optimizer_main, optimizer_logit = fabric.setup(
            model, optimizer["main"], optimizer["logit"]
        )
        optimizer = {"main": optimizer_main, "logit": optimizer_logit}
    else:
        model, optimizer = fabric.setup(model, optimizer)

    train_loader = fabric.setup_dataloaders(train_loader)

    if opt.skip_train:
        # After training loop, run linear evaluation on saved checkpoints
        if fabric.is_global_zero:
            run_linear_eval_on_saved_checkpoints(opt, fabric)
        return

    # training routine
    for epoch in range(1, opt.epochs + 1):
        if isinstance(optimizer, dict):
            adjust_learning_rate(opt, optimizer["main"], epoch)
            if opt.cosine_logit:
                adjust_learning_rate(opt, optimizer["logit"], epoch, logit_optimizer=True)
        else:
            adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(fabric, train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        if fabric.is_global_zero:
            print(f"epoch {epoch}, total time {time2 - time1:.2f}")

        # Log metrics using Fabric's logger
        log_data = {
            "epoch": epoch,
            "loss": loss,
        }
        if isinstance(optimizer, dict):
            log_data.update(
                {
                    "learning_rate": optimizer["main"].param_groups[0]["lr"],
                    "learning_rate_logit": optimizer["logit"].param_groups[0]["lr"],
                }
            )
        else:
            log_data.update({"learning_rate": optimizer.param_groups[0]["lr"]})
        if opt.method.startswith("Sig") or opt.method.startswith("Focal"):
            log_data.update(
                {
                    "logit_scale": model.logit_scale.item(),
                    "logit_bias": model.logit_bias.item(),
                }
            )
        fabric.log_dict(log_data, step=epoch)

        if fabric.is_global_zero and epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{epoch}.pth")
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    if fabric.is_global_zero:
        save_file = os.path.join(opt.save_folder, "last.pth")
        save_model(model, optimizer, opt, opt.epochs, save_file)

    # Clean up memory before linear evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # After training loop, run linear evaluation on saved checkpoints
    if fabric.is_global_zero:
        run_linear_eval_on_saved_checkpoints(opt, fabric)


if __name__ == "__main__":
    main()
