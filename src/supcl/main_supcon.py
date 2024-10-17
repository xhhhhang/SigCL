import argparse
import math
import os
import sys
import time

import numpy as np
import rootutils
import torch
import torch.backends.cudnn as cudnn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from networks.resnet_big import SigCLResNet, SupConResNet
from util import (
    AverageMeter,
    TwoCropTransform,
    adjust_learning_rate,
    save_model,
    set_optimizer,
    warmup_learning_rate,
)

from losses import SupConLoss
from src.losses.loss import SigCLossBase, SigCLossNegWeight, SigCLossPN


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
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
        choices=["cifar10", "cifar100", "path"],
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
        choices=["SupCon", "SimCLR", "SigCL", "SigCLPN", "SigCLBase"],
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
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument(
        "--init_logit_scale", type=float, default=np.log(10), help="initial logit scale"
    )
    parser.add_argument("--init_logit_bias", type=float, default=0, help="initial logit bias")

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

    opt.model_name = "{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}".format(
        opt.method,
        opt.dataset,
        opt.model,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.temp,
        opt.trial,
    )

    if opt.cosine:
        opt.model_name = f"{opt.model_name}_cosine"

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = f"{opt.model_name}_warm"
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
        else:
            opt.warmup_to = opt.learning_rate

    # Load environment variables from .env file
    load_dotenv()

    # Add wandb initialization
    if opt.log_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=opt.model_name,
            config=vars(opt),
            entity=os.getenv("WANDB_ENTITY"),
        )

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
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

    if opt.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True
        )
    elif opt.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True
        )
    elif opt.dataset == "path":
        train_dataset = datasets.ImageFolder(
            root=opt.data_folder, transform=TwoCropTransform(train_transform)
        )
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    return train_loader


def set_model(opt):
    if opt.method == "SupCon":
        model = SupConResNet(name=opt.model)
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == "SigCLPN":
        model = SigCLResNet(
            name=opt.model,
            init_logit_scale=opt.init_logit_scale,
            init_logit_bias=opt.init_logit_bias,
        )
        criterion = SigCLossPN()
    elif opt.method == "SigCL":
        model = SigCLResNet(
            name=opt.model,
            init_logit_scale=opt.init_logit_scale,
            init_logit_bias=opt.init_logit_bias,
        )
        criterion = SigCLossNegWeight(
            max_neg_weight=opt.max_neg_weight, neg_weight_step=opt.neg_weight_step
        )
    elif opt.method == "SigCLBase":
        model = SigCLResNet(
            name=opt.model,
            init_logit_scale=opt.init_logit_scale,
            init_logit_bias=opt.init_logit_bias,
        )
        criterion = SigCLossBase()

    if torch.cuda.is_available():
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.device_count() > 1:
            print(f"cuda device count: {torch.cuda.device_count()}")
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """One epoch training."""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.method == "SigCL" or opt.method == "SigCLPN" or opt.method == "SigCLBase":
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
                loss = criterion(features, labels)
            elif opt.method == "SimCLR":
                loss = criterion(features)
            else:
                raise ValueError(f"contrastive method not supported: {opt.method}")

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            if opt.method == "SigCL" or opt.method == "SigCLPN":
                log_str = (
                    "Train: [{0}][{1}/{2}]\t"
                    "logit_scale {logit_scale:.3f}\t"
                    "logit_bias {logit_bias:.3f}\t"
                    "loss {loss.val:.5f} ({loss.avg:.5f})"
                )
                if opt.method == "SigCL":
                    log_str = "neg_weight {neg_weight:.5f}\t" + log_str
                print(
                    log_str.format(
                        epoch,
                        idx + 1,
                        len(train_loader),
                        neg_weight=criterion.neg_weight if opt.method == "SigCL" else None,
                        logit_scale=logit_scale.item(),
                        logit_bias=logit_bias.item(),
                        loss=losses,
                    )
                )
            else:
                print(
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
                    )
                )
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print(f"epoch {epoch}, total time {time2 - time1:.2f}")

        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        # Replace tensorboard logging with wandb logging
        if opt.log_wandb:
            log_data = {
                "epoch": epoch,
                "loss": loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": total_norm,
            }
            if opt.method.startswith("Sig"):
                log_data.update(
                    {
                        "logit_scale": model.logit_scale.item(),
                        "logit_bias": model.logit_bias.item(),
                    }
                )
            wandb.log(log_data)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{epoch}.pth")
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # Close wandb run
    if opt.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
