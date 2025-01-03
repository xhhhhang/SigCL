import argparse
import math
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from main_ce import set_loader
from networks.resnet_big import LinearClassifier, SigCLResNet, SupConResNet
from tqdm import tqdm
from util import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    seed_everything,
    set_optimizer,
    warmup_learning_rate,
    create_optimizer,
)

# Import wandb conditionally
try:
    import wandb
except ImportError:
    wandb = None


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=8, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs", type=str, default="60,75,90", help="where to decay lr, can be a list"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.2, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
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

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")

    parser.add_argument("--ckpt", type=str, default="", help="path to pre-trained model")
    parser.add_argument(
        "--method",
        type=str,
        default="SupCon",
        help="method",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="enable logging to Weights & Biases"
    )
    parser.add_argument(
        "--disable_progress",
        action="store_true",
        help="disable all print and wandb except the last print",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "lars", "rmsprop"],
        help="optimizer for linear evaluation",
    )

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = "./datasets/"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "{}_{}_lr_{}_decay_{}_bsz_{}".format(
        opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size
    )

    if opt.cosine:
        opt.model_name = f"{opt.model_name}_cosine"

    # warm-up for large-batch training,
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

    if opt.dataset == "cifar10":
        opt.n_cls = 10
    elif opt.dataset == "cifar100":
        opt.n_cls = 100
    elif opt.dataset == "imagenet":
        opt.n_cls = 1000
    else:
        raise ValueError(f"dataset not supported: {opt.dataset}")

    return opt


def set_model(opt):
    if opt.method == "SupCon":
        model = SupConResNet(name=opt.model)
    elif opt.method.startswith("SigCL") or opt.method.startswith("Focal"):
        model = SigCLResNet(name=opt.model)
    else:
        raise ValueError(f"method not supported: {opt.method}")

    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt["model"]

    if torch.cuda.is_available():
        #     if torch.cuda.device_count() > 1:
        #         model.encoder = torch.nn.DataParallel(model.encoder)
        #     else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError("This code requires GPU")

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """One epoch training."""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, examples in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.dataset == "imagenet":
            images, labels = examples["image"], examples["label"]
        else:
            images, labels = examples

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        # print(output.shape, labels.shape)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 and not opt.disable_progress:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

        # Log to wandb if enabled
        if opt.log_wandb and not opt.disable_progress:
            wandb.log(
                {
                    "train/loss": losses.val,
                    "train/acc@1": top1.val,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch * len(train_loader) + idx,
            )

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation."""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, examples in enumerate(val_loader):
            if opt.dataset == "imagenet":
                images, labels = examples["image"], examples["label"]
            else:
                images, labels = examples

            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0 and not opt.disable_progress:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                    )
                )

    # Log to wandb if enabled
    if opt.log_wandb and not opt.disable_progress:
        wandb.log(
            {
                "val/loss": losses.avg,
                "val/acc@1": top1.avg,
            }
        )

    if not opt.disable_progress:
        print(f" * Acc@1 {top1.avg:.3f}")
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    seed_everything(42)

    # Initialize wandb if enabled and progress is not disabled
    if opt.log_wandb and not opt.disable_progress:
        if wandb is None:
            raise ImportError("Please install wandb to use logging functionality")

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "SigCL"),
            entity=os.environ.get("WANDB_ENTITY", None),
            config=vars(opt),
            name=opt.model_name,
        )

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = create_optimizer(
        opt.optimizer,
        classifier.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    # training routine
    for epoch in tqdm(range(1, opt.epochs + 1)):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        time2 = time.time()
        if not opt.disable_progress:
            print(f"Train epoch {epoch}, total time {time2 - time1:.2f}, accuracy:{acc:.2f}")

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

        # Log best accuracy to wandb if enabled and progress is not disabled
        if opt.log_wandb and not opt.disable_progress:
            wandb.log({"best_acc": best_acc}, step=(epoch + 1) * len(train_loader))

    print(f"{opt.method} {opt.model} {opt.dataset} best accuracy: {best_acc:.2f}")

    # Finish wandb run if enabled and progress is not disabled
    if opt.log_wandb and not opt.disable_progress:
        wandb.finish()


if __name__ == "__main__":
    main()
