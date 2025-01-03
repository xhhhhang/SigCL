import argparse
import math
import os
import sys
import time

import rootutils
import torch
import torch.backends.cudnn as cudnn
from dotenv import load_dotenv
from networks.resnet_big import SupCEResNet
from torchvision import datasets, transforms
from util import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    save_model,
    set_optimizer,
    warmup_learning_rate,
    load_imagenet_hf,
    seed_everything,
)

# Import wandb and other necessary modules
import wandb

import torch.optim as optim


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

# Add this near the top of the file, after other imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.2, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="350,400,450",
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
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="dataset"
    )

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--syncBN", action="store_true", help="using synchronized batch normalization"
    )
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    parser.add_argument("--trial", type=str, default="0", help="id for recording multiple runs")

    # Add wandb logging option
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument("--eval_freq", type=int, default=10, help="evaluation frequency")

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = "./datasets/"
    opt.model_path = f"./save/SupCon/{opt.dataset}_models"

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}".format(
        opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.trial
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

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == "cifar10":
        opt.n_cls = 10
    elif opt.dataset == "cifar100":
        opt.n_cls = 100
    else:
        raise ValueError(f"dataset not supported: {opt.dataset}")

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

    return opt


def set_loader(opt):
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
    else:
        raise ValueError(f"dataset not supported: {opt.dataset}")
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    if opt.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=opt.data_folder, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=opt.data_folder, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == "imagenet":
        dataset = load_imagenet_hf(opt, train_transform)
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True
    )

    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
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
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

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
        if (idx + 1) % opt.print_freq == 0:
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

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation."""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                    )
                )

    print(f" * Acc@1 {top1.avg:.3f}")
    return losses.avg, top1.avg


def main():
    seed_everything()
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print(f"epoch {epoch}, total time {time2 - time1:.2f}")

        log_dict = {
            "epoch": epoch,
            "train_loss": loss,
            "train_acc": train_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        # evaluation
        if epoch % opt.eval_freq == 0:
            loss, val_acc = validate(val_loader, model, criterion, opt)
            log_dict.update({
                "val_loss": loss,
                "val_acc": val_acc,
            })

            if val_acc > best_acc:
                best_acc = val_acc

        # log to wandb
        if opt.log_wandb:
            wandb.log(log_dict)

        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{epoch}.pth")
        #     save_model(model, optimizer, opt, epoch, save_file)

    # # save the last model
    # save_file = os.path.join(opt.save_folder, "last.pth")
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    print(f"best accuracy: {best_acc:.2f}")

    # Close wandb run
    if opt.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
