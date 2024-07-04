import torch
from util.torch_dist_sum import *
from util.meter import *
from network.ressl import ReSSL
import time
import os
from dataset.data import *
import math
import argparse
from torch.utils.data import DataLoader

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--k', type=int, default=4096)
parser.add_argument('--m', type=float, default=0.99)
parser.add_argument('--weak', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=25)
args = parser.parse_args()
print(args)

epochs = args.epochs
warm_up = 5


def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('CE', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ce_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        data_time.update(time.time() - end)
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)

        # compute output
        loss = model(img1, img2)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        ce_losses.update(loss.item(), img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


def validate(val_loader, model):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, (img1, img2) in enumerate(val_loader):
            img1 = img1.cuda(non_blocking=True)
            img2 = img2.cuda(non_blocking=True)

            loss = model(img1, img2)

            total_loss += loss.item() * img1.size(0)
            total_samples += img1.size(0)

        average_loss = total_loss / total_samples
        return average_loss


def main():
    # batch_size = 256
    # base_lr = 0.06

    batch_size = 2
    base_lr = 0.0005
    #batch_size = 32
    #base_lr = 0.0075
    # batch_size = 64
    # base_lr = 0.015

    # base_lr = 0.3

    model = ReSSL(dataset=args.dataset, K=args.k, m=args.m)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'cifar10':
        dataset = CIFAR10Pair(root='data', download=True, transform=get_contrastive_augment('cifar10'),
                              weak_aug=get_weak_augment('cifar10'))
    elif args.dataset == 'cross1':
        dataset = ImageFolderPair("./data/cross1/Train", transform=get_contrastive_augment('cross1'),
                                  weak_aug=get_weak_augment('cross1'))
        dataset_valid = ImageFolderPair("./data/cross1/Valid", transform=get_contrastive_augment('cross1'),
                                        weak_aug=get_weak_augment('cross1'))
    elif args.dataset == 'Pandora18k':
        dataset = ImageFolderPair("./data/Pandora18k/train", transform=get_contrastive_augment('Pandora18k'),
                                  weak_aug=get_weak_augment('Pandora18k'))
        dataset_valid = ImageFolderPair("./data/Pandora18k/test", transform=get_contrastive_augment('Pandora18k'),
                                        weak_aug=get_weak_augment('Pandora18k'))
    elif args.dataset == 'AVA':
        dataset = ImageFolderPair("./data/AVA/train", transform=get_contrastive_augment('AVA'),
                                  weak_aug=get_weak_augment('AVA'))
        dataset_valid = ImageFolderPair("./data/AVA/valid", transform=get_contrastive_augment('AVA'),
                                        weak_aug=get_weak_augment('AVA'))
    elif args.dataset == 'painting91':
        dataset = ImageFolderPair("./data/painting91/train", transform=get_contrastive_augment('painting91'),
                                  weak_aug=get_weak_augment('painting91'))
        dataset_valid = ImageFolderPair("./data/painting91/test", transform=get_contrastive_augment('painting91'),
                                        weak_aug=get_weak_augment('painting91'))
    elif args.dataset == 'SIW-13':
        dataset = ImageFolderPair("./data/SIW-13/train", transform=get_contrastive_augment('SIW-13'),
                                  weak_aug=get_weak_augment('SIW-13'))
    elif args.dataset == 'stl10':
        dataset = STL10Pair(root='data', download=True, split='train+unlabeled',
                            transform=get_contrastive_augment('stl10'), weak_aug=get_weak_augment('stl10'))
    elif args.dataset == 'tinyimagenet':
        dataset = TinyImagenetPair(root='data/tiny-imagenet-200/train',
                                   transform=get_contrastive_augment('tinyimagenet'),
                                   weak_aug=get_weak_augment('tinyimagenet'))
    else:
        dataset = CIFAR100Pair(root='data', download=True, transform=get_contrastive_augment('cifar100'),
                               weak_aug=get_weak_augment('cifar100'))

    train_loader = DataLoader(dataset, shuffle=True, num_workers=0, pin_memory=True, batch_size=batch_size,
                              drop_last=True)
    val_loader = DataLoader(dataset_valid, shuffle=True, num_workers=0, pin_memory=True, batch_size=batch_size,
                            drop_last=True)

    iteration_per_epoch = train_loader.__len__()

    checkpoint_path = 'checkpoints/ressl-16-{}.pth'.format(args.dataset)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(checkpoint_path, 'found, start from epoch', start_epoch)
    else:
        start_epoch = 0
        print(checkpoint_path, 'not found, start from epoch 0')

    model.train()
    best_val_loss = float('inf')
    best_model_state_dict = None
    for epoch in range(start_epoch, epochs):
        train(train_loader, model, optimizer, epoch, iteration_per_epoch, base_lr)

        validation_loss = validate(val_loader, model)
        print("Validation Reconstruction Loss: {:.4f}".format(validation_loss))

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model_state_dict = model.state_dict()

            print("Validation Best Loss: {:.4f}".format(best_val_loss))
            torch.save(
                {
                    'model': best_model_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)

if __name__ == "__main__":
    main()