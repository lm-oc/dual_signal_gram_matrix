import torch
from util.torch_dist_sum import *
from util.meter import *
from network.bgt_kl import ReSSL
import time
import os
from dataset.data1 import *
import math
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import ImageFile
from util.dist_init import dist_init
from network.base_model_kl import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--k', type=int, default=4096)
parser.add_argument('--m', type=float, default=0.99)
parser.add_argument('--weak', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=25)

parser.add_argument('--checkpoint', type=str, default='')
args = parser.parse_args()
print(args)

# base +gram + third

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
        lr = 0.05 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('gram', ':.4e')
    ce_losses = AverageMeter('CE', ':.4e')
    losses3 = AverageMeter('loss3', ':.4e')
    losses4 = AverageMeter('loss4', ':.4e')
    #losses5 = AverageMeter('loss5', ':.4e')
    # losses_1 = AverageMeter('loss_1', ':.4e')
    # losses_2 = AverageMeter('loss_2', ':.4e')
    # losses_3 = AverageMeter('loss_3', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ce_losses, losses3, losses, losses4],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img1, img2, img3) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        data_time.update(time.time() - end)
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)
        img3 = img3.cuda(non_blocking=True)

        # compute output
        # loss = model(img1, img2)
        # loss1, output, target, loss2 = model(img1, img2)
        # loss1, loss3, loss2 = model(img1, img2)
        # loss3 = criterion(output, target)
        loss4, loss1, loss2, loss3 = model(img1, img2, img3)

        loss = loss4 * 0.5 + loss1 * 0.5 + loss2 * 0.5 + loss3 * 0.5
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss1.item(), img1.size(0))
        ce_losses.update(loss2.item(), img1.size(0))
        losses3.update(loss3.item(), img1.size(0))
        losses4.update(loss4.item(), img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    # 在for循环之外
    average_loss = (losses.avg + ce_losses.avg + losses3.avg + losses4.avg) / 4
    return average_loss

def validate(val_loader, model):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, (img1, img2, img3) in enumerate(val_loader):
            img1 = img1.cuda(non_blocking=True)
            img2 = img2.cuda(non_blocking=True)
            img3 = img3.cuda(non_blocking=True)

            loss4, loss1, loss2, loss3 = model(img1, img2, img3)
            loss = loss4 * 0.5 + loss1 * 0.5 + loss2 * 0.5 + loss3 * 0.5

            total_loss += loss.item() * img1.size(0)
            total_samples += img1.size(0)

        average_loss = total_loss / total_samples
        return average_loss


def main():
    batch_size = 1  # 256
    base_lr = 0.0005 # 0.6


    model = ReSSL(dataset=args.dataset, K=args.k, m=args.m)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'cifar10':
        dataset = CIFAR10Pair(root='./data', download=True, transform=get_contrastive_augment('cifar10'),
                              weak_aug=get_weak_augment('cifar10'))
    elif args.dataset == 'AVA':
        dataset = ImageFolderPair("./data/AVA/train", transform=get_contrastive_augment('AVA'),
                                  weak_aug=get_weak_augment('AVA'), style_aug=get_style_augment('AVA'))
        dataset_valid = ImageFolderPair("./data/AVA/valid", transform=get_contrastive_augment('AVA'),
                                        weak_aug=get_weak_augment('AVA'), style_aug=get_style_augment('AVA'))
    elif args.dataset == 'Pandora18k':
        dataset = ImageFolderPair("./data/Pandora18k/train", transform=get_contrastive_augment('Pandora18k'),
                                  weak_aug=get_weak_augment('Pandora18k'), style_aug=get_style_augment('Pandora18k'))
        dataset_valid = ImageFolderPair("./data/Pandora18k/test", transform=get_contrastive_augment('Pandora18k'),
                                        weak_aug=get_weak_augment('Pandora18k'),
                                        style_aug=get_style_augment('Pandora18k'))
        image1, image2, image3 = dataset[0]
        print(image1.shape)
        # image1, image2, image3 = dataset[0]
        # print(image1.shape)

    train_loader = DataLoader(dataset, shuffle=True, num_workers=0, pin_memory=True, batch_size=batch_size,
                              drop_last=True)

    val_loader = DataLoader(dataset_valid, shuffle=True, num_workers=0, pin_memory=True, batch_size=batch_size,
                            drop_last=True)
    iteration_per_epoch = train_loader.__len__()

    # optimizer = torch.optim.SGD(model.module.fc.parameters(), lr=base_lr, momentum=0.9, weight_decay=0, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    torch.backends.cudnn.benchmark = True

    checkpoint_path = 'checkpoints/bgt-kl-{}.pth'.format(args.dataset)
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
    average_losses = []
    for epoch in range(start_epoch, epochs):
        avg_loss = train(train_loader, model, optimizer, epoch, iteration_per_epoch, base_lr)
        average_losses.append(avg_loss)

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

    # 将数据保存到文件
    np.save('./log/average_losses.npy', average_losses)

    # plt.figure()
    # plt.plot(range(start_epoch, epochs), average_losses, '-', label='Average Loss', color='lightblue')  # 修改颜色为淡蓝色
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('./log/AVA_loss_1.png', format='png', dpi=300)  # 您可以根据需要更改文件名和格式

if __name__ == "__main__":
    main()