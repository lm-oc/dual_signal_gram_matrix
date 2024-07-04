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



def main():
    batch_size = 2  # 256
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
        # image1, image2, image3 = dataset[0]
        # print(image1.shape)

    train_loader = DataLoader(dataset, shuffle=True, num_workers=0, pin_memory=True, batch_size=batch_size,
                              drop_last=True)

    val_loader = DataLoader(dataset_valid, shuffle=True, num_workers=0, pin_memory=True, batch_size=batch_size,
                            drop_last=True)
    iteration_per_epoch = train_loader.__len__()

    optimizer = torch.optim.SGD(model.module.fc.parameters(), lr=base_lr, momentum=0.9, weight_decay=0, nesterov=True)

    torch.backends.cudnn.benchmark = True

    pre_train = ModelBase(dataset=dataset)
    prefix = 'net.'
    state_dict = torch.load('checkpoints/' + "bgt-bs16-lr0005-size416-AVA.pth", map_location='cpu')['model']
    for k in list(state_dict.keys()):
        if not k.startswith(prefix):
            del state_dict[k]
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict[k]
            del state_dict[k]
    pre_train.load_state_dict(state_dict)
    # model = LinearHead(pre_train, dim_in=dim_in, num_class=num_classes)
    # model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank,
    #                                 find_unused_parameters=True)
    rank, local_rank, world_size = dist_init(port=args.port)

    teacher_model = pre_train.to(local_rank)
    student1_model = ModelBaseMoCo_kl(dataset=dataset).cuda()
    student2_model = ModelBaseMoCo_kl(dataset=dataset).cuda()

    teacher_model.eval()

    for epoch in range(5):
        model.train()
        for i, (img1, img2, img3) in enumerate(train_loader):

            img1 = img1.cuda(non_blocking=True)
            img2 = img2.cuda(non_blocking=True)
            img3 = img3.cuda(non_blocking=True)
            with torch.no_grad():
                teacher_output = teacher_model(image)

        for image, label in train_loder:
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                teacher_output = teacher_model(image)
            optim.zero_grad()
            out = model(image)
            loss = hard_loss(out, label)
            ditillation_loss = soft_loss(F.softmax(out / T, dim=1), F.softmax(teacher_output / T, dim=1))
            loss_all = loss * alpha + ditillation_loss * (1 - alpha)
            loss.backward()
            optim.step()

        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for image, label in test_dataloder:
                image = image.to(device)
                label = label.to(device)
                out = model(image)
                pre = out.max(1).indices
                num_correct += (pre == label).sum()
                num_samples += pre.size(0)
            acc = (num_correct / num_samples).item()

        model.train()
        print("epoches:{},accurate={}".format(epoch, acc))



if __name__ == "__main__":
    main()