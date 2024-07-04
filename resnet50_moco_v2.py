import torch
from dataset.data import *
from network.head import *
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from util.meter import *
import time
from util.torch_dist_sum import *
from util.dist_init import dist_init
import argparse
# from network.base_model_resnet50 import *
import torchvision.models as models
import os
from src.utils import bool_flag

ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=21120)
parser.add_argument('--s', type=str, default='cos')
parser.add_argument('--dataset', type=str, default='Pandora18k')
parser.add_argument('--checkpoint', type=str, default='./model_lm/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220225-5b9f06a0.pth')
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=True, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")
args = parser.parse_args()
print(args)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(1, -1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    rank, local_rank, world_size = dist_init(args.port)
    epochs = 100
    batch_size = 16 // world_size  # 256
    num_workers = 0
    if args.s == 'cos':
        lr = 0.3  # 0.2
    else:
        lr = 30

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='data', download=True, transform=get_train_augment('cifar10'))
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=get_test_augment('cifar10'))
        num_classes = 10
    elif args.dataset == 'cross1':
        train_dataset = ImageFolder("./data/cross1/Train", transform=get_train_augment('cross1'))
        test_dataset = ImageFolder("./data/cross1/Test", transform=get_test_augment('cross1'))
        image, label = train_dataset[0]
        print(image.shape)
    elif args.dataset == 'Pandora18k':
        train_dataset = ImageFolder("./data/Pandora18k/train", transform=get_train_augment('Pandora18k'))
        test_dataset = ImageFolder("./data/Pandora18k/test", transform=get_test_augment('Pandora18k'))
        image, label = train_dataset[0]
        print(image.shape)
    elif args.dataset == 'AVA':
        train_dataset = ImageFolder("./data/AVA/train", transform=get_train_augment('AVA'))
        test_dataset = ImageFolder("./data/AVA/test", transform=get_test_augment('AVA'))
        image, label = train_dataset[0]
        print(image.shape)
    elif args.dataset == 'Flickr6':
        train_dataset = ImageFolder("./data/Flickr6/train", transform=get_train_augment('Flickr6'))
        test_dataset = ImageFolder("./data/Flickr6/test", transform=get_test_augment('Flickr6'))
        image, label = train_dataset[0]
        print(image.shape)
    elif args.dataset == 'painting91':
        train_dataset = ImageFolder("./data/painting91/train", transform=get_train_augment('painting91'))
        test_dataset = ImageFolder("./data/painting91/test", transform=get_test_augment('painting91'))
        image, label = train_dataset[0]
        print(image.shape)
    elif args.dataset == 'Pandora18k':
        train_dataset = ImageFolder("./data/Pandora18k/train", transform=get_train_augment('Pandora18k'))
        test_dataset = ImageFolder("./data/Pandora18k/test", transform=get_test_augment('Pandora18k'))
        image, label = train_dataset[0]
        print(image.shape)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='data', download=True, transform=get_train_augment('cifar100'))
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True,
                                         transform=get_test_augment('cifar100'))
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        train_dataset = TinyImagenet(root='data/tiny-imagenet-200/train', transform=get_train_augment('tinyimagenet'))
        test_dataset = TinyImagenet(root='data/tiny-imagenet-200/val', transform=get_test_augment('tinyimagenet'))
        num_classes = 200
    else:
        train_dataset = datasets.STL10(root='data', download=True, split='train', transform=get_train_augment('stl10'))
        test_dataset = datasets.STL10(root='data', download=True, split='test', transform=get_test_augment('stl10'))
        num_classes = 10

    dim_in = 1024
    # pre_train = ModelBase(dataset=args.dataset)

    prefix = 'net.'
    model = models.__dict__[args.arch]()

    model.fc = nn.Identity()
    linear_classifier = RegLog(18, args.arch, args.global_pooling, args.use_bn)

    model.cuda()
    linear_classifier = linear_classifier.cuda()

    model.eval()
    if os.path.isfile(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cuda:0")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
    else:
        print("No pretrained weights found => training with random weights")

    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0,
        nesterov=True
    )
    # for k in list(state_dict.keys()):
    #     if not k.startswith(prefix):
    #         del state_dict[k]
    #     if k.startswith(prefix):
    #         state_dict[k[len(prefix):]] = state_dict[k]
    #         del state_dict[k]
    # pre_train.load_state_dict(state_dict)

    # model = LinearHead(pre_train1, dim_in=dim_in, num_class=num_classes)
    model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank,
                                    find_unused_parameters=True)
    # optimizer = torch.optim.SGD(model.module.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)

    torch.backends.cudnn.benchmark = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler, drop_last=True)

    if args.s == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * train_loader.__len__())
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    best_acc = 0
    best_acc5 = 0

    for epoch in range(epochs):
        # ---------------------- Train --------------------------

        train_sampler.set_epoch(epoch)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            train_loader.__len__(),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch)
        )
        end = time.time()

        model.eval()
        linear_classifier.train()
        for i, (image, label) in enumerate(train_loader):
            data_time.update(time.time() - end)

            image = image.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            out = model(image)
            out = linear_classifier(out)

            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item())

            if i % 10 == 0 and rank == 0:
                progress.display(i)

            if args.s == 'cos':
                scheduler.step()

        if args.s == 'step':
            scheduler.step()

        # ---------------------- Test --------------------------
        model.eval()
        linear_classifier.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            end = time.time()
            for i, (image, label) in enumerate(test_loader):
                image = image.cuda(local_rank, non_blocking=True)
                label = label.cuda(local_rank, non_blocking=True)

                # compute output
                output = model(image)
                output = linear_classifier(output)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)

        top1_acc = sum(sum1.float()) / sum(cnt1.float())
        top5_acc = sum(sum5.float()) / sum(cnt5.float())

        best_acc = max(top1_acc, best_acc)
        best_acc5 = max(top5_acc, best_acc5)

        if rank == 0:
            print(
                'Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(
                    epoch, top1_acc=top1_acc, top5_acc=top5_acc, best_acc=best_acc, best_acc5=best_acc5))


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            # if arch == "resnet50":
            #     s = 512
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # # average pool the final feature map
        # x = self.av_pool(x)
        #
        # # optional BN
        # if self.bn is not None:
        #     x = self.bn(x)
        #
        # # flatten
        # x = x.view(x.size(0), -1)
        #
        # # linear layer
        return self.linear(x)


if __name__ == "__main__":
    main()
