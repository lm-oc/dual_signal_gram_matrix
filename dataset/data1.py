from torch.utils.data import dataset, Dataset
from torchvision import datasets, transforms
from PIL import ImageFilter, Image
import numpy as np
import memcache as mc
import io
import random
import kornia
from torchvision.transforms import GaussianBlur
from kornia import filters
import torch
from torchvision.datasets import ImageFolder
from network.auto_augment import AutoAugment
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TinyImagenetBase(datasets.ImageFolder):

    def __init__(self, root):
        super().__init__(root)
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)

        buff = io.BytesIO(value_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img


class TinyImagenet(TinyImagenetBase):
    def __init__(self, root, transform):
        super().__init__(root)

        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.load_image(path)
        return self.transform(img), label


class TinyImagenetPair(TinyImagenetBase):
    def __init__(self, root, transform, weak_aug=None):
        super().__init__(root)
        self.transform = transform
        self.weak_aug = weak_aug

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.load_image(path)
        pos_1 = self.transform(img)
        if self.weak_aug is not None:
            pos_2 = self.weak_aug(img)
        else:
            pos_2 = self.transform(img)
        return pos_1, pos_2


class STL10Pair(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
        return pos_1, pos_2


class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)

        return pos_1, pos_2


class ImageFolderPair(ImageFolder):
    def __init__(self, root, transform=None, weak_aug=None, style_aug=None):
        super().__init__(root, transform=transform)
        self.weak_aug = weak_aug
        self.style_aug = style_aug

    def __getitem__(self, index):
        # 获取 index 对应的原始样本及其标签
        path, target = self.samples[index]
        img = self.loader(path)

        # 对原始样本做两次不同的变换，得到一组正样本对（pos_1, pos_2）
        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)

            if self.style_aug is not None:
                pos_3 = self.style_aug(img)
            else:
                pos_3 = self.transform(img)

        return pos_1, pos_2, pos_3


class CIFAR100Pair(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
        return pos_1, pos_2


def get_contrastive_augment(dataset):
    size = 352
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def get_moco_augment(dataset):
    size = 224
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),

        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def get_weak_augment(dataset):
    size = 352
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # mean = (0.5071, 0.4867, 0.4408)
        # std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def  get_style_augment(dataset):
    size = 352
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)
    style_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    return style_transform

def get_train_augment(dataset):
    size = 352
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform


def get_test_augment(dataset):
    size = 352
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)

    if dataset == 'stl10':
        test_transform = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform
