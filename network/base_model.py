import timm
import torch
import torch.nn as nn
from torchvision import models
import network.resnet as resnet
from functools import partial

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, dataset='cifar10'):
        super(ModelBase, self).__init__()
        # net = resnet.resnet18()

        # dim_mlp = net.fc.weight.shape[1]
        # self.net = []
        self.net = timm.create_model(
            "convnext_base",
            pretrained=True,
            drop_path_rate=0,
            head_init_scale=1.0,
            features_only=True
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)


    def forward(self, x):
        # x = self.net(x)

        x = self.net(x)
        out = x[len(x) - 1]
        x = self.pool(out)
        x = self.flatten(x)
        return x

# class ModelBase(nn.Module):
#     """
#     Common CIFAR ResNet recipe.
#     Comparing with ImageNet ResNet recipe, it:
#     (i) replaces conv1 with kernel=3, str=1
#     (ii) removes pool1
#     """
#
#     def __init__(self, dataset='cifar10'):
#         super(ModelBase, self).__init__()
#         # net = resnet.resnet18()
#
#         self.net = timm.create_model(
#             "resnet50",
#             pretrained=True,
#             # drop_path_rate=0,
#             # head_init_scale=1.0,
#             # features_only=True
#         )
#         self.net.global_pool = torch.nn.Identity()
#         self.net.fc = torch.nn.Identity()
#
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.flatten = nn.Flatten(1)
#
#     def forward(self, x):
#         # x = self.net(x)
#
#         x = self.net(x)
#         # out = x[len(x) - 1]
#         x = self.pool(x)
#         x = self.flatten(x)
#         return x


class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBaseMoCo(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, feature_dim=128, arch='', dataset='cifar10', bn_splits=8):
        super(ModelBaseMoCo, self).__init__()

        # # use split batchnorm
        # norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        # resnet_arch = resnet.resnet18
        # net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)
        #
        # dim_mlp = net.fc.weight.shape[1]
        # self.net = []
        # """
        # self.net1 = RefConv(3, 3, kernel_size=3, stride=1)
        self.net = timm.create_model(
            "convnext_base",
            pretrained=True,
            drop_path_rate=0,
            head_init_scale=1.0,
            features_only=True
        )
        # self.net.global_pool = torch.nn.Identity()
        # self.net.fc = torch.nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # # self.conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten(1)


    def forward(self, x):
        # x = self.net(x)
        # # note: not normalized here

        x = self.net(x)
        #out = x
        out = x[len(x) - 1]
        # x = self.pool(x)
        # x = self.flatten(x)

        return out

if __name__ == '__main__':
    net = ModelBaseMoCo()
    print(net)
