import torch.nn as nn
import torch
from network.base_model import ModelBaseMoCo_234
from network.head import MoCoHead
import torch.nn.functional as F
from scipy.spatial.distance import pdist

# base + gram + third

def gram_loss(gram1_feature, gram2_feature):
    outs = torch.mean(torch.stack([gram1_feature, gram2_feature]), dim=0)
    gram_avg = fn_gram(outs)
    means_avg = fn_means(gram_avg)
    gram = torch.sub(gram_avg, means_avg)

    gram_im1 = fn_gram(gram1_feature)
    # print(gram_im1.shape)
    gram_im2 = fn_gram(gram2_feature)
    loss = compute_loss(gram, gram_im1) * 0.5 + compute_loss(gram, gram_im2) * 0.5
    return loss

def gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = torch.unsqueeze(gram, -1)
    gram = gram / (C * H * W * 1)
    return gram


def compute_loss(gram1, gram2):
    gram1 = gram1.view(-1)
    gram2 = gram2.view(-1)

    similarity = F.cosine_similarity(gram1, gram2, dim=0)

    loss = 1 - similarity
    return loss


def fn_gram(x):
    x = x.permute(0, 2, 3, 1)
    gram_result = torch.einsum('bijc,bijd->bcd', x, x)
    gram_result = torch.unsqueeze(gram_result, -1)
    return gram_result


def fn_means(x):
    means = torch.mean(x, dim=3, keepdim=False)
    means = torch.mean(means, dim=2)
    means = torch.mean(means, dim=1)
    means = torch.unsqueeze(means, -1)
    means = torch.unsqueeze(means, -1)
    means = torch.unsqueeze(means, -1)
    return means

def distance_loss(student_outputs, teacher_outputs, margin=1.0):
    """
    计算教师和学生输出特征之间的平方欧氏距离。
    """
    pairwise_distance = torch.nn.functional.pairwise_distance(student_outputs, teacher_outputs)
    loss = torch.mean(torch.clamp(pairwise_distance - margin, min=0.0))
    return loss

def angle_loss(student_outputs, teacher_outputs):
    """
    计算教师和学生输出特征之间的角度损失。
    """
    dot_product = torch.sum(student_outputs * teacher_outputs, dim=1)
    norm_product = torch.linalg.norm(student_outputs, dim=1) * torch.linalg.norm(teacher_outputs, dim=1)
    cosine = dot_product / norm_product
    angle_diff = 1 - cosine  # Minimize the angle difference
    loss = torch.mean(angle_diff)
    return loss

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # 添加一个小的正值 epsilon，避免除零
        epsilon = 1e-8

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2, eps=epsilon)  # 使用eps参数增加稳定性
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2, eps=epsilon)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')  # 'elementwise_mean' to 'mean'
        return loss

# class RKdAngle(nn.Module):
#     def forward(self, student, teacher):
#         # N x C
#         # N x N x C
#
#         with torch.no_grad():
#             td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
#             norm_td = F.normalize(td, p=2, dim=2)
#             t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
#
#         sd = (student.unsqueeze(0) - student.unsqueeze(1))
#         norm_sd = F.normalize(sd, p=2, dim=2)
#         s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
#
#         loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
#         return loss

def pdist_torch(x, squared=False):
    # x is a 2D tensor N x C
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, x.t())
    dist = dist.clamp(min=0.0)
    if not squared:
        dist = torch.sqrt(dist)
    return dist

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        epsilon = 1e-8

        with torch.no_grad():
            t_d = torch.pdist(teacher, p=2) + epsilon  # 加上 epsilon 避免除零
            mean_td = t_d.mean() if t_d.numel() > 0 else epsilon
            t_d = t_d / (mean_td + epsilon)  # 再次确保不会除以零

        d = torch.pdist(student, p=2) + epsilon
        mean_d = d.mean() if d.numel() > 0 else epsilon
        d = d / (mean_d + epsilon)

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')  # 'elementwise_mean' to 'mean'
        return loss


class ReSSL(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, dataset='cifar10', bn_splits=8):
        super(ReSSL, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        self.net = ModelBaseMoCo_234(dataset=dataset, bn_splits=bn_splits)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv2d(1024,256, kernel_size=1, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten(1)

        self.encoder_k = ModelBaseMoCo_234(dataset=dataset, bn_splits=bn_splits)
        self.encoder_style = ModelBaseMoCo_234(dataset=dataset, bn_splits=bn_splits)

        self.head_q = MoCoHead(input_dim=1024)
        self.head_k = MoCoHead(input_dim=1024)
        self.head_style = MoCoHead(input_dim=1024)

        for param_q, param_k in zip(self.net.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.net.parameters(), self.encoder_style.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_style.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.net.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.net.parameters(), self.encoder_style.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_style.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def forward(self, im1, im2, im3):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # dist_criterion = RkdDistance()
        # angle_criterion = RKdAngle()
        at_criterion = AttentionTransfer()

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute query features
        # qfe1, qfe2, qfe3, q = self.net(im1)
        q, qack1, qack2 = self.net(im1)  # queries: NxC
        # print(q.shape)
        gram_im1_feature = q

        # gram1 = fn_gram(q)
        q = self.pool(q)
        q = self.flatten(q)
        # print(q.shape)
        q = self.head_q(q)
        q = nn.functional.normalize(q, dim=1)  # already normalized
        # print(q.shape)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im2)

            # kfe1, kfe2, kfe3, k = self.encoder_k(im_k_)
            k, kack1, kack2 = self.encoder_k(im_k_)  # keys: NxC
            gram_im2_feature = k

            k = self.pool(k)
            k = self.flatten(k)
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

            im_style_, idx_unshuffle_style = self._batch_shuffle_single_gpu(im3)
            im_style_.cuda()
            # vfe1, vfe2, vfe3, v = self.encoder_style(im_style_)
            v, vack1, vack2 = self.encoder_style(im_style_)  # keys: NxC
            gram_im3_feature = v

            v = self.pool(v)
            v = self.flatten(v)
            v = self.head_style(v)
            v = nn.functional.normalize(v, dim=1)  # already normalized
            # undo shuffle
            v = self._batch_unshuffle_single_gpu(v, idx_unshuffle_style)

        logits_q = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits_k = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        logits_v = torch.einsum('nc,ck->nk', [v, self.queue.clone().detach()])
        loss_qk = - torch.sum(F.softmax(logits_k.detach() / 0.04, dim=1) * F.log_softmax(logits_q / 0.1, dim=1),
                              dim=1).mean()
        loss_qv = - torch.sum(F.softmax(logits_v.detach() / 0.04, dim=1) * F.log_softmax(logits_q / 0.1, dim=1),
                              dim=1).mean()

        loss4_gram = gram_loss(gram_im1_feature, gram_im2_feature) * 0.5 + gram_loss(gram_im1_feature, gram_im3_feature) * 0.5

        loss3_gram = gram_loss(qack1, kack1) * 0.5 + gram_loss(qack1, vack1) * 0.5
        loss2_gram = gram_loss(qack2, kack2) * 0.5 + gram_loss(qack2, vack2) * 0.5
        # loss4_gram = gram_loss(qack3, kack3) * 0.5 + gram_loss(qack3, vack3) * 0.5

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue(v)
        return loss2_gram, loss3_gram, loss4_gram, loss_qk, loss_qv
        # return loss3, loss2, rkd_dist_loss,  rkd_angle_loss
        # return loss3, loss2, loss_qk, loss_qv, loss_1, loss_2, loss_3


