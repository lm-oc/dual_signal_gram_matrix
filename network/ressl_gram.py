import torch.nn as nn
import torch
from network.base_model import ModelBaseMoCo
from network.head import MoCoHead
import torch.nn.functional as F

#base + gram

def gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram / (C * H * W)
    return gram


def compute_loss(gram1, gram2):
    gram1 = gram1.view(-1)
    gram2 = gram2.view(-1)

    similarity = F.cosine_similarity(gram1, gram2, dim=0)

    loss = 1 - similarity
    return loss


def fn_gram(x):
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


class ReSSL(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, dataset='cifar10', bn_splits=8):
        super(ReSSL, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        self.net = ModelBaseMoCo(dataset=dataset, bn_splits=bn_splits)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv2d(1024,256, kernel_size=1, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten(1)

        self.encoder_k = ModelBaseMoCo(dataset=dataset, bn_splits=bn_splits)

        self.head_q = MoCoHead(input_dim=1024)
        self.head_k = MoCoHead(input_dim=1024)

        for param_q, param_k in zip(self.net.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
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

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute query features
        q = self.net(im1)  # queries: NxC

        gram_im1_feature = q

        q = self.pool(q)
        q = self.flatten(q)

        q = self.head_q(q)
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im2)

            k = self.encoder_k(im_k_)  # keys: NxC

            gram_im2_feature = k

            k = self.pool(k)
            k = self.flatten(k)
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        logits_q = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits_k = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        loss = - torch.sum(F.softmax(logits_k.detach() / 0.04, dim=1) * F.log_softmax(logits_q / 0.1, dim=1),
                            dim=1).mean()

        outs = torch.mean(torch.stack([gram_im1_feature, gram_im2_feature]), dim=0)
        gram_avg = fn_gram(outs)
        means_avg = fn_means(gram_avg)
        gram = torch.sub(gram_avg, means_avg)

        gram_im1 = fn_gram(gram_im1_feature)
        gram_im2 = fn_gram(gram_im2_feature)
        loss2 = compute_loss(gram, gram_im1) * 0.5 + compute_loss(gram, gram_im2) * 0.5

        self._dequeue_and_enqueue(k)
        return loss2, loss



