import torch.nn as nn
import torch
from network.base_model import ModelBaseMoCo
from network.head import MoCoHead
import torch.nn.functional as F

#base + third

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
        self.encoder_style = ModelBaseMoCo(dataset=dataset, bn_splits=bn_splits)

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

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute query features
        q = self.net(im1)  # queries: NxC

        # gram1 = fn_gram(q)
        q = self.pool(q)
        q = self.flatten(q)
        q = self.head_q(q)
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im2)

            k = self.encoder_k(im_k_)  # keys: NxC

            k = self.pool(k)
            k = self.flatten(k)
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

            im_style_, idx_unshuffle_style = self._batch_shuffle_single_gpu(im3)
            im_style_.cuda()
            v = self.encoder_style(im_style_) # keys: NxC

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
        # loss = (loss_qv + loss_qk)/2.0

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue(v)
        return loss_qk, loss_qv



