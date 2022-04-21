import timm
import time
import math

import torch

import torch.nn as nn
import torch.nn.functional as F
from config import opt

num_classes = 15587
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    dist_mat.to("cuda")
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).to("cuda")
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).to("cuda")

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # print( dist_mat[is_pos].size())
    # print(dist_mat)
    # print(dist_mat.size())
    # print(dist_mat[[True]])
    print(is_pos.dtype)
    print(dist_mat.dtype)
    dist_mat_pmask = torch.masked_select(dist_mat, is_pos)
    print(dist_mat_pmask.dtype)
    disk_mat_nmask = torch.masked_select(dist_mat, is_neg)
    dist_ap, relative_p_inds = torch.max(
        dist_mat_pmask.contiguous().flatten(1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        disk_mat_nmask.contiguous().flatten(1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)
    return loss


class MagrginLinear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=10008, s=64., m=0.5):
        super(MagrginLinear, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label, is_infer=False):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)

        if not is_infer:
            output[idx_, label] = cos_theta_m[idx_, label]

        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class BinaryHead(nn.Module):

    def __init__(self, num_class=15587, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class MarginHead(nn.Module):

    def __init__(self, num_class=15587, emb_size=2048, s=64., m=0.5):
        super(MarginHead, self).__init__()
        self.fc = MagrginLinear(embedding_size=emb_size, classnum=num_class, s=s, m=m)

    def forward(self, fea, label, is_infer):
        fea = l2_norm(fea)
        logit = self.fc(fea, label, is_infer)
        return logit


class HappyWhaleModel(nn.Module):
    def __init__(self, modelName, numClasses, noNeurons, embeddingSize):
        super(HappyWhaleModel, self).__init__()
        #self.fea_extra_layer = [2, 3]
        self.model = timm.create_model(modelName, pretrained=True, features_only=True,
                                       #out_indices=self.fea_extra_layer
                                       )
        self.embsize = embeddingSize
        # in_features = self.model.classifier.in_features
        in_features = 1920
        # self.model.classifier = nn.Identity()
        # self.model.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.poolingl2 = nn.AdaptiveAvgPool2d(1)
        self.poolingl3 = nn.AdaptiveAvgPool2d(1)
        self.poolingl4 = nn.AdaptiveAvgPool2d(1)
        self.poolingl5 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features, embeddingSize),
            nn.BatchNorm1d(embeddingSize),
        )

        # self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.arc_head = ArcMarginProduct(in_features=embeddingSize, out_features=numClasses, m=0.50)
        #self.class_head = BinaryHead(numClasses, emb_size=embeddingSize, s=16.0)

    def forward(self, images, labels=None):
        features = self.model(images)

        # pooled_features = self.pooling(features).flatten(1)
        # pooled_drop = self.drop(pooled_features)
        features[0] = self.pooling(features[0])
        features[1] = self.poolingl2(features[1])
        features[2] = self.poolingl3(features[2])
        features[3] = self.poolingl4(features[3])
        #features[4] = self.poolingl5(features[4])
        emb = torch.cat(features, dim=1)
        emb = emb.flatten(1)
        emb = self.fc(emb)

        if labels != None:

            arc_output = self.arc_head(emb, labels)
            #class_output = self.class_head(emb)
            return arc_output, emb
        else:

            return emb  # feartures
