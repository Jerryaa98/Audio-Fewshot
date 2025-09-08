# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/LiWXHGL19,
  author    = {Wenbin Li and
               Lei Wang and
               Jinglin Xu and
               Jing Huo and
               Yang Gao and
               Jiebo Luo},
  title     = {Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot
               Learning},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
               2019, Long Beach, CA, USA, June 16-20, 2019},
  pages     = {7260--7268},
  year      = {2019},
  url       = {http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To
  -Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html},
  doi       = {10.1109/CVPR.2019.00743}
}
https://arxiv.org/abs/1903.12290

Adapted from https://github.com/WenbinLee/DN4.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy, majority_vote, vote_catagorical_acc
from .metric_model import MetricModel


class DN4Layer(nn.Module):
    def __init__(self, n_k):
        super(DN4Layer, self).__init__()
        self.n_k = n_k

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
    ):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        try:
            query_feat = query_feat.view(t, way_num * query_num, c, h * w).permute(
                0, 1, 3, 2
            )
        except:
            query_feat = query_feat.view(t, query_num, c, h * w).permute(
                0, 1, 3, 2
            )
        query_feat = F.normalize(query_feat, p=2, dim=-1).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = (
            support_feat.view(t, way_num, shot_num, c, h * w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(t, way_num, c, shot_num * h * w)
        )
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])

        return score


class DN4(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        super(DN4, self).__init__(**kwargs)
        self.dn4_layer = DN4Layer(n_k)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        if len(batch) == 2:
            image, target = batch
            repeats = None
            support_size = 0
        else:
            image, target, repeats, support_size = batch
        image = image.to(self.device)
        
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2, repeats=repeats, support_size=support_size
        )
        episode_size = len(query_feat)

        output = []
        for i in range(episode_size):
            output_per_episode = self.dn4_layer(
                query_feat[i].unsqueeze(0), support_feat[i].unsqueeze(0), self.way_num, self.shot_num, query_feat[i].size(0)
            ).view(-1, self.way_num)
            output.append(output_per_episode)
        output = torch.cat(output, 0)
        
        # acc = accuracy(output, query_target.reshape(-1))
        soft_logits = output.softmax(dim=1)
        pre_query_pred = majority_vote(soft_logits, repeats).to('cuda', dtype=torch.long)
        post_query_y = torch.repeat_interleave(query_target.reshape(-1), repeats).to('cuda', dtype=torch.long)
        acc = vote_catagorical_acc(query_target.reshape(-1).to('cuda'), pre_query_pred.to('cuda'))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        if len(batch) == 2:
            image, target = batch
            repeats = None
            support_size = 0
        else:
            image, target, repeats, support_size = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2, repeats=repeats, support_size=support_size
        )

        output = self.dn4_layer(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
