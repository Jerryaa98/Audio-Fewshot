# -*- coding: utf-8 -*-
'''

'''
import torch
from torch import nn
import torch.nn.functional as F
from libfewshot_core.utils import accuracy, majority_vote, vote_catagorical_acc
from .finetuning_model import FinetuningModel

class ProtoLayer(nn.Module):
    def __init__(self):
        super(ProtoLayer, self).__init__()

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
        mode="cos_sim",
    ):
        t, wq, c = query_feat.size()
        t2, ws, _ = support_feat.size()

        # t, wq, c
        # query_feat = query_feat.reshape(t, way_num * query_num, c)
        # t, w, c
        support_feat = support_feat.reshape(t2, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)

        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            ),
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            ),
        }[mode](query_feat, proto_feat)
        
class MetabaselinePretrain(FinetuningModel):
    def __init__(self, feat_dim, num_class, **kwargs):
        super(MetabaselinePretrain, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()
        


    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        if len(batch) == 2:
            image, global_target = batch
            repeats = None
            support_size = 0
        else:
            image, global_target, repeats, support_size = batch
        
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
             # use tsne to visualize the feature distribution


        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1, repeats=repeats, support_size=support_size
        )
        episode_size = support_feat.size(0)
        output = []
        for i in range(episode_size):
            output_per_episode = self.proto_layer(
                query_feat[i].unsqueeze(0), support_feat[i].unsqueeze(0), self.way_num, self.shot_num, query_feat[i].size(0)
            ).reshape(-1, self.way_num)
            output.append(output_per_episode)
        output = torch.cat(output, dim=0)
        
        soft_logits = output.softmax(dim=1)
        pre_query_pred = majority_vote(soft_logits, repeats).to('cuda', dtype=torch.long)
        post_query_y = torch.repeat_interleave(query_target.reshape(-1), repeats).to('cuda', dtype=torch.long)
        acc = vote_catagorical_acc(query_target.reshape(-1).to('cuda'), pre_query_pred.to('cuda'))
        # acc = accuracy(output, query_target.reshape(-1))

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
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    