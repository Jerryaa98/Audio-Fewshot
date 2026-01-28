# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/RaghuRBV20,
  author    = {Aniruddh Raghu and
               Maithra Raghu and
               Samy Bengio and
               Oriol Vinyals},
  title     = {Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness
               of {MAML}},
  booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
               Addis Ababa, Ethiopia, April 26-30, 2020},
  year      = {2020},
  url       = {https://openreview.net/forum?id=rkgMkCEtPB}
}
https://arxiv.org/abs/1909.09157
"""
import torch
from torch import nn

from libfewshot_core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module


class ANILLayer(nn.Module):
    def __init__(self, feat_dim, hid_dim, way_num):
        super(ANILLayer, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(feat_dim, hid_dim),
            nn.Linear(feat_dim, way_num)
        )

    def forward(self, x):
        return self.layers(x)


def vote_catagorical_acc(targets, predictions):
    return (predictions == targets).sum().float() / targets.size(0) * 100.0


def majority_vote(soft_logits, query_nums):
    y_preds = soft_logits.argmax(dim=1)

    end_index = 0
    aggregrated_preds = torch.zeros(len(query_nums))
    for idx, num in enumerate(query_nums):
        slice = y_preds[end_index:(end_index + num)]
        value, indices = torch.mode(slice)
        aggregrated_preds[idx] = value
        end_index += slice.shape[0]
    return aggregrated_preds



class ANIL(MetaModel):
    def __init__(self, inner_param, feat_dim, hid_dim=640, **kwargs):
        super(ANIL, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = ANILLayer(
            feat_dim=feat_dim, hid_dim=hid_dim, way_num=self.way_num
        )
        self.inner_param = inner_param

        convert_maml_module(self.classifier)

    def set_forward(self, batch):
        if len(batch) == 2:
            image, global_target = batch
            repeats = None
            support_size = 0
        else:
            image, global_target, repeats, support_size = batch
        image = image.to(self.device)
    
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target, query_mask = self.split_by_episode(
            feat, mode=1, repeats=repeats, support_size=support_size
        )
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.set_forward_adaptation(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        soft_logits = output.softmax(dim=1)
        pre_query_pred = majority_vote(soft_logits, repeats).to('cuda', dtype=torch.long)
        post_query_y = torch.repeat_interleave(query_target.reshape(-1).to('cpu'), repeats.to('cpu')).to('cuda', dtype=torch.long)
        pre_acc = vote_catagorical_acc(query_target.reshape(-1).to('cuda'), pre_query_pred.to('cuda'))
        # acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, pre_acc.item()

    def set_forward_loss(self, batch):
        if len(batch) == 2:
            image, global_target = batch
            repeats = None
        else:
            image, global_target, repeats, support_size = batch
        
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target, query_mask = self.split_by_episode(
            feat, mode=1, repeats=repeats, support_size=support_size
        )
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.set_forward_adaptation(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()

        for i in range(
            self.inner_param["train_iter"]
            if self.training
            else self.inner_param["test_iter"]
        ):
            output = self.classifier(support_feat)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.classifier.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)
