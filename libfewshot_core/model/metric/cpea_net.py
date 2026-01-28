# -*- coding: utf-8 -*-
"""
@InProceedings{Hao_2023_ICCV,
    author    = {Hao, Fusheng and He, Fengxiang and Liu, Liu and Wu, Fuxiang and Tao, Dacheng and Cheng, Jun},
    title     = {Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18905-18915}
}
"""
import torch
from torch import nn
from torch.nn import functional as F
from libfewshot_core.model.metric.metric_model import MetricModel
from libfewshot_core.utils import accuracy, majority_vote, vote_catagorical_acc


def rearrange_data(data, num_classes, k):
    n = len(data)
    # print(n, data)
    rearranged_data = [[] for _ in range(n)]
    # print(rearranged_data)
    for i in range(n):
        new_index = (i // k) + (i % k) * num_classes
        rearranged_data[new_index] = data[i]
    return rearranged_data

def rearrange_data_old(data, num_classes, k):
    n = len(data)
    rearranged_data = torch.empty_like(data)
    for i in range(n):
        new_index = (i // k) + (i % k) * num_classes
        rearranged_data[new_index] = data[i]
    return rearranged_data


def accuracy(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


class SmoothCELoss(nn.Module):
    def __init__(self, eps=0.1, way=5):
        super(SmoothCELoss, self).__init__()
        self.eps = eps

    def forward(self, results, label, way=5):
        one_hot = torch.zeros_like(results).scatter(1, label.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (way - 1)
        log_prb = F.log_softmax(results, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
        return loss


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CPEALayer(nn.Module):
    def __init__(self, in_dim=384):
        super(CPEALayer, self).__init__()

        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim / 4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)

        self.fc2 = Mlp(in_features=72**2, hidden_features=256, out_features=1)

    def forward(self, feat_query, feat_shot, shot):
        # query: Q x n x C
        # feat_shot: KS x n x C
        _, n, c = feat_query.size()
        # print(feat_query.size())

        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query  # Q x n x C
        feat_shot = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot  # KS x n x C
        feat_query = self.fc_norm1(feat_query)
        feat_shot = self.fc_norm1(feat_shot)

        query_class = feat_query[:, 0, :].unsqueeze(1)  # Q x 1 x C
        query_image = feat_query[:, 1:, :]  # Q x L x C

        support_class = feat_shot[:, 0, :].unsqueeze(1)  # KS x 1 x C
        support_image = feat_shot[:, 1:, :]  # KS x L x C

        feat_query = query_image + 2.0 * query_class  # Q x L x C
        feat_shot = support_image + 2.0 * support_class  # KS x L x C

        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)

        feat_shot = feat_shot.contiguous().reshape(shot, -1, n - 1, c)  # K x S x n x C
        feat_shot = feat_shot.mean(dim=0)  # S x n x C
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)

        # similarity measure
        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx]  # n x C
            tmp_query = tmp_query.unsqueeze(0)  # 1 x n x C
            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2))  # S x L x L
            out = out.flatten(1)  # S x L*L
            # print(out.shape)
            # input()
            out = self.fc2(out.pow(2))  # S x 1
            out = out.transpose(0, 1)  # 1 x S
            results.append(out)

        return results


class CPEANet(MetricModel):
    def __init__(self, in_dim=384, **kwargs):
        super(CPEANet, self).__init__(**kwargs)
        self.CPEA = CPEALayer()
        self.loss_func = SmoothCELoss()

    def set_forward(self, batch):
        # if torch.cuda.is_available():
        #     data, _ = [_.cuda() for _ in batch]
        # else:
        #     data = batch[0]
        if len(batch) == 2:
            image, target = batch
            repeats = None
            support_size = 0
        else:
            image, target, repeats, support_size = batch
        if repeats is None or repeats.sum().item() == self.query_num * self.way_num:
            data = rearrange_data(image, self.way_num, self.shot_num + self.query_num)
        else:
            cnt = 0
            tmp = []
            repeat_counter = []
            idx = []
            for i in range(0, len(repeats), self.query_num):
                repeat_counter.extend([1 for _ in range(self.shot_num)])
                tmp.extend(image[cnt:cnt + self.shot_num])
                idx.extend([-1 for _ in range(self.shot_num)])
                cnt += self.shot_num
                for j in range(i, i+self.query_num):
                    tmp.extend(image[cnt:cnt + 1])
                    idx.append(cnt)
                    cnt += repeats[j].item()
                    repeat_counter.extend([repeats[j].item()])
            data = rearrange_data(tmp, self.way_num, self.shot_num + self.query_num)
            repeats_tmp = rearrange_data(repeat_counter, self.way_num, self.shot_num + self.query_num)
            idx_repeats = rearrange_data(idx, self.way_num, self.shot_num + self.query_num)
            data = torch.stack(data).to(self.device)
            data = torch.repeat_interleave(data, torch.tensor(repeats_tmp).to(torch.long).to('cuda'), dim=0)
            cnt = 0
            for i in range(len(idx_repeats)):
                if repeats_tmp[i] != 1:
                    data[cnt:cnt + repeats_tmp[i]] = image[idx_repeats[i] : idx_repeats[i] + repeats_tmp[i]]
                cnt += repeats_tmp[i]
                
        p = self.shot_num * self.way_num
        data_shot, data_query = data[:p], data[p:]
        if type(data_shot) is list:
            data_shot = torch.stack(data_shot).to(self.device)
        if type(data_query) is list:
            try:
                data_query = torch.stack(data_query).to(self.device)
            except Exception as e:
                print(repeats, len(repeats), repeats.sum())
                print(support_size)
                print(p)
                print(self.shot_num)
                print(self.way_num)
                print(self.query_num)
                print(image.shape)
                print(type(data_query))
                # print(data_query)
                print(len(data_query))
                print([x.shape for x in data_query])
        # print(type(data_shot))
        # print(type(data_query))
        # step = data.shape[0] // (self.way_num*self.shot_num)
        # data_shot = data[::step]
        # data_query = []
        # for i in range(len(data)):
        #     if i % step != 0:
        #         data_query.append(data[i])
        # data_query = torch.stack(data_query).to(self.device)
        # data_shot = data_shot.to(self.device)
        feat_shot, feat_query = self.emb_func(data_shot), self.emb_func(data_query)
        results = self.CPEA(feat_query, feat_shot, self.shot_num)
        results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]
        results = torch.cat(results, dim=0)  # Q x S
        
        pre_query_pred = majority_vote(results, repeats).to('cuda', dtype=torch.long)
        # post_query_y = torch.repeat_interleave(query_target.reshape(-1), repeats).to('cuda', dtype=torch.long)
        
        label = torch.arange(self.way_num).repeat(self.query_num).long().to(self.device)
        # label = torch.arange(self.way_num).repeat_interleave(self.query_num).to(self.device)
        # acc = 100 * accuracy(results.data, label)
        acc = vote_catagorical_acc(label.to('cuda'), pre_query_pred.to('cuda'))
        
        return results, acc

    def set_forward_loss(self, batch):
        if len(batch) == 2:
            data, target = batch
            repeats = None
            support_size = 0
        else:
            data, target, repeats, support_size = batch
        # print(data.shape)
        # input()
        # if torch.cuda.is_available():
        #     data, _ = [_.cuda() for _ in batch]
        # else:
        #     data = batch[0]

        data = rearrange_data(data, self.way_num, self.shot_num + self.query_num)
        p = self.shot_num * self.way_num
        data_shot, data_query = data[:p], data[p:]
        # step = data.shape[0] // (self.way_num*self.shot_num)
        # data_shot = data[::step]
        # data_query = []
        # for i in range(len(data)):
        #     if i % step != 0:
        #         data_query.append(data[i])
        data_query = torch.stack(data_query).to(self.device)
        data_shot = torch.stack(data_shot).to(self.device)
        data_shot = data_shot.to(self.device)
        # print(data_shot.shape)
        # print(data_query.shape)
        # input()
        feat_shot, feat_query = self.emb_func(data_shot), self.emb_func(data_query)
        # print(feat_shot.shape)
        # print(feat_query.shape)
        # input()
        results = self.CPEA(feat_query, feat_shot, self.shot_num)
        results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]
        results = torch.cat(results, dim=0)  # Q x S
        label = torch.arange(self.way_num).repeat(self.query_num).long().to(self.device)
        # label = torch.arange(self.way_num).repeat_interleave(self.query_num).to(self.device)
        # print(results)
        # print(label)
        acc = 100 * accuracy(results.data, label)
        # print("acc:",acc)
        # exit()
        loss = self.loss_func(results, label, self.way_num)
        return results, acc, loss
