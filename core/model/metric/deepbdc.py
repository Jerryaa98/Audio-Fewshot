import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy, majority_vote, vote_catagorical_acc
from core.model.metric.metric_model import MetricModel


"""
@inproceedings{DeepBDC-CVPR2022,
    title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
    author={Jiangtao Xie and Fei Long and Jiaming Lv and Qilong Wang and Peihua Li}, 
    booktitle={CVPR},
    year={2022}
 }
"""


class ProtoLayer(nn.Module):
    """
    This Proto Layer is partly different from Proto_layer @ ProtoNet
    """
    def __init__(self):
        super(ProtoLayer, self).__init__()
    
    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        t = query_feat.size()[0]
        _, ws, c = support_feat.size()

        # t, wq, c
        # query_feat = query_feat.reshape(t, way_num * query_num, c)
        # t, w, c -- proto_feat
        support_feat = support_feat.reshape(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)

        if shot_num > 1:
            # euclidean, 
            # t, wq, 1, d - t, 1, w, d -> t, wq, w
            return (lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            ))(query_feat, proto_feat)
        else:
            # cosine similarity
            # t, wq, d - t, d, w -> t, wq, w
            return (lambda x, y: torch.matmul(
                # F.normalize(x, p=2, dim=-1),
                # torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # DeepBDC did not normalize the query_feat and proto_feat
                x,
                torch.transpose(y, -1, -2)
            ))(query_feat, proto_feat)


class DeepBDC(MetricModel):
    """ 
    This class is modified from ProtoNet @ ProtoNet
    """
    def __init__(self, **kwargs):
        super(DeepBDC, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()
    
    def visualize_features(self, feat, shot=None, way=None, query=None, normalize=True, method='tsne'):
        """
        Visualize features using a 2D projection. Assumes features are ordered in
        alternating per-class blocks: for each class -> [shot supports, query queries].

        Parameters:
            feat: torch.Tensor or np.ndarray of shape (N, D)
            shot: number of supports per class
            way: number of classes
            query: number of queries per class
            normalize: whether to L2-normalize features before projection
            method: 'tsne' or 'umap'
        """
        print(shot)
        print(way)
        print(query)
        try:
            import numpy as np
            import os, datetime
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            HAVE_UMAP = True
            try:
                import umap
            except Exception:
                HAVE_UMAP = False
        except ImportError:
            print("Please install sklearn and matplotlib (and optionally umap-learn) to visualize features.")
            return

        # Convert to numpy
        if isinstance(feat, torch.Tensor):
            feat_np = feat.detach().cpu().numpy()
        else:
            feat_np = np.asarray(feat)

        N, D = feat_np.shape
        print(f"Visualizing {N} samples with feature dimension {D}")

        # infer shot/way/query from arguments or from model attributes
        if shot is None:
            shot = getattr(self, 'shot_num', None)
        if way is None:
            way = getattr(self, 'way_num', None)
        if query is None:
            query = getattr(self, 'query_num', None)

        print(shot)
        print(way)
        print(query)

        if shot is None or way is None or query is None:
            # try a simple inference: if total N divisible by (shot+query) guessed via way
            print("visualize_features: shot/way/query not fully specified; please pass them for correct plotting")
            raise ValueError("Insufficient information to infer shot/way/query")
        
        block = None
        if shot is not None and way is not None and query is not None:
            block = shot + query
            expected = way * block
            if expected != N:
                print(f"Warning: expected {expected} rows (way*(shot+query)) but got {N}; plotting anyway using inferred indices")
                return
        # optional normalization
        if normalize:
            from sklearn.preprocessing import normalize as sk_normalize
            feat_proc = sk_normalize(feat_np, norm='l2')
        else:
            feat_proc = feat_np

        # PCA pre-reduction to speed up TSNE/UMAP
        pca = PCA(n_components=min(50, D), random_state=0)
        feat_pca = pca.fit_transform(feat_proc)

        if method == 'umap' and HAVE_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=0)
            feat_2d = reducer.fit_transform(feat_pca)
        else:
            tsne = TSNE(n_components=2, random_state=0, init='pca')
            feat_2d = tsne.fit_transform(feat_pca)

        # Prepare plotting indices
        supports_idx = []
        queries_idx = []
        classes = []
        if block is not None:
            for c in range(way):
                start = c * block
                s_start = start
                s_end = start + shot
                q_start = s_end
                q_end = s_end + query
                supports_idx.extend(list(range(s_start, min(s_end, N))))
                queries_idx.extend(list(range(q_start, min(q_end, N))))
                classes.extend([c] * (min(s_end, N) - s_start + max(0, min(q_end, N) - q_start)))
        else:
            # fallback: everything as queries
            queries_idx = list(range(N))

        # Color map
        cmap = plt.get_cmap('tab10')
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(8, 8))

        # Plot supports (circles) and queries (x)
        if supports_idx:
            sup = np.array(supports_idx, dtype=int)
            # determine class per support by integer division within block
            if block is not None:
                sup_classes = ((sup // block)).astype(int)
            else:
                sup_classes = np.zeros(len(sup), dtype=int)
            for cls in np.unique(sup_classes):
                sel = sup[sup_classes == cls]
                plt.scatter(feat_2d[sel, 0], feat_2d[sel, 1], s=60, marker='o',
                            color=cmap(int(cls) % 10), edgecolor='k', label=f'class {cls} support' if cls == 0 else None)

        if queries_idx:
            qry = np.array(queries_idx, dtype=int)
            if block is not None:
                qry_classes = ((qry // block)).astype(int)
            else:
                qry_classes = np.zeros(len(qry), dtype=int)
            for cls in np.unique(qry_classes):
                sel = qry[qry_classes == cls]
                plt.scatter(feat_2d[sel, 0], feat_2d[sel, 1], s=30, marker='x',
                            color=cmap(int(cls) % 10), label=f'class {cls} query' if cls == 0 else None)

        plt.title('Feature projection (supports: o, queries: x)')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.grid(True)
        # build legend (handles supports/queries)
        plt.legend(loc='best', fontsize='small')
        fname = os.path.join('plots', f"featproj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved feature projection to {fname}")
    
    def set_forward(self, batch):
        if len(batch) == 2:
            image, global_target = batch
            repeats = None
            support_size = 0
        else:
            image, global_target, repeats, support_size = batch
        
        image = image.to(self.device)
        
        feat = self.emb_func(image)     # [bsize, c] -- [bsize = t * way_num * (n_s + n_q), c=r*(r+1)/2
        # use tsne to visualize the feature distribution
        self.visualize_features(feat, shot=10, way=5, query=10)
        input()
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1, repeats=repeats, support_size=support_size
        )

        episode_size = support_feat.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )

        # support_feat -- [t, ws, c], query_feat -- [t, wq, c], 
        # support_target -- [t, ws],        query_feat -- [t, wq]
        output = []
        for i in range(len(query_feat)):
            # print(query_feat[i].shape, support_feat[i].shape)
            # input()
            output.append(self.proto_layer(
                query_feat[i].unsqueeze(0).view(1, query_feat[i].size(0), -1), support_feat[i].unsqueeze(0), self.way_num, self.shot_num, self.query_num
            ).reshape(-1, self.way_num))
        output = torch.cat(output, dim=0)
        
        # output expected be [t, wq, w] ---> [t*w*q, w]

        soft_logits = output.softmax(dim=1)
        pre_query_pred = majority_vote(soft_logits, repeats).to('cuda', dtype=torch.long)
        post_query_y = torch.repeat_interleave(query_target.reshape(-1).cpu(), repeats.cpu()).to('cuda', dtype=torch.long)
        acc = vote_catagorical_acc(query_target.reshape(-1).to('cuda'), pre_query_pred.to('cuda'))
        # acc = accuracy(output, query_target.reshape(-1))
        

        
        return output, acc


    def set_forward_loss(self, batch):
        if len(batch) == 2:
            images, global_target = batch
            repeats = None
            support_size = 0
        else:
            images, global_target, repeats, support_size = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.proto_layer(
            query_feat[0].unsqueeze(0), support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(-1, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc, loss
