# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

from libfewshot_core.utils import accuracy, majority_vote, vote_catagorical_acc
from .metric_model import MetricModel


class ProtoLayer_temperature(nn.Module):
    def __init__(self):
        super(ProtoLayer_temperature, self).__init__()

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
        mode="cos_sim"
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
            )
            ,
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            )
            ,
        }[mode](query_feat, proto_feat)


class MetaBaseline(MetricModel):
    def __init__(self, **kwargs):
        super(MetaBaseline, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer_temperature()
        self.loss_func = nn.CrossEntropyLoss()
        self.temp=nn.Parameter(torch.tensor(10.))
        # self.temp=torch.tensor(10.)
        
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
        try:
            import numpy as np
            import os, datetime
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            HAVE_UMAP = True
            try:
                import umap
            except Exception:
                HAVE_UMAP = False
        except ImportError:
            print("Please install sklearn, plotly and optionally umap-learn to visualize features.")
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

        if shot is None or way is None or query is None:
            # try a simple inference: if total N divisible by (shot+query) guessed via way
            print("visualize_features: shot/way/query not fully specified; please pass them for correct plotting")
            raise ValueError("Insufficient information to infer shot/way/query")
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('plots', exist_ok=True)
        
        block = None
        if shot is not None and way is not None and query is not None:
            block = shot + query
            expected = way * block
            if expected != N:
                print(f"Warning: expected {expected} rows (way*(shot+query)) but got {N}; plotting anyway using inferred indices")
                return
            
        # Save the raw features
        feat_fname = os.path.join('plots', f"featdata_{timestamp}.npz")
        np.savez(feat_fname, 
                 raw_features=feat_np,
                 shot=shot, way=way, query=query,
                 timestamp=timestamp,
                 normalize=normalize,
                 method=method)
        print(f"Saved raw feature data to {feat_fname}")
                
        # # optional normalization
        # if normalize:
        #     from sklearn.preprocessing import normalize as sk_normalize
        #     feat_proc = sk_normalize(feat_np, norm='l2')
        # else:
        #     feat_proc = feat_np

        # # PCA pre-reduction to speed up TSNE/UMAP
        # pca = PCA(n_components=min(50, D), random_state=0)
        # feat_pca = pca.fit_transform(feat_proc)

        # if method == 'umap' and HAVE_UMAP:
        #     reducer = umap.UMAP(n_components=2, random_state=0)
        #     feat_2d = reducer.fit_transform(feat_pca)
        #     method_name = "UMAP"
        # else:
        #     tsne = TSNE(n_components=2, random_state=0, init='pca')
        #     feat_2d = tsne.fit_transform(feat_pca)
        #     method_name = "t-SNE"

        # # Prepare plotting indices
        # supports_idx = []
        # queries_idx = []
        # classes = []
        # if block is not None:
        #     for c in range(way):
        #         start = c * block
        #         s_start = start
        #         s_end = start + shot
        #         q_start = s_end
        #         q_end = s_end + query
        #         supports_idx.extend(list(range(s_start, min(s_end, N))))
        #         queries_idx.extend(list(range(q_start, min(q_end, N))))
        #         classes.extend([c] * (min(s_end, N) - s_start + min(q_end, N) - q_start))
        # else:
        #     # fallback: everything as queries
        #     queries_idx = list(range(N))
            
        # # Create arrays for plotly
        # all_x = []
        # all_y = []
        # all_labels = []  # class labels 0,1,2,...
        # all_types = []   # "support" or "query"
        # all_symbols = []  # circle for support, x for query
        # all_sizes = []    # size of markers
        # all_colors = []   # color index
        
        # # Add supports
        # if supports_idx:
        #     sup = np.array(supports_idx, dtype=int)
        #     if block is not None:
        #         sup_classes = ((sup // block)).astype(int)
        #     else:
        #         sup_classes = np.zeros(len(sup), dtype=int)
                
        #     for cls in np.unique(sup_classes):
        #         sel = sup[sup_classes == cls]
        #         all_x.extend(feat_2d[sel, 0].tolist())
        #         all_y.extend(feat_2d[sel, 1].tolist())
        #         all_labels.extend([f"Class {cls}" for _ in sel])
        #         all_types.extend(["support" for _ in sel])
        #         all_symbols.extend(["circle" for _ in sel])
        #         all_sizes.extend([10 for _ in sel])
        #         all_colors.extend([cls for _ in sel])
        
        # # Add queries
        # if queries_idx:
        #     qry = np.array(queries_idx, dtype=int)
        #     if block is not None:
        #         qry_classes = ((qry // block)).astype(int)
        #     else:
        #         qry_classes = np.zeros(len(qry), dtype=int)
                
        #     for cls in np.unique(qry_classes):
        #         sel = qry[qry_classes == cls]
        #         all_x.extend(feat_2d[sel, 0].tolist())
        #         all_y.extend(feat_2d[sel, 1].tolist())
        #         all_labels.extend([f"Class {cls}" for _ in sel])
        #         all_types.extend(["query" for _ in sel])
        #         all_symbols.extend(["x" for _ in sel])
        #         all_sizes.extend([8 for _ in sel])
        #         all_colors.extend([cls for _ in sel])
        
        # # Create dataframe for plotly
        # import pandas as pd
        # df = pd.DataFrame({
        #     'x': all_x,
        #     'y': all_y,
        #     'label': all_labels,
        #     'type': all_types,
        #     'symbol': all_symbols,
        #     'size': all_sizes,
        #     'color': all_colors
        # })
        
        # # Create plotly figure
        # fig = px.scatter(
        #     df, x='x', y='y', 
        #     color='label', 
        #     symbol='type',
        #     size='size',
        #     size_max=12,
        #     title=f'Feature projection ({method_name})',
        #     labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        #     hover_data=['label', 'type'],
        #     color_discrete_sequence=px.colors.qualitative.Set1
        # )
        
        # # Update layout for better visibility
        # fig.update_layout(
        #     legend=dict(
        #         orientation="h",
        #         yanchor="bottom",
        #         y=1.02,
        #         xanchor="right",
        #         x=1
        #     ),
        #     width=800,
        #     height=800,
        # )
        
        # # Save as interactive HTML
        # html_fname = os.path.join('plots', f"featproj_{timestamp}.html")
        # fig.write_html(html_fname)
        
        # # Also save as static image
        # png_fname = os.path.join('plots', f"featproj_{timestamp}.png")
        # fig.write_image(png_fname)
        
        # print(f"Saved interactive plot to {html_fname}")
        # print(f"Saved static image to {png_fname}")
        
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
        
        # episode_size = image.size(0) // (
        #     self.way_num * (self.shot_num + self.query_num)
        # )
        
        feat = self.emb_func(image)
        
        # self.visualize_features(feat, shot=10, way=5, query=10)
        # input()
        
        support_feat, query_feat, support_target, query_target, query_mask = self.split_by_episode(
            feat, mode=1, repeats=repeats, support_size=support_size
        )

        output = []

        for i in range(len(query_feat)):
            output_per_episode = self.proto_layer(
                query_feat[i].unsqueeze(0), support_feat[i].unsqueeze(0), self.way_num, self.shot_num, query_feat[i].size(0)
            ).reshape(-1, self.way_num)*self.temp
            output.append(output_per_episode)
        output = torch.cat(output, dim=0)
        soft_logits = output.softmax(dim=1)
        pre_query_pred = majority_vote(soft_logits, repeats).to('cuda', dtype=torch.long)
        post_query_y = torch.repeat_interleave(query_target.reshape(-1).cpu(), repeats.cpu()).to('cuda', dtype=torch.long)
        acc = vote_catagorical_acc(query_target.reshape(-1).to('cuda'), pre_query_pred.to('cuda'))
        # acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
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
        emb = self.emb_func(images)
        support_feat, query_feat, support_target, query_target, query_mask = self.split_by_episode(
            emb, mode=1
        )

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)*self.temp
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
