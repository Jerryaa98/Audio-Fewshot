# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/BertinettoHTV19,
  author    = {Luca Bertinetto and
               Jo{\\~{a}}o F. Henriques and
               Philip H. S. Torr and
               Andrea Vedaldi},
  title     = {Meta-learning with differentiable closed-form solvers},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HyxnZh0ct7}
}
https://arxiv.org/abs/1805.08136

Adapted from https://github.com/kjunelee/MetaOptNet.
"""
import torch
from torch import nn

from libfewshot_core.utils import accuracy, majority_vote, vote_catagorical_acc
from .meta_model import MetaModel


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert A.dim() == 3, "A must be a 3-D Tensor."
    assert B.dim() == 3, "B must be a 3-D Tensor."
    assert A.size(0) == B.size(0) and A.size(2) == B.size(
        2
    ), "A and B must have the same batch size and feature dimension."

    return torch.bmm(A, B.transpose(1, 2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.

    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).to(b_mat.device)
    b_inv = torch.linalg.solve(b_mat, id_matrix)

    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicie = torch.zeros(indices.size() + torch.Size([depth])).to(
        indices.device
    )
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicie = encoded_indicie.scatter_(1, index, 1)

    return encoded_indicie


class R2D2Layer(nn.Module):
    def __init__(self):
        super(R2D2Layer, self).__init__()
        self.register_parameter("alpha", nn.Parameter(torch.tensor([1.0])))
        self.register_parameter("beta", nn.Parameter(torch.tensor([0.0])))
        self.register_parameter("gamma", nn.Parameter(torch.tensor([50.0])))

    def forward(self, way_num, shot_num, query, support, support_target):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        support_target = support_target.squeeze()

        assert query.dim() == 3, "query must be a 3-D Tensor."
        assert support.dim() == 3, "support must be a 3-D Tensor."
        assert query.size(0) == support.size(0) and query.size(2) == support.size(
            2
        ), "query and support must have the same batch size and feature dimension."
        assert (
            n_support == way_num * shot_num
        ), "n_support must be equal to way_num * shot_num."  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(
            support_target.view(tasks_per_batch * n_support), way_num
        )
        support_labels_one_hot = support_labels_one_hot.view(
            tasks_per_batch, n_support, way_num
        )

        id_matrix = (
            torch.eye(n_support)
            .expand(tasks_per_batch, n_support, n_support)
            .to(query.device)
        )

        # Compute the dual form solution of the ridge regression.
        # W = X^T(X X^T - lambda * I)^(-1) Y
        ridge_sol = computeGramMatrix(support, support) + self.gamma * id_matrix
        ridge_sol = binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
        ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

        # Compute the classification score.
        # score = W X
        logit = torch.bmm(query, ridge_sol)
        logit = self.alpha * logit + self.beta
        return logit, ridge_sol


class R2D2(MetaModel):
    def __init__(self, **kwargs):
        super(R2D2, self).__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = R2D2Layer()
        self._init_network()
        
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
        if len(batch) == 2:
            image, global_target = batch
            repeats = None
            support_size = 0
        else:
            image, global_target, repeats, support_size = batch   # unused global_target
        
        image = image.to(self.device)

        feat = self.emb_func(image)
        
        # self.visualize_features(feat, shot=1, way=5, query=10)
        # input()
        
        support_feat, query_feat, support_target, query_target, query_mask = self.split_by_episode(
            feat, mode=1, support_size=support_size, repeats=repeats
        )

        episode_size = support_feat.size(0)

        output = []
        for i in range(episode_size):
            output_per_episode, weight = self.classifier(
                self.way_num, self.shot_num, \
                    query_feat[i].unsqueeze(0) if query_feat[i].ndim == 2 else query_feat[i],\
                         support_feat, support_target
            )
            output.append(output_per_episode)
        
        output = torch.stack(output, dim=0)
        output = output.contiguous().reshape(-1, self.way_num)
        output = output.softmax(dim=-1)
        pre_query_pred = majority_vote(output, repeats).to('cuda', dtype=torch.long)
        post_query_y = torch.repeat_interleave(query_target.reshape(-1).cpu(), repeats.cpu()).to('cuda', dtype=torch.long)
        acc = vote_catagorical_acc(query_target.reshape(-1).to('cuda'), pre_query_pred.to('cuda'))
        # acc = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        if len(batch) == 2:
            image, global_target = batch
            repeats = None
            support_size = 0
        else:
            image, global_target, repeats, support_size = batch   # unused global_target
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target, query_mask = self.split_by_episode(
            feat, mode=1
        )
        output, weight = self.classifier(
            self.way_num, self.shot_num, query_feat, support_feat, support_target
        )

        output = output.contiguous().reshape(-1, self.way_num)
        loss = self.loss_func(output, query_target.contiguous().reshape(-1))
        acc = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))
        return output, acc, loss

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
