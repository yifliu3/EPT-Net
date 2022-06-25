from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import point_transformer_ops.point_transformer_utils as pt_utils



class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.prev_linear = nn.Linear(dim, dim)
        self.k = k
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        self.final_linear = nn.Linear(dim, dim)

    def forward(self, x, pos):
        # queries, keys, values

        x_pre = x
        knn_idx = pt_utils.kNN_torch(pos, pos, self.k)
        knn_xyz = pt_utils.index_points(pos, knn_idx)

        q = self.to_q(x)
        k = pt_utils.index_points(self.to_k(x), knn_idx)
        v = pt_utils.index_points(self.to_v(x), knn_idx)
        
        pos_enc = (pos[:, :, None] - knn_xyz).permute(0, 3, 1, 2).contiguous()
        for i, layer in enumerate(self.pos_mlp): pos_enc = layer(pos_enc)
        pos_enc = pos_enc.permute(0, 2, 3, 1).contiguous()
        
        attn = (q[:, :, None] - k + pos_enc).permute(0, 3, 1, 2).contiguous()
        for i, layer in enumerate(self.attn_mlp): attn = layer(attn)
        attn = attn.permute(0, 2, 3, 1).contiguous()
        
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)

        agg = einsum('b i j d, b i j d -> b i d', attn, v+pos_enc)
        agg = self.final_linear(agg) + x_pre

        return agg


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio, fast=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        self.fast = fast
        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, x, p1):
        """
        inputs
            x: (B, N, in_channels) shaped torch Tensor (A set of feature vectors)
            p1: (B, N, 3) shaped torch Tensor (3D coordinates)
        outputs
            y: (B, M, out_channels) shaped torch Tensor
            p2: (B, M, 3) shaped torch Tensor
        M = N * sampling ratio
        """
        B, N, _ = x.shape
        M = int(N * self.sampling_ratio)

        # 1: Farthest Point Sampling
        p1_flipped = p1.transpose(1, 2).contiguous()
        p2 = (
            pt_utils.gather_operation(
                p1_flipped, pt_utils.farthest_point_sample(p1, M)
            )
            .transpose(1, 2)
            .contiguous()
        )  # p2: (B, M, 3)

        # 2: kNN & MLP
        knn_fn = pt_utils.kNN_torch if self.fast else pt_utils.kNN
        neighbors = knn_fn(p2, p1, self.k)  # neighbors: (B, M, k)

        # 2-1: Apply MLP onto each feature
        x_flipped = x.transpose(1, 2).contiguous()
        mlp_x = self.mlp(x_flipped).transpose(1, 2).contiguous() # mlp_x: (B, N, out_channels)

        # 2-2: Extract features based on neighbors
        features = pt_utils.index_points(mlp_x, neighbors)  # features: (B, M, k, out_channels)

        # 3: Local Max Pooling
        y = torch.max(features, dim=2)[0]  # y: (B, M, out_channels)

        return y, p2


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )
        self.lateral_mlp = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, p1, x2, p2):
        """
            x1: (B, N, in_channels) torch.Tensor
            p1: (B, N, 3) torch.Tensor
            x2: (B, M, out_channels) torch.Tensor
            p2: (B, M, 3) torch.Tensor
        Note that N is smaller than M because this module upsamples features.
        """
        x1 = self.up_mlp(x1.transpose(1, 2).contiguous())
        dist, idx= pt_utils.three_nn(p2, p1)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pt_utils.three_interpolate(
            x1, idx, weight
        )
        x2 = self.lateral_mlp(x2.transpose(1, 2).contiguous())
        y = interpolated_feats + x2
        return y.transpose(1, 2).contiguous()


class BFM_torch(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.GCN_1 = GCN(in_channels, out_channels)
        self.GCN_2 = GCN(out_channels, out_channels)


    def forward(self, seg_features, edge_preds, gmatrix, idxs):
        # import pdb; pdb.set_trace()
        B = idxs.shape[0]
        seg_features = seg_features.transpose(1, 2).contiguous()
        refined_features_list = []

        # construct separated topology graph
        for i in range(B):
            edge_preds_this = edge_preds[i, ...].argmax(0) # N
            gmatrix_this = gmatrix[i, ...] # N, N
            idxs_this = idxs[i, ...]
            features_this = seg_features[i, ...]

            # get unique attributes
            unique_idxs, indices, inverse_indices = np.unique(idxs_this, return_index=True, return_inverse=True)
            gmatrix_unique = gmatrix_this[indices, :][:, indices]
            features_unique = features_this[indices, :]
            edge_preds_unique = edge_preds_this[indices]

            # find neighbors based on the geodesic distance
            adjacency_matrix = torch.zeros(gmatrix_unique.shape).cuda()
            neighbor_idxs_matrix = torch.argsort(gmatrix_unique, axis=-1)[:, 0:self.n_neighbors]
            seq_idxs_matrix = torch.arange(neighbor_idxs_matrix.shape[0])[:, None]
            adjacency_matrix[seq_idxs_matrix, neighbor_idxs_matrix] = 1
            adjacency_matrix[neighbor_idxs_matrix, seq_idxs_matrix] = 1

            # cutoff connections and check
            # edge points are disconnected with other points
            edge_index = torch.nonzero(edge_preds_unique==1, as_tuple=True)[0]
            non_edge_index = torch.nonzero(edge_preds_unique==0, as_tuple=True)[0]
            adjacency_matrix[edge_index, :] = torch.zeros(edge_preds_unique.shape).cuda()
            # nonedge points are disconnected with edge points and other rules
            if non_edge_index.shape[0] == 0:
                adjacency_matrix = torch.diag(torch.ones(adjacency_matrix.shape[0])).cuda()
            else:
                nonedge_adj = adjacency_matrix[non_edge_index, :]
                nonedge_geo = gmatrix_unique[non_edge_index, :]
                edge_mask = torch.zeros(nonedge_adj.shape[-1]).cuda()
                edge_mask[edge_index] = 1
                edge_mask = edge_mask[None, :].repeat(nonedge_adj.shape[0], 1)
                tmp_fill = (torch.ones_like(edge_mask)*1000).cuda()
                maxgeolimit, _ = torch.where(torch.logical_and(nonedge_adj==1, edge_mask==1), nonedge_geo, tmp_fill).min(dim=-1)
                zero_mask = torch.zeros_like(nonedge_adj).cuda()
                nonedge_adj = torch.where(nonedge_geo > maxgeolimit[:, None], zero_mask, nonedge_adj)
                nonedge_adj[:, edge_index] = 0
                adjacency_matrix[non_edge_index, :] = nonedge_adj
                adjacency_matrix_trans = adjacency_matrix.transpose(0, 1)
                adjacency_matrix = torch.logical_or(adjacency_matrix, adjacency_matrix_trans)
                adjacency_matrix = adjacency_matrix.type(torch.cuda.FloatTensor)
            
            # GCN layer
            refined_features_unique = self.GCN_1(features_unique, adjacency_matrix)
            refined_features_unique = self.GCN_2(refined_features_unique, adjacency_matrix)
            refined_features = refined_features_unique[inverse_indices, :]
            refined_features_list.append(refined_features)
        
        refined_features = torch.stack(refined_features_list, dim=0)
        refined_features = refined_features + seg_features
        
        return refined_features


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.trans_msg = nn.Linear(in_channels, out_channels)
        self.nonlinear = nn.ReLU()
    
    def forward(self, features, adj_matrix):
        N, C = features.shape
        identity = features
        features = self.nonlinear(self.trans_msg(features))
        row_degree = torch.sum(adj_matrix, dim=-1, keepdim=True) # (N, 1)
        col_degree = torch.sum(adj_matrix, dim=-2, keepdim=True) # (1, N)
        degree = torch.mm(torch.sqrt(row_degree), torch.sqrt(col_degree)) # (N, N)
        if degree[degree==0].shape[0] != 0:
            return identity
        else:
            refined_features = torch.sparse.mm(adj_matrix / degree, features) # (B, N, C)
            refined_features = refined_features + identity
            return refined_features