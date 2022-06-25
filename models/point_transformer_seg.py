import torch.nn as nn
from utils.timer import Timer
import torch.nn.functional as F
from point_transformer_lib.point_transformer_ops.point_transformer_modules import PointTransformerBlock, TransitionDown, TransitionUp, BFM_torch


class PointTransformerSemSegmentation(nn.Module):
    def __init__(self, args):
        super().__init__()
        npoints = args.npoints
        dim = [args.fea_dim, 32, 64, 128, 256]
        sampling_ratio = args.downsampling_ratio
        output_dim = args.classes
        
        self.Encoder = nn.ModuleList()
        for i in range(len(dim)-1):
            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i+1], bias=False))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i+1], npoints, sampling_ratio, fast=True))
            self.Encoder.append(PointTransformerBlock(dim[i+1], npoints))

        self.SegDecoder = nn.ModuleList()
        for i in range(len(dim)-1,0,-1):
            if i == len(dim)-1:
                self.SegDecoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.SegDecoder.append(TransitionUp(dim[i+1], dim[i]))
            self.SegDecoder.append(PointTransformerBlock(dim[i], npoints))
        
        self.EdgeDecoder = nn.ModuleList()
        for i in range(len(dim)-1,0,-1):
            if i == len(dim)-1:
                self.EdgeDecoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.EdgeDecoder.append(TransitionUp(dim[i+1], dim[i]))
            self.EdgeDecoder.append(PointTransformerBlock(dim[i], npoints))
        
        self.seg_fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

        self.edge_fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

        self.proj_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim[1], dim[1], kernel_size=1),
        )

        self.BFM = BFM_torch(dim[1], dim[1], args.n_neighbors)
        
        self.seg_refine_fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

    def forward(self, features, gmatrix, idxs):

        xyz = features[...,0:3].contiguous()

        # Encoding period
        l_xyz, l_features = [xyz], [features]
        for i in range(int(len(self.Encoder)/2)):
            if i == 0:
                li_features = self.Encoder[2*i](l_features[i])
                li_xyz = l_xyz[i]
            else:
                li_features, li_xyz = self.Encoder[2*i](l_features[i], l_xyz[i])
            li_features = self.Encoder[2*i+1](li_features, li_xyz)

            l_features.append(li_features)
            l_xyz.append(li_xyz)
            del li_features, li_xyz
        
        e_features = [feature.clone() for feature in l_features]

        # Decoding period
        D_n = int(len(self.SegDecoder)/2)
        for i in range(D_n):
            if i == 0:
                l_features[D_n-i] = self.SegDecoder[2*i](l_features[D_n-i])
                l_features[D_n-i] = self.SegDecoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])
                e_features[D_n-i] = self.EdgeDecoder[2*i](e_features[D_n-i])
                e_features[D_n-i] = self.EdgeDecoder[2*i+1](e_features[D_n-i], l_xyz[D_n-i])
            else:
                l_features[D_n-i] = self.SegDecoder[2*i](l_features[D_n-i+1], l_xyz[D_n-i+1], l_features[D_n-i], l_xyz[D_n-i])
                l_features[D_n-i] = self.SegDecoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])
                e_features[D_n-i] = self.EdgeDecoder[2*i](e_features[D_n-i+1], l_xyz[D_n-i+1], e_features[D_n-i], l_xyz[D_n-i])
                e_features[D_n-i] = self.EdgeDecoder[2*i+1](e_features[D_n-i], l_xyz[D_n-i])

        del l_features[0], l_features[1:], e_features[0], e_features[1:], l_xyz

        # Final output
        seg_features = l_features[0].transpose(1, 2).contiguous()
        edge_features = e_features[0].transpose(1, 2).contiguous()
        seg_preds = self.seg_fc_layer(seg_features)
        edge_preds = self.edge_fc_layer(edge_features)

        seg_refine_features = self.BFM(seg_features, edge_preds, gmatrix, idxs)
        seg_refine_preds = self.seg_refine_fc_layer(seg_refine_features.transpose(1, 2).contiguous())

        seg_embed = F.normalize(self.proj_layer(seg_features), p=2, dim=1)

        return seg_preds, seg_refine_preds, seg_embed, edge_preds
