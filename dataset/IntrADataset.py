import os
import numpy as np
import warnings
import torch
import copy
from pathlib import Path


from torch.utils.data import Dataset
from point_transformer_lib.point_transformer_ops.point_transformer_utils import FPS


warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def load_txtfile(path):
    file_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            file_list.append(line)
    return file_list


def load_adfile(path):
    points = []
    normals = []
    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            s_line = line.split()
            points.append([float(s_line[0]), float(s_line[1]), float(s_line[2])])
            normals.append([float(s_line[3]), float(s_line[4]), float(s_line[5])])
            labels.append(int(s_line[6]))
    return points, normals, labels


def load_gdfile(path):
    matrix = []
    with open(path, 'r') as f:
        for line in f.readlines():
            s_line = line.split()
            numbers_float = list(map(float, s_line))
            matrix.append(numbers_float)
    return np.array(matrix).astype(np.float16)


class IntrADataset(Dataset):
    def __init__(self, data_root, num_points, use_uniform_sample, use_normals, test_fold, \
                num_edge_neighbor, mode='train', transform=None):
        self.data_root = Path(data_root)
        self.npoints = num_points
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.test_fold = test_fold
        self.split_path = self.data_root / "split/seg/"
        self.mode = mode
        self.num_edge_neighbor = num_edge_neighbor
        self.transform =transform
        self.data_path = []
        self.geo_data_path = []
        self.whole_data_path = []
        self.preloaded_gmatrix = []

        for i in range(5):
            file_path = "annSplit_%d.txt" % i
            geo_file_path = "geoSplit_%d.txt" % i
            self.whole_data_path.extend(load_txtfile(self.split_path/file_path))
            if mode == 'train' and i != test_fold:
                self.data_path.extend(load_txtfile(self.split_path/file_path))
                self.geo_data_path.extend(load_txtfile(self.split_path/geo_file_path))
            if mode == 'test' and i == test_fold:
                self.data_path.extend(load_txtfile(self.split_path/file_path))
                self.geo_data_path.extend(load_txtfile(self.split_path/geo_file_path))
            else:
                continue
        
        segweights = np.zeros(2)
        label_list = []
        for path in self.whole_data_path:
            _, _, labels = load_adfile(self.data_root/path)
            label_list.extend(labels)
        label_list[label_list==2] == 1
        tmp, _ = np.histogram(label_list, range(3))
        segweights += tmp
        segweights = segweights.astype(np.float32)
        segweights = segweights / np.sum(segweights)
        self.segweights = torch.from_numpy(np.power(np.amax(segweights)/segweights, 1 / 3.0))

        # preload geodesic matrix
        for path in self.geo_data_path:
            gmatrix = load_gdfile(self.data_root/path)
            self.preloaded_gmatrix.append(gmatrix)
    
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        points, normals, labels = load_adfile(self.data_root/self.data_path[index])
        gmatrix = self.preloaded_gmatrix[index].astype(np.float32)

        num_avail_points = len(points)
        points, normals, labels = np.array(points), np.array(normals), np.array(labels)
        labels[labels==2] = 1
        point_idxs = range(num_avail_points)
        npoints = self.npoints

        if num_avail_points >= npoints:
            if self.uniform:
                points_cuda = torch.from_numpy(points).float().unsqueeze(0)
                selected_points_idxs = FPS(points_cuda, npoints).squeeze().numpy().astype(np.int64)
        else:
            if self.uniform:
                points_cuda = torch.from_numpy(points).float().unsqueeze(0)
                scale = npoints // num_avail_points
                extra = npoints % num_avail_points
                extra_idxs = FPS(points_cuda, extra).squeeze().numpy().astype(np.int64)
                selected_points_idxs = np.concatenate((np.array(list(point_idxs)*scale).astype(np.int64), extra_idxs))

        selected_points = points[selected_points_idxs]
        selected_normals = normals[selected_points_idxs]
        selected_gmatrix = gmatrix[selected_points_idxs, :][:, selected_points_idxs]
        selected_labels = labels[selected_points_idxs]

        selected_points = pc_normalize(selected_points)
        if self.use_normals:
            selected_points = np.concatenate((selected_points, selected_normals), axis=1)
        if self.transform is not None:
            selected_points = self.transform(selected_points).float()
        else:
            selected_points = torch.from_numpy(selected_points).float()
        selected_labels = torch.from_numpy(selected_labels).long()
        selected_gmatrix = torch.from_numpy(selected_gmatrix).float()
        
        selected_edge_labels, edgeweights = self.get_edge_label(selected_points_idxs, selected_labels, selected_gmatrix, self.num_edge_neighbor)

        return selected_points, selected_labels, selected_edge_labels, edgeweights, selected_gmatrix, selected_points_idxs
    

    def get_edge_label(self, idxs, labels, gmatrix, k): 
        _, indices, reverse_indices = np.unique(idxs, return_index=True, return_inverse=True)
        unique_labels = labels[indices]
        unique_gmatrix = gmatrix[indices, :][:, indices]
        edge_labels = torch.zeros(unique_labels.shape[0])
        idxs_neighbor = unique_gmatrix.argsort(dim=-1)[:, :k] # (N, K)
        gts_neighbor = torch.gather(unique_labels[None, :].repeat(idxs_neighbor.shape[0], 1), 1, idxs_neighbor) # (N, K)
        gts_neighbor_sum = gts_neighbor.sum(dim=-1)
        edge_mask = torch.logical_and(gts_neighbor_sum!=0, gts_neighbor_sum!=k)
        edge_labels[edge_mask] = 1
        edge_labels = edge_labels[reverse_indices]
        edgeweights = torch.histc(edge_labels, bins=2, min=0, max=1)
        edgeweights = edgeweights / torch.sum(edgeweights)
        edgeweights = (edgeweights.max() / edgeweights) ** (1/3)
        edge_labels = edge_labels.long()
        return edge_labels, edgeweights


if __name__ == '__main__':
    import torch
    np.random.seed(666)

    data = IntrADataset(data_root='/home/yifliu3/data/IntrA/', num_points=512, use_normals=True,\
                test_fold=0, num_edge_neighbor=4, mode='train', use_uniform_sample=True)

    data_val = IntrADataset(data_root='/home/yifliu3/data/IntrA/', num_points=512, use_normals=True,\
                test_fold=0, num_edge_neighbor=4, mode='test', use_uniform_sample=True)
    
    DataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)
    DataLoader_val = torch.utils.data.DataLoader(data_val, batch_size=8, shuffle=False)
    for point, label in DataLoader_val:
        print(torch.sum(label==2, dim=-1))
        
        