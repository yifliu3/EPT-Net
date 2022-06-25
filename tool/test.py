import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import IntrADataset
import dataset.data_utils as d_utils

from utils import config
from utils.tools import record_statistics, cal_IoU_Acc_batch, get_contra_loss 


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/IntrA/IntrA_pointtransformer_seg_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/IntrA/IntrA_segmentation.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.ngpus_per_node = len(args.train_gpu)

    logger = get_logger()
    logger.info(args)

    loss_list, loss_seg_list, loss_edge_list, loss_seg_refine_list = [], [], [], []
    iou_list, inner_iou_list, outer_iou_list = [], [], []
    iou_refine_list, inner_iou_refine_list, outer_iou_refine_list = [], [], []

    for fold in args.folds:
        record = main_worker(args.train_gpu, args.ngpus_per_node, test_fold=fold, test_times=args.test_times)
        loss_list.append(record['loss_avg'])
        loss_seg_list.append(record['loss_seg'])
        loss_edge_list.append(record['loss_edge'])
        loss_seg_refine_list.append(record['loss_seg_refine'])
        iou_list.append(record['iou_list'].cpu().numpy())
        inner_iou_list.append(record['inner_iou_list'].cpu().numpy())
        outer_iou_list.append(record['outer_iou_list'].cpu().numpy())
        iou_refine_list.append(record['iou_refine_list'].cpu().numpy())
        inner_iou_refine_list.append(record['inner_iou_refine_list'].cpu().numpy())
        outer_iou_refine_list.append(record['outer_iou_refine_list'].cpu().numpy())

    loss, loss_seg, loss_edge, loss_seg_refine = np.mean(loss_list), np.mean(loss_seg_list), np.mean(loss_edge_list), np.mean(loss_seg_refine_list)
    iou = np.mean(np.stack(iou_list, axis=0), axis=0)
    miou = np.mean(iou)
    inner_miou = np.mean(np.stack(inner_iou_list, axis=0))
    outer_miou = np.mean(np.stack(outer_iou_list, axis=0))
    iou_refine = np.mean(np.stack(iou_refine_list, axis=0), axis=0)
    miou_refine = np.mean(iou_refine)
    inner_miou_refine = np.mean(np.stack(inner_iou_refine_list, axis=0))
    outer_miou_refine = np.mean(np.stack(outer_iou_refine_list, axis=0))
    logger.info("=> Final mIoU is {:.4f}, vIoU is {:.4f}, aIoU is {:.4f}, inner mIoU is {:.4f}, outer mIoU is {:.4f}"\
        .format(miou, iou[0], iou[1], inner_miou, outer_miou))
    logger.info("=> Final mIoU_refine is {:.4f}, vIoU_refine is {:.4f}, aIoU_refine is {:.4f}, inner mIoU refine is {:.4f}, outer mIoU refine is {:.4f}"\
        .format(miou_refine, iou_refine[0], iou_refine[1], inner_miou_refine, outer_miou_refine))
    logger.info("=> Final loss is {:.4f}, loss_seg is {:.4f}, loss_edge is {:.4f}, loss_seg_refine is {:.4f}".format(loss, loss_seg, loss_edge, loss_seg_refine))


def main_worker(gpu, ngpus_per_node, test_fold, test_times=3):
    global args, logger
    logger.info("===============Test fold {}===============".format(test_fold))


    logger.info("=> Creating model ...")
    if args.arch == 'IntrA_pointtransformer_seg_repro':
        from models.point_transformer_seg import PointTransformerSemSegmentation as Model
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    model = Model(args=args).cuda()
    default_ckpt_path = 'exp/IntrA/pointtransformer_seg_repro/'
    if args.test_points == 512:
        ckpt_path = default_ckpt_path
        args.num_edge_neighbor = 4
    elif args.test_points == 1024:
        ckpt_path = default_ckpt_path.replace('exp', 'exp_'+str(args.test_points))
        args.num_edge_neighbor = 6
    else:
        ckpt_path = default_ckpt_path.replace('exp', 'exp_'+str(args.test_points))
        args.num_edge_neighbor = 8
        
    ckpt_path = os.path.join(ckpt_path, 'fold'+str(test_fold), 'model_best.pth')
    ckpt = torch.load(ckpt_path)['state_dict']
    model.load_state_dict(ckpt)


    logger.info("=> Loading data ...")
    if args.data_name == "IntrA":
        val_data = IntrADataset(args.data_root, args.test_points, args.use_uniform_sample, args.use_normals, 
                    test_fold=test_fold, num_edge_neighbor=args.num_edge_neighbor, mode='test', transform=None, test_all=False)
        val_loader = DataLoader(val_data, batch_size=args.batch_size_val,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)


    logger.info("=> Testing ...")
    record_val = validation(val_loader, model)
    print(record_val)

    return record_val


def validation(val_loader, model):
    global args
    model.eval()

    loss_avg_list, loss_seg_list, loss_seg_refine_list, loss_edge_list, loss_contra_list = [], [], [], [], []
    iou_avg_list, inner_iou_avg_list, outer_iou_avg_list = [], [], []
    iou_refine_avg_list, inner_iou_refine_avg_list, outer_iou_refine_avg_list = [], [], []

    for i in range(args.num_votes):
        loss_avg, loss_seg_avg, loss_seg_refine_avg, loss_edge_avg, loss_contra_avg = 0.0, 0.0, 0.0, 0.0, 0.0 
        iou_avg, inner_iou_avg, outer_iou_avg = [], [], []
        iou_refine_avg, inner_iou_refine_avg, outer_iou_refine_avg = [], [], []
        with torch.no_grad():
            for batch_idx, (pts, gts, egts, eweights, gmatrix, idxs) in enumerate(val_loader):
                pts, gts, egts, eweights, gmatrix = pts.cuda(), gts.cuda(), egts.cuda(), eweights.mean(dim=0).cuda(), gmatrix.cuda()
                seg_preds, seg_refine_preds, seg_embed, edge_preds = model(pts, gmatrix, idxs)
                loss_seg = F.cross_entropy(seg_preds, gts, weight=val_loader.dataset.segweights.cuda())
                loss_seg_refine = F.cross_entropy(seg_refine_preds, gts, weight=val_loader.dataset.segweights.cuda())
                loss_edge = F.cross_entropy(edge_preds, egts, weight=eweights)
                loss_contra = get_contra_loss(egts, gts, seg_embed, gmatrix, num_class=args.classes, temp=args.temp)
                loss = loss_seg + args.weight_edge * loss_edge + args.weight_contra * loss_contra + args.weight_refine * loss_seg_refine

                loss_avg += loss.item()
                loss_seg_avg += loss_seg.item()
                loss_seg_refine_avg += loss_seg_refine.item()
                loss_edge_avg += loss_edge.item()
                loss_contra_avg += loss_contra.item()
                
                iou, inner_iou, outer_iou = cal_IoU_Acc_batch(seg_preds, gts, egts)
                iou_avg.append(iou)
                inner_iou_avg.append(inner_iou)
                outer_iou_avg.append(outer_iou)

                iou_refine, inner_iou_refine, outer_iou_refine = cal_IoU_Acc_batch(seg_refine_preds, gts, egts)
                iou_refine_avg.append(iou_refine)
                inner_iou_refine_avg.append(inner_iou_refine)
                outer_iou_refine_avg.append(outer_iou_refine)
            
            dataset_len = len(val_loader.dataset)
            loss_seg_list.append(loss_seg_avg/dataset_len)
            loss_seg_refine_list.append(loss_seg_refine_avg/dataset_len)
            loss_edge_list.append(loss_edge_avg/dataset_len)
            loss_contra_list.append(loss_contra_avg/dataset_len)
            loss_avg_list.append(loss_avg/dataset_len)

            iou_avg_list.append(torch.cat(iou_avg, dim=0).mean(dim=0))
            inner_iou_avg_list.append(torch.cat(inner_iou_avg, dim=0).mean(dim=0))
            outer_iou_avg_list.append(torch.cat(outer_iou_avg, dim=0).mean(dim=0))

            iou_refine_avg_list.append(torch.cat(iou_refine_avg, dim=0).mean(dim=0))
            inner_iou_refine_avg_list.append(torch.cat(inner_iou_refine_avg, dim=0).mean(dim=0))
            outer_iou_refine_avg_list.append(torch.cat(outer_iou_refine_avg, dim=0).mean(dim=0))
    
    record = {}
    record['loss_avg'] = np.mean(loss_avg_list)
    record['loss_seg'] = np.mean(loss_seg_list)
    record['loss_seg_refine'] = np.mean(loss_seg_refine_list)
    record['loss_edge'] = np.mean(loss_edge_list)
    record['loss_contra'] = np.mean(loss_contra_list)
    record['iou_list'] = torch.stack(iou_avg_list, dim=0).mean(dim=0)
    record['inner_iou_list'] = torch.stack(inner_iou_avg_list, dim=0).mean(dim=0)
    record['outer_iou_list'] = torch.stack(outer_iou_avg_list, dim=0).mean(dim=0)
    record['iou_refine_list'] = torch.stack(iou_refine_avg_list, dim=0).mean(dim=0)
    record['inner_iou_refine_list'] = torch.stack(inner_iou_refine_avg_list, dim=0).mean(dim=0)
    record['outer_iou_refine_list'] = torch.stack(outer_iou_refine_avg_list, dim=0).mean(dim=0)
    return record


if __name__ == "__main__":
    main()
