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
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import IntrADataset
import dataset.data_utils as d_utils

from utils import config
from utils.tools import cal_IoU_Acc_batch, get_contra_loss, record_statistics


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

    iou_list = []
    for fold in args.folds:
        best_iou_list = main_worker(args.train_gpu, args.ngpus_per_node, test_fold=fold)
        iou_list.append(best_iou_list)
    iou = torch.stack(iou_list, dim=0).mean(dim=0)
    miou = torch.mean(iou)
    logger.info("=> Final mIoU is {:.4f}, vIoU is {:.4f}, aIoU is {:.4f}".format(miou, iou[0], iou[1]))


def main_worker(gpu, ngpus_per_node, test_fold):
    global args, logger, writer

    fold_path = os.path.join(args.save_path, "fold{}".format(test_fold))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    writer = SummaryWriter(fold_path)
    logger.info("===============test fold {}===============".format(test_fold))

    logger.info("=====================> Creating model ...")
    if args.arch == 'IntrA_pointtransformer_seg_repro':
        from models.point_transformer_seg import PointTransformerSemSegmentation as Model
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    model = Model(args=args).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.base_lr*0.01)
    logger.info("=> Features:{}, Classes: {}".format(args.fea_dim, args.classes))

    logger.info("=====================> Loading data ...")
    if args.data_name == "IntrA":
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
                d_utils.PointcloudRandomInputDropout(),
            ]
        )
        train_data = IntrADataset(args.data_root, args.sample_points, args.use_uniform_sample, args.use_normals, 
                    test_fold=test_fold, num_edge_neighbor=args.num_edge_neighbor, mode='train', transform=train_transforms)
        train_loader = DataLoader(train_data, batch_size=args.batch_size_train,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_data = IntrADataset(args.data_root, args.sample_points, args.use_uniform_sample, args.use_normals, 
                    test_fold=test_fold, num_edge_neighbor=args.num_edge_neighbor, mode='test', transform=None)
        val_loader = DataLoader(val_data, batch_size=args.batch_size_val,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    logger.info("=> Loaded {} training samples, {} testing samples".format(len(train_data), len(val_data)))

    logger.info("=====================> Training loop...")
    best_miou = 0
    best_iou_list = []
    for epoch in range(args.start_epoch, args.epochs):
        record_train = train_one_epoch(train_loader, model, optimizer)
        writer = record_statistics(writer, record_train, mode='train', epoch=epoch)

        scheduler.step()

        if args.evaluate and (epoch % args.eval_freq == 0):
            is_best = False
            record_val = val_one_epoch(val_loader, model)
            writer = record_statistics(writer, record_val, mode='val', epoch=epoch)

            iou_list = record_val['iou_list']
            miou_val = torch.mean(iou_list)
            is_best = miou_val > best_miou
            best_miou = miou_val if is_best else best_miou
            best_iou_list = iou_list if is_best else best_iou_list
            filename = os.path.join(fold_path, 'model_last.pth')
            torch.save({'epoch': epoch, 'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict(), 'best_miou': best_miou, 
                        'best_viou': best_iou_list[0], 'best_aiou': best_iou_list[1]}, filename)
            if is_best:
                logger.info('Epoch{}: best validation mIoU updated to {:.4f}, vIoU is {:.4f} and aIoU is {:.4f}'.format(
                    epoch, best_miou, best_iou_list[0], best_iou_list[1]))
                shutil.copyfile(filename, os.path.join(fold_path, 'model_best.pth'))

    writer.close()
    logger.info("===============test fold {} training done===============\n\
        Best mIoU is {:.4f}, vIoU is {:.4f}, aIoU is {:.4f}".format(test_fold, best_miou, best_iou_list[0], best_iou_list[1]))
    return best_iou_list


def train_one_epoch(train_loader, model, optimizer):
    global args
    model.train()

    loss_avg, loss_seg_avg, loss_seg_refine_avg, loss_edge_avg, loss_contra_avg = 0.0, 0.0, 0.0, 0.0, 0.0
    iou_list, iou_refine_list = [], []
    for batch_i, (pts, gts, egts, eweights, gmatrix, idxs) in enumerate(train_loader):
        pts, gts, egts, eweights, gmatrix = pts.cuda(), gts.cuda(), egts.cuda(), eweights.mean(dim=0).cuda(), gmatrix.cuda()
        seg_preds, seg_refine_preds, seg_embed, edge_preds = model(pts, gmatrix, idxs)
        loss_seg = F.cross_entropy(seg_preds, gts, weight=train_loader.dataset.segweights.cuda())
        loss_seg_refine = F.cross_entropy(seg_refine_preds, gts, weight=train_loader.dataset.segweights.cuda())
        loss_edge = F.cross_entropy(edge_preds, egts, weight=eweights)
        loss_contra = get_contra_loss(egts, gts, seg_embed, gmatrix, num_class=args.classes, temp=args.temp)
        loss = loss_seg + args.weight_refine * loss_seg_refine + args.weight_edge * loss_edge + args.weight_contra * loss_contra
        
        loss_avg += loss.item()
        loss_seg_avg += loss_seg.item()
        loss_seg_refine_avg += loss_seg_refine.item()
        loss_edge_avg += loss_edge.item()
        loss_contra_avg += loss_contra.item()
        iou_list.append(cal_IoU_Acc_batch(seg_preds, gts))
        iou_refine_list.append(cal_IoU_Acc_batch(seg_refine_preds, gts))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    record = {}
    dataset_len = len(train_loader.dataset)
    record['loss_all'] = loss_avg / dataset_len
    record['loss_seg'] = loss_seg_avg / dataset_len
    record['loss_seg_refine'] = loss_seg_refine_avg / dataset_len
    record['loss_edge'] = loss_edge_avg / dataset_len
    record['loss_contra'] = loss_contra_avg / dataset_len
    record['iou_list'] = torch.cat(iou_list, dim=0).mean(dim=0)
    record['iou_refine_list'] = torch.cat(iou_refine_list, dim=0).mean(dim=0)
    return record


def val_one_epoch(val_loader, model):
    global args
    model.eval()

    loss_avg_list, loss_seg_avg_list, loss_seg_refine_avg_list, loss_edge_avg_list, loss_contra_avg_list = [], [], [], [], []
    iou_avg_list, iou_refine_avg_list = [], []
    for i in range(args.num_votes):
        loss_avg, loss_seg_avg, loss_seg_refine_avg, loss_edge_avg, loss_contra_avg = 0.0, 0.0, 0.0, 0.0, 0.0
        iou_avg, iou_refine_avg = [], []
        with torch.no_grad():
            for batch_idx, (pts, gts, egts, eweights, gmatrix, idxs) in enumerate(val_loader):
                pts, gts, egts, eweights, gmatrix = pts.cuda(), gts.cuda(), egts.cuda(), eweights.mean(dim=0).cuda(), gmatrix.cuda()
                seg_preds, seg_refine_preds, seg_embed, edge_preds = model(pts, gmatrix, idxs)
                loss_seg = F.cross_entropy(seg_preds, gts, weight=val_loader.dataset.segweights.cuda())
                loss_seg_refine = F.cross_entropy(seg_refine_preds, gts, weight=val_loader.dataset.segweights.cuda())
                loss_edge = F.cross_entropy(edge_preds, egts, weight=eweights)
                loss_contra = get_contra_loss(egts, gts, seg_embed, gmatrix, num_class=args.classes, temp=args.temp)
                loss = loss_seg + args.weight_refine * loss_seg_refine + args.weight_edge * loss_edge + args.weight_contra * loss_contra

                loss_avg += loss.item()
                loss_seg_avg += loss_seg.item()
                loss_seg_refine_avg += loss_seg_refine.item()
                loss_edge_avg += loss_edge.item()
                loss_contra_avg += loss_contra.item()
                iou_avg.append(cal_IoU_Acc_batch(seg_preds, gts))
                iou_refine_avg.append(cal_IoU_Acc_batch(seg_refine_preds, gts))

            dataset_len = len(val_loader.dataset)
            loss_avg_list.append(loss_avg / dataset_len)
            loss_seg_avg_list.append(loss_seg_avg / dataset_len)
            loss_seg_refine_avg_list.append(loss_seg_refine_avg / dataset_len)
            loss_edge_avg_list.append(loss_edge_avg / dataset_len)
            loss_contra_avg_list.append(loss_contra_avg / dataset_len)
            iou_avg_list.append(torch.cat(iou_avg, dim=0).mean(dim=0))
            iou_refine_avg_list.append(torch.cat(iou_refine_avg, dim=0).mean(dim=0))
    
    record = {}
    record['loss_all'] = np.mean(loss_avg_list)
    record['loss_seg'] = np.mean(loss_seg_avg_list)
    record['loss_seg_refine'] = np.mean(loss_seg_refine_avg_list)
    record['loss_edge'] = np.mean(loss_edge_avg_list)
    record['loss_contra'] = np.mean(loss_contra_avg_list)
    record['iou_list'] = torch.stack(iou_avg_list, dim=0).mean(dim=0)
    record['iou_refine_list'] = torch.stack(iou_refine_avg_list, dim=0).mean(dim=0)
    return record




if __name__ == "__main__":
    main()
