import torch


def cal_IoU_Acc_batch(preds, labels):
    B, C, _ = preds.shape
    preds = torch.argmax(preds, dim=1)
    IoU = torch.zeros(B, C).cuda()
    for j in range(C):
        tmp_and_num = torch.sum(torch.bitwise_and(preds==j, labels==j), dim=1, keepdim=False)
        tmp_or_num = torch.sum(torch.bitwise_or(preds==j, labels==j), dim=1, keepdim=False)
        IoU[..., j] = tmp_and_num / tmp_or_num

    return IoU


def record_statistics(writer, record, mode, epoch):
    for k, v in record.items():
        if k == 'iou_list':
            writer.add_scalar('miou_{}'.format(mode), v.mean(), epoch)
            writer.add_scalar('viou_{}'.format(mode), v[0], epoch)
            writer.add_scalar('aiou_{}'.format(mode), v[1], epoch)
        elif k == 'iou_refine_list':
            writer.add_scalar('miou_refine_{}'.format(mode), v.mean(), epoch)
            writer.add_scalar('viou_refine_{}'.format(mode), v[0], epoch)
            writer.add_scalar('aiou_refine_{}'.format(mode), v[1], epoch)
        else:
            writer.add_scalar(k+'_{}'.format(mode), v, epoch)
    return writer


def get_contra_loss(egts, gts, seg_emb, gmatrix, num_class, temp):
    B, D, N = seg_emb.shape
    seg_emb = seg_emb.transpose(1, 2).contiguous()
    detach_emb = seg_emb.clone().detach()
    loss_contra = 0.0
    # import pdb; pdb.set_trace()
    for i in range(B):
        egts_this, gts_this, seg_emb_this, gmatrix_this = egts[i, :], gts[i, :], seg_emb[i, ...], gmatrix[i, ...]
        detach_emb_this = detach_emb[i, ...]
        nonedge_idxs = torch.nonzero(egts_this==0, as_tuple=True)[0]
        edge_idxs = torch.nonzero(egts_this==1, as_tuple=True)[0]

        edge_gts = gts_this[edge_idxs]
        edge_emb = seg_emb_this[edge_idxs, :]
        nonedge_gts = gts_this[nonedge_idxs]
        nonedge_detach_emb = detach_emb_this[nonedge_idxs, :]
        nonedge_gmatrix = gmatrix_this[nonedge_idxs, :]

        keys = []
        for j in range(num_class):
            jclass_nonedge_emb = nonedge_detach_emb[nonedge_gts==j, :]
            jclass_nonedge_gmatrix = nonedge_gmatrix[nonedge_gts==j, ][:, edge_idxs].mean(dim=-1)
            jclass_nonedge_neighbor_idxs = jclass_nonedge_gmatrix.argsort(dim=-1)[ :16]
            keys.append(jclass_nonedge_emb[jclass_nonedge_neighbor_idxs, :])
        
        for j in range(num_class):
            positive_key = keys[j].view(-1, D) # (B, D)
            negative_key = torch.stack(keys[:j] + keys[j+1:], dim=0).view(-1, D) #(C-1*B, D)
            jclass_emb = edge_emb[edge_gts==j, :] # (M, D)
            jclass_pos_logits = - torch.mm(jclass_emb, positive_key.transpose(0, 1)) / temp # (M, B)
            jclass_neg_logits = torch.log(torch.sum(torch.exp(torch.mm(jclass_emb, negative_key.transpose(0, 1)) / temp), dim=-1)) # M
            loss_contra_tmp = torch.mean(jclass_pos_logits + jclass_neg_logits[:, None])
            loss_contra += loss_contra_tmp
    
    loss_contra = loss_contra / (B * num_class)
    return loss_contra
