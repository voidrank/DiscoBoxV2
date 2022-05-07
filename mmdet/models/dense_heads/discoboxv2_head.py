# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiscoBox. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import mmcv
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.core import multi_apply
from mmcv.ops.roi_align import RoIAlign
from mmcv import tensor2imgs
from mmcv.runner.fp16_utils import force_fp32
from ..builder import build_loss, HEADS

from torch.cuda.amp import autocast


from mmcv.cnn import bias_init_with_prob, ConvModule
import numpy as np


def relu_and_l2_norm_feat(feat, dim=1):
    feat = F.relu(feat, inplace=True)
    feat_norm = ((feat ** 2).sum(dim=dim, keepdim=True) + 1e-6) ** 0.5
    feat = feat / (feat_norm + 1e-6)
    return feat

#@autocast(enabled=False)
def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


#@autocast(enabled=False)
def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels



#@autocast(enabled=False)
def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

#@autocast(enabled=False)
def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


#@autocast(enabled=False)
def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1).float()
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

#@autocast(enabled=False)
def mil_loss(loss_func, input, _, target):
    row_labels = target.max(1)[0]
    column_labels = target.max(2)[0]

    row_input = input.max(1)[0]
    column_input = input.max(2)[0]

    loss = loss_func(column_input, column_labels) +\
           loss_func(row_input, row_labels)

    return loss



#@autocast(enabled=False)
def vis_seg(img_tensor, cur_mask, img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=0):
    img = tensor2imgs(img_tensor, **img_norm_cfg)[0]

    h, w = img.shape[:2]

    cur_mask = cur_mask.cpu().numpy()
    cur_mask = mmcv.imresize(cur_mask, (w, h))
    cur_mask = (cur_mask > 0.5)
    cur_mask = cur_mask.astype(np.int32)

    seg_show = img.copy()
    color_mask = np.random.randint(
        0, 256, (1, 1, 3), dtype=np.uint8)
    cur_mask_bool = cur_mask.astype(np.bool)
    seg_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

    mmcv.imwrite(seg_show, '{}/_{}.jpg'.format(save_dir, data_id))



class MeanField(nn.Module):

    # feature map (RGB)
    # B = #num of object
    # shape of [N 3 H W]

    #@autocast(enabled=False)
    def __init__(self, feature_map, kernel_size=3, require_grad=False, theta0=0.5, theta1=30, theta2=10, alpha0=3,
                 iter=20, base=0.45, gamma=0.01):
        super(MeanField, self).__init__()
        self.require_grad = require_grad
        self.kernel_size = kernel_size
        with torch.no_grad():
            self.unfold = torch.nn.Unfold(kernel_size, stride=1, padding=kernel_size//2)
            feature_map = feature_map + 10
            unfold_feature_map = self.unfold(feature_map).view(feature_map.size(0), feature_map.size(1), kernel_size**2, -1)
            self.feature_map = feature_map
            self.theta0 = theta0
            self.theta1 = theta1
            self.theta2 = theta2
            self.alpha0 = alpha0
            self.gamma = gamma
            self.base = base
            self.spatial = torch.tensor((np.arange(kernel_size**2)//kernel_size - kernel_size//2) ** 2 +\
                                        (np.arange(kernel_size**2) % kernel_size - kernel_size//2) ** 2).to(feature_map.device).float()

            self.kernel = alpha0 * torch.exp((-(unfold_feature_map - feature_map.view(feature_map.size(0), feature_map.size(1), 1, -1)) ** 2).sum(1) / (2 * self.theta0 ** 2) + (-(self.spatial.view(1, -1, 1) / (2 * self.theta1 ** 2))))
            self.kernel = self.kernel.unsqueeze(1)

            self.iter = iter

    # input x
    # shape of [N H W]
    #@autocast(enabled=False)
    def forward(self, x, targets, inter_img_mask=None):
        with torch.no_grad():
            x = x * targets
            x = (x > 0.5).float() * (1 - self.base*2) + self.base
            U = torch.cat([1-x, x], 1)
            U = U.view(-1, 1, U.size(2), U.size(3))
            if inter_img_mask is not None:
                inter_img_mask.reshape(-1, 1, inter_img_mask.shape[2], inter_img_mask.shape[3])
            ret = U
            for _ in range(self.iter):
                nret = self.simple_forward(ret, targets, inter_img_mask)
                ret = nret
            ret = ret.view(-1, 2, ret.size(2), ret.size(3))
            ret = ret[:,1:]
            ret = (ret > 0.5).float()
            count = ret.reshape(ret.shape[0], -1).sum(1)
            valid = (count >= ret.shape[2] * ret.shape[3] * 0.05) * (count <= ret.shape[2] * ret.shape[3] * 0.95)
            valid = valid.float()
        return ret, valid

    #@autocast(enabled=False)
    def simple_forward(self, x, targets, inter_img_mask):
        h, w = x.size(2), x.size(3)
        unfold_x = self.unfold(-torch.log(x)).view(x.size(0)//2, 2, self.kernel_size**2, -1)
        aggre = (unfold_x * self.kernel).sum(2)
        aggre = aggre.view(-1, 1, h, w)
        f = torch.exp(-aggre)
        f = f.view(-1, 2, h, w)
        if inter_img_mask is not None:
            f += inter_img_mask * self.gamma
        f[:, 1:] *= targets
        f = f + 1e-6
        f = f / f.sum(1, keepdim=True)
        f = (f > 0.5).float() * (1 - self.base*2) + self.base
        f = f.view(-1, 1, h, w)

        return f



@HEADS.register_module()
class DiscoBoxv2Head(nn.Module):

    #@autocast(enabled=False)
    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,
                 loss_ts=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None):
        super(DiscoBoxv2Head, self).__init__()
        self.fp16_enabled = False
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.scale_mids = torch.tensor(np.array(scale_ranges))
        self.scale_mids = (self.scale_mids[:, 0] * self.scale_mids[:, 1]) ** 0.5
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.ins_loss_type = loss_ins['type']
        self.ts_loss_weight = loss_ts['loss_weight']
        self.alpha0 = loss_ts['alpha0']
        self.theta0 = loss_ts['theta0']
        self.theta1 = loss_ts['theta1']
        self.theta2 = loss_ts['theta2']
        self.mkernel = loss_ts['kernel']
        self.crf_base = loss_ts['base']
        self.crf_max_iter = loss_ts['max_iter']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self._init_layers()

        # for debug
        self.cnt = 0

    #@autocast(enabled=False)
    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

    #@autocast(enabled=False)
    def init_weights(self):
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)

    def forward(self, feats, eval=False):
        feats = [feat.float() for feat in feats]
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                       list(range(len(self.seg_num_grids))),
                                                       eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    @autocast()
    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
        
        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred

    def superres_T(self, T):
        # bilinear interpolation
        T_bchw = T.reshape(-1, self.corr_feat_height * self.corr_feat_width, self.corr_feat_height, self.corr_feat_width)
        T_bchw = F.interpolate(T_bchw, (self.corr_mask_height, self.corr_mask_width),
                               mode='bilinear', align_corners=False)
        T_b1hwc = T_bchw.reshape(-1, 1, self.corr_feat_height, self.corr_feat_width,
                                 self.corr_mask_height * self.corr_mask_width)
        T_b1hwc = F.interpolate(T_b1hwc,
                                (self.corr_mask_height, self.corr_mask_width, self.corr_mask_height * self.corr_mask_width),
                                mode='trilinear', align_corners=False)
        T_superres = T_b1hwc.reshape(-1, self.corr_mask_height * self.corr_mask_width,
                                     self.corr_mask_height * self.corr_mask_width) * \
                     (1.0 * self.corr_feat_height * self.corr_feat_width / self.corr_mask_height / self.corr_mask_width)

        return T_superres

    def vis_corr(self, img_a_tensor, img_b_tensor, T, a_mask, b_mask, **img_norm_cfg):

        img_a = tensor2imgs(img_a_tensor, **img_norm_cfg)[0]
        img_b = tensor2imgs(img_b_tensor, **img_norm_cfg)[0]


        img_ab = np.zeros((img_a.shape[0], img_a.shape[1] + img_b.shape[1], 3), np.uint8)
        img_ab[:, :img_a.shape[1]] = img_a
        img_ab[:, img_a.shape[1]:] = img_b

        img_size = img_a.shape[0]

        assignment = T.argmax(1)
        size = int(assignment.shape[0] ** 0.5)
        assignment = assignment.reshape(size, size)

        a_mask = F.interpolate(a_mask.unsqueeze(0), (size, size), mode='bilinear', align_corners=False).squeeze()
        b_mask = F.interpolate(b_mask.unsqueeze(0), (size, size), mode='bilinear', align_corners=False).squeeze()

        scale = img_size / size

        for i in range(0,size,2):
            for j in range(0,size,2):
                x = assignment[i, j] % size
                y = assignment[i, j] // size
                if a_mask[i,j] > 0.5 and b_mask[x,y] > 0.5:
                    cv2.line(img_ab, (int(i * scale), int(j * scale)), (int((x + size) * scale), int(y * scale)),
                             color=tuple(self.color_panel[i*size + j].tolist()), thickness=5)

        self.vis_cnt += 1

        cv2.imwrite('corr_vis/{}.jpg'.format(self.vis_cnt), img_ab)

    def corr_loss(self,
                  cate_preds,
                  s_kernel_preds_raw,
                  t_kernel_preds_raw,
                  s_ins_pred,
                  t_ins_pred,
                  gt_bbox_list,
                  gt_label_list,
                  gt_mask_list,
                  mean_fields,
                  img_metas,
                  cfg,
                  img=None,
                  gt_bboxes_ignore=None,
                  use_loss_ts=False,
                  use_ind_teacher=False,
                  s_feat=None,
                  t_feat=None):

        mask_feat_size = s_ins_pred.size()[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.best_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_label_list = [torch.cat([
            cate_label_list[batch_idx][level_idx].reshape(-1)[grid_order_list[batch_idx][level_idx]]
            for batch_idx in range(len(grid_order_list))], 0)
            for level_idx in range(len(grid_order_list[0]))]

        s_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(s_kernel_preds_raw, zip(*grid_order_list))]

        if use_ind_teacher:
            t_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                               for kernel_preds_level_img, grid_orders_level_img in
                               zip(kernel_preds_level, grid_orders_level)]
                              for kernel_preds_level, grid_orders_level in zip(t_kernel_preds_raw, zip(*grid_order_list))]
        else:
            t_kernel_preds = s_kernel_preds


        # generate masks
        s_ins_pred_list = []
        t_ins_pred_list = []
        color_feats = F.interpolate(img, (s_ins_pred.shape[2], s_ins_pred.shape[3]), mode='bilinear',
                                    align_corners=True)


        img_ind_list = []
        # This code segmentation is for weakly supervised instance segmentation
        # if no independent teacher, t_kenerl_preds is assigned to be s_kernel_preds
        for b_s_kernel_pred, b_t_kernel_pred in zip(s_kernel_preds, t_kernel_preds):
            b_s_mask_pred = []
            b_t_mask_pred = []
            b_img_inds = []
            for idx, (s_kernel_pred, t_kernel_pred) in enumerate(zip(b_s_kernel_pred, b_t_kernel_pred)):

                if s_kernel_pred.size()[-1] == 0:
                    continue
                s_cur_ins_pred = s_ins_pred[idx, ...]
                H, W = s_cur_ins_pred.shape[-2:]
                N, I = s_kernel_pred.shape
                s_cur_ins_pred = s_cur_ins_pred.unsqueeze(0)
                s_kernel_pred = s_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                s_cur_ins_pred = F.conv2d(s_cur_ins_pred, s_kernel_pred, stride=1).view(-1, H, W)
                b_s_mask_pred.append(s_cur_ins_pred)

                if use_ind_teacher:
                    t_cur_ins_pred = t_ins_pred[idx, ...]
                    t_cur_ins_pred = t_cur_ins_pred.unsqueeze(0)
                    t_kernel_pred = t_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    t_cur_ins_pred = F.conv2d(t_cur_ins_pred, t_kernel_pred, stride=1).view(-1, H, W)
                    b_t_mask_pred.append(t_cur_ins_pred)

                b_img_inds.append(torch.ones(s_cur_ins_pred.shape[0]) * idx)
            if len(b_s_mask_pred) == 0:
                b_s_mask_pred = None
                if use_ind_teacher:
                    b_t_mask_pred = None
                b_img_inds = None
            else:
                b_s_mask_pred = torch.cat(b_s_mask_pred, 0)
                if use_ind_teacher:
                    b_t_mask_pred = torch.cat(b_t_mask_pred, 0)
                b_img_inds = torch.cat(b_img_inds, 0)
            s_ins_pred_list.append(b_s_mask_pred)
            # if no independent teacher, t_ins_pred_list is assigned to be s_ins_pred_list
            if use_ind_teacher:
                t_ins_pred_list.append(b_t_mask_pred)
            else:
                t_ins_pred_list = s_ins_pred_list
            img_ind_list.append(b_img_inds)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()
        corr_loss = torch.tensor(0).to(s_ins_pred).float()
        num_ins = 0
        loss_ts = []

        for s_input, t_input, img_inds, target, kernel_labels in \
                zip(s_ins_pred_list, t_ins_pred_list, img_ind_list, ins_labels, kernel_label_list):
            if s_input is None:
                continue
            s_input = torch.sigmoid(s_input)
            if use_ind_teacher:
                t_input = torch.sigmoid(t_input)
            else:
                t_input = s_input

            # remove all-zero target
            mask = torch.tensor([t.sum() for t in target]).to(s_input).bool()
            if mask.sum() == 0:
                continue
            # keep non-zero target
            s_input, t_input, img_inds, target = s_input[mask], t_input[mask], img_inds[mask], target[mask]

            pos_inds = [torch.where(t) for t in target]
            min_y, max_y, min_x, max_x = \
                torch.tensor([ids[0].min() for ids in pos_inds]), torch.tensor(
                    [ids[0].max() for ids in pos_inds]) + 1, \
                torch.tensor([ids[1].min() for ids in pos_inds]), torch.tensor(
                    [ids[1].max() for ids in pos_inds]) + 1
            boxes = torch.cat([min_x.unsqueeze(1), min_y.unsqueeze(1), max_x.unsqueeze(1), max_y.unsqueeze(1)],
                              1).to(s_input)

            roi_s_feat = relu_and_l2_norm_feat(self.feat_roi_align(
                s_feat, torch.cat([img_inds.to(s_feat).unsqueeze(1), boxes], 1)))
            with torch.no_grad():
                roi_t_feat = relu_and_l2_norm_feat(self.feat_roi_align(
                    t_feat.detach(), torch.cat([img_inds.to(s_feat).unsqueeze(1), boxes], 1)))
            if self.save_corr_img:
                roi_img = self.img_roi_align(img, torch.cat(
                    [img_inds.to(t_input).unsqueeze(1), boxes*4], 1)).clone().detach()

            with torch.no_grad():
                roi_s_mask = self.mask_roi_align(s_input.unsqueeze(1).detach(), torch.cat(
                    [torch.arange(target.shape[0]).to(t_input).unsqueeze(1), boxes], 1)).squeeze(1).detach()
                roi_t_mask = self.mask_roi_align(t_input.unsqueeze(1).detach(), torch.cat(
                    [torch.arange(target.shape[0]).to(t_input).unsqueeze(1), boxes], 1)).squeeze(1).detach()
                iiu = torch.zeros(t_input.shape[0] * 2, *t_input.shape[1:]).to(t_input)
                iiu_mask = torch.zeros(t_input.shape[0] * 2).to(t_input)
                queue_area_mask = ((max_x - min_x) > self.objbank_min_size) * (
                        (max_y - min_y) > self.objbank_min_size)

            for idx in torch.arange(len(queue_area_mask)):
                if self.qobj is None:
                    self.qobj = ObjectFactory.create_one(mask=roi_s_mask[idx:idx + 1].detach(),
                                                         feature=roi_s_feat[idx:idx + 1].detach(),
                                                         box=boxes[idx:idx + 1].detach(),
                                                         category=kernel_labels[idx],
                                                         img=roi_img[idx:idx + 1] if self.save_corr_img else None)
                else:
                    self.qobj.mask[...] = roi_s_mask[idx:idx + 1].detach()
                    self.qobj.feature[...] = roi_s_feat[idx:idx + 1].detach()
                    self.qobj.box[...] = boxes[idx:idx + 1].detach()
                    self.qobj.category = int(kernel_labels[idx])
                    if self.save_corr_img:
                        self.qobj.img[...] = roi_img[idx:idx + 1]

                kobjs = self.object_queues.get_similar_obj(self.qobj)
                if kobjs is not None and kobjs['mask'].shape[0] >= 5:
                    Cu, T, fg_mask, bg_mask = self.semantic_corr_solver.solve(self.qobj, kobjs, roi_s_feat[idx:idx + 1])

                    if self.save_corr_img:
                        self.vis_corr(self.qobj.img, kobjs['img'][0:1], T[0], self.qobj.mask, kobjs['mask'][0:1], **self.img_norm_cfg)
                    nce_loss = nn.CrossEntropyLoss()
                    assignment = T.argmax(2).reshape(-1)
                    Cu = Cu.float()
                    Cu = F.softmax(Cu, 2).reshape(-1, Cu.shape[2])
                    corr_loss += nce_loss(Cu, assignment)
                    num_ins += 1

                    with torch.no_grad():
                        T = T * Cu.reshape(T.shape)

                    T = T / (T.sum(2, keepdim=True) + 1e-5)

                    T_superres = self.superres_T(T)

                    fg_ci = torch.matmul(T_superres * (fg_mask > 0.5).float(), torch.clamp(kobjs['mask'], min=0.1, max=0.9).reshape(T_superres.shape[0], T_superres.shape[2], 1).to(Cu)).mean(0).reshape(roi_s_mask.shape[1:])
                    bg_ci = torch.matmul(T_superres * (bg_mask > 0.5).float(), torch.clamp(1-kobjs['mask'], min=0.1, max=0.9).reshape(T_superres.shape[0], T_superres.shape[2], 1).to(Cu)).mean(0).reshape(roi_s_mask.shape[1:])

                    fg_ci = F.interpolate(fg_ci.reshape(1, 1, fg_ci.shape[0], fg_ci.shape[1]),
                                          (int(boxes[idx][3] - boxes[idx][1]), int(boxes[idx][2] - boxes[idx][0])),
                                          mode='bilinear', align_corners=False).squeeze()
                    bg_ci = F.interpolate(bg_ci.reshape(1, 1, bg_ci.shape[0], bg_ci.shape[1]),
                                          (int(boxes[idx][3] - boxes[idx][1]), int(boxes[idx][2] - boxes[idx][0])),
                                          mode='bilinear', align_corners=False).squeeze()
                    iiu[idx*2, int(boxes[idx, 1]):int(boxes[idx, 3]),
                                int(boxes[idx, 0]):int(boxes[idx, 2])] = bg_ci
                    iiu[idx*2+1, int(boxes[idx, 1]):int(boxes[idx, 3]),
                                int(boxes[idx, 0]):int(boxes[idx, 2])] = fg_ci

                    if self.save_corr_img:
                        self.cnt += 1
                        vis_seg(self.qobj.img, ci, self.img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=self.cnt)
                        self.cnt += 1
                        vis_seg(self.qobj.img, roi_s_mask[idx], self.img_norm_cfg, save_dir='work_dirs/corr_vis', data_id=self.cnt)
                if queue_area_mask[idx]:
                    if self.num_created_gpu_bank < self.num_gpu_bank:
                        device = mask.device
                    else:
                        device = 'cpu'
                    created_gpu_bank = self.object_queues.append(int(kernel_labels[idx]),
                                                                 idx,
                                                                 roi_t_feat,
                                                                 roi_t_mask,
                                                                 boxes.detach(),
                                                                 roi_img if self.save_corr_img else None,
                                                                 device=device)
                    self.num_created_gpu_bank += created_gpu_bank

            iiu = iiu.reshape(iiu.shape[0] // 2, 2, iiu.shape[1], iiu.shape[2])
            for img_idx in range(len(mean_fields)):
                obj_inds = (img_inds == img_idx)
                enlarged_target = F.max_pool2d(target.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1).byte()
                if obj_inds.sum() > 0:
                    pseudo_label, valid = mean_fields[int(img_idx)](
                        (t_input[obj_inds].unsqueeze(1) + s_input[obj_inds].unsqueeze(1)) / 2,
                        target[obj_inds].unsqueeze(1), iiu[obj_inds])
                    cropped_s_input = s_input[obj_inds] * enlarged_target[obj_inds]
                    cropped_s_input = cropped_s_input * mean_fields[int(img_idx)].gamma + cropped_s_input.detach() * (1 - mean_fields[int(img_idx)].gamma)
                    loss_ts.append(dice_loss(cropped_s_input, pseudo_label))

        return corr_loss / (num_ins + 1e-4), loss_ts


    @autocast()
    def loss(self,
                 cate_preds,
                 s_kernel_preds_raw,
                 t_kernel_preds_raw,
                 s_ins_pred,
                 t_ins_pred,
                 gt_bbox_list,
                 gt_label_list,
                 gt_mask_list,
                 img_metas,
                 cfg,
                 img=None,
                 gt_bboxes_ignore=None,
                 use_loss_ts=False,
                 use_ind_teacher=False,
                 use_corr=False,
                 s_feat=None,
                 t_feat=None):

        mask_feat_size = s_ins_pred.size()[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)

        if s_feat is not None:
            s_feat = s_feat[0]
        if t_feat is not None:
            t_feat = t_feat[0]

        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        s_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(s_kernel_preds_raw, zip(*grid_order_list))]

        if use_ind_teacher:
            t_kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                               for kernel_preds_level_img, grid_orders_level_img in
                               zip(kernel_preds_level, grid_orders_level)]
                              for kernel_preds_level, grid_orders_level in zip(t_kernel_preds_raw, zip(*grid_order_list))]
        else:
            t_kernel_preds = s_kernel_preds

        kernel_label_list = [torch.cat([
            cate_label_list[batch_idx][level_idx].reshape(-1)[grid_order_list[batch_idx][level_idx]]
            for batch_idx in range(len(grid_order_list))], 0)
            for level_idx in range(len(grid_order_list[0]))]

        # generate masks
        s_ins_pred_list = []
        t_ins_pred_list = []
        color_feats = F.interpolate(img, (s_ins_pred.shape[2], s_ins_pred.shape[3]), mode='bilinear', align_corners=True)

        img_ind_list = []
        # This code segmentation is for weakly supervised instance segmentation
        # if no independent teacher, t_kenerl_preds is assigned to be s_kernel_preds
        for b_s_kernel_pred, b_t_kernel_pred in zip(s_kernel_preds, t_kernel_preds):
            b_s_mask_pred = []
            b_t_mask_pred = []
            b_img_inds = []
            for idx, (s_kernel_pred, t_kernel_pred) in enumerate(zip(b_s_kernel_pred, b_t_kernel_pred)):

                if s_kernel_pred.size()[-1] == 0:
                    continue
                s_cur_ins_pred = s_ins_pred[idx, ...]
                H, W = s_cur_ins_pred.shape[-2:]
                N, I = s_kernel_pred.shape
                s_cur_ins_pred = s_cur_ins_pred.unsqueeze(0)
                s_kernel_pred = s_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                s_cur_ins_pred = F.conv2d(s_cur_ins_pred, s_kernel_pred, stride=1).view(-1, H, W)
                b_s_mask_pred.append(s_cur_ins_pred)

                if use_ind_teacher:
                    t_cur_ins_pred = t_ins_pred[idx, ...]
                    t_cur_ins_pred = t_cur_ins_pred.unsqueeze(0)
                    t_kernel_pred = t_kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    t_cur_ins_pred = F.conv2d(t_cur_ins_pred, t_kernel_pred, stride=1).view(-1, H, W)
                    b_t_mask_pred.append(t_cur_ins_pred)

                b_img_inds.append(torch.ones(s_cur_ins_pred.shape[0]) * idx)
            if len(b_s_mask_pred) == 0:
                b_s_mask_pred = None
                if use_ind_teacher:
                    b_t_mask_pred = None
                b_img_inds = None
            else:
                b_s_mask_pred = torch.cat(b_s_mask_pred, 0)
                if use_ind_teacher:
                    b_t_mask_pred = torch.cat(b_t_mask_pred, 0)
                b_img_inds = torch.cat(b_img_inds, 0)
            s_ins_pred_list.append(b_s_mask_pred)
            # if no independent teacher, t_ins_pred_list is assigned to be s_ins_pred_list
            if use_ind_teacher:
                t_ins_pred_list.append(b_t_mask_pred)
            else:
                t_ins_pred_list = s_ins_pred_list
            img_ind_list.append(b_img_inds)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()


        # This code segmentation is for weakly supervised semantic correspondence

        # dice loss
        loss_ins = []
        loss_ts = []

        # Mean Field Init

        mean_fields = [MeanField(color_feat.unsqueeze(0), alpha0=self.alpha0,
                                 theta0=self.theta0, theta1=self.theta1, theta2=self.theta2,
                                 iter=self.crf_max_iter, kernel_size=self.mkernel, base=self.crf_base) \
                       for color_feat in color_feats]

        for s_input, t_input, img_inds, target, kernel_labels in \
                zip(s_ins_pred_list, t_ins_pred_list, img_ind_list, ins_labels, kernel_label_list):
            if s_input is None:
                continue
            s_input = torch.sigmoid(s_input)
            if use_ind_teacher:
                t_input = torch.sigmoid(t_input)
            else:
                t_input = s_input

            # remove all-zero target
            mask = torch.tensor([t.sum() for t in target]).to(s_input).bool()
            if mask.sum() == 0:
                continue
            # keep non-zero target
            s_input, t_input, img_inds, target = s_input[mask], t_input[mask], img_inds[mask], target[mask]

            # unary loss
            loss_ins.append(dice_loss(s_input, target))

            # pairwise loss
            # crf

            if use_loss_ts:
                enlarged_target = F.max_pool2d(target.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1).byte()
                for img_idx in range(len(mean_fields)):
                    obj_inds = (img_inds == img_idx)
                    if obj_inds.sum() > 0:
                        pseudo_label, valid = mean_fields[int(img_idx)](
                            (t_input[obj_inds].unsqueeze(1) + s_input[obj_inds].unsqueeze(1))/2, target[obj_inds].unsqueeze(1))
                        loss_ts.append(dice_loss(s_input[obj_inds] * enlarged_target[obj_inds], pseudo_label))

        if len(loss_ins) > 0:
            loss_ins = torch.cat(loss_ins).mean()
            loss_ins = loss_ins * self.ins_loss_weight
        else:
            loss_ins = torch.zeros(1).to(color_feats)

        if use_loss_ts and len(loss_ts) > 0:
            loss_ts = torch.cat(loss_ts).mean()
        else:
            loss_ts = torch.zeros(1).to(color_feats)

        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        flatten_cate_preds = flatten_cate_preds.float()

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_ts=loss_ts,
            loss_cate=loss_cate,
            )
        

    #@autocast(enabled=False)
    def best_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           mask_feat_size):


        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1])).unsqueeze(1)

        scale_mids = self.scale_mids.to(device).unsqueeze(0)
        scale_diffs = scale_mids / (gt_areas + 1e-6)
        scale_diffs[scale_diffs < 1] = 1 / (scale_diffs[scale_diffs < 1] + 1e-6)
        scale_ids = scale_diffs.argmin(1)

        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []
        grid_order_list = []
        for level_ids, (stride, num_grid) in enumerate(zip(self.strides, self.seg_num_grids)):

            hit_indices = (level_ids == scale_ids).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks.to_ndarray()).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                cate_label[coord_h, coord_w] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                label = int(coord_h * num_grid + coord_w)
                cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                            device=device)
                cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                ins_label.append(cur_ins_label)
                ins_ind_label[label] = True
                grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list



    #@autocast(enabled=False)
    def solov2_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks.to_ndarray()).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    #@autocast(enabled=False)
    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale=None, img=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor,
                                         cfg, rescale, img=img)
            result_list.append(result)
        return result_list

    #@autocast(enabled=False)
    @autocast()
    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False,
                       img=None,
                       debug=False):

        assert len(cate_preds) == len(kernel_preds)
        bbox_results = [[] for _ in range(self.num_classes)]
        mask_results = [[] for _ in range(self.num_classes)]
        score_results = [[] for _ in range(self.num_classes)]

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            bbox_results = [np.zeros((0, 5)) for bbox_result in bbox_results]
            return bbox_results, (mask_results, score_results)

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.

        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            bbox_results = [np.zeros((0, 5)) for bbox_result in bbox_results]
            return bbox_results, (mask_results, score_results)

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg.kernel,sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            bbox_results = [np.zeros((0, 5)) for bbox_result in bbox_results]
            return bbox_results, (mask_results, score_results)
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]

        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr ##

        for cate_label, cate_score, seg_mask in zip(cate_labels, cate_scores, seg_masks):
            if seg_mask.sum() > 0:
                mask_results[cate_label].append(seg_mask.cpu())
                score_results[cate_label].append(cate_score.cpu())
                ys, xs = torch.where(seg_mask)
                min_x, min_y, max_x, max_y = xs.min().cpu().data.numpy(), ys.min().cpu().data.numpy(), xs.max().cpu().data.numpy(), ys.max().cpu().data.numpy()
                bbox_results[cate_label].append([min_x, min_y, max_x+1, max_y+1, cate_score.cpu().data.numpy()])

        bbox_results = [np.array(bbox_result) if len(bbox_result) > 0 else np.zeros((0, 5)) for bbox_result in bbox_results ]


        return bbox_results, (mask_results, score_results)
