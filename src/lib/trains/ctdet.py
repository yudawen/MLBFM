from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fvcore.nn
import torch
import numpy as np
from models.dance_lib.utils.dance import dance_config, dance_gcn_utils, dance_decode
from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from torch import nn
from lib.models.dance_lib.utils import net_utils
from lib.trains.DMLoss import DMLoss
import torch.nn.functional as F


import torch
def build_ce_with_fixed_weights(device="cuda", reduction="mean"):
  """
  使用固定权重的交叉熵：
  - 类别 0：背景，权重 = 1
  - 类别 1-10：直接使用给定数量作为权重
  """

  weights = [

    1,  # class 1
    85,  # class 2
    56,  # class 3
    3,  # class 4
    3,  # class 5
    88,  # class 6
    315,  # class 7
    362,  # class 8
    2,  # class 9
    37,  # class 10

  ]

  weights = torch.tensor(weights, dtype=torch.float32, device=device)
  weights.requires_grad = False
  return nn.CrossEntropyLoss(weight=weights, reduction=reduction)

from scipy.optimize import linear_sum_assignment
def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pred=torch.sigmoid(pred)
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()
  # neg_inds_ = gt.eq(0)#.int()
  # print(torch.sum(neg_inds_))
  # print(neg_inds_.size())
  neg_weights = torch.pow(1 - gt, 4)
  # print(torch.sum(neg_weights))
  # neg_weights[neg_inds_]=0
  # print(torch.sum(neg_weights))

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


class FocalLossv2(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', auto_alpha=True, alpha=None):
      """
      :param gamma: 聚焦参数，默认 2.0
      :param reduction: 'mean' | 'sum' | 'none'
      :param auto_alpha: 是否自动根据正负样本比例设置 α
      :param alpha: 手动指定 α (给正样本的权重)，auto_alpha=False 时才用
      """
      super(FocalLossv2, self).__init__()
      self.gamma = gamma
      self.reduction = reduction
      self.auto_alpha = auto_alpha
      self.alpha = alpha

    def forward(self, inputs, targets):
      """
      :param inputs: 模型输出的概率（未经过 sigmoid 之后），形状 (N,) 或 (N,1)
      :param targets: 真实标签 (0/1)，形状 (N,) 或 (N,1)
      """
      inputs=torch.sigmoid(inputs)
      if inputs.ndim > 1:
        inputs = inputs.view(-1)
      if targets.ndim > 1:
        targets = targets.view(-1)

      # 防止 log(0)
      eps = 1e-8
      inputs = torch.clamp(inputs, eps, 1. - eps)

      # 动态计算 α
      if self.auto_alpha:
        pos_count = targets.sum().item()
        neg_count = targets.numel() - pos_count
        total = pos_count + neg_count + 1e-6
        alpha = neg_count / total  # 给正样本的权重
      else:
        alpha = self.alpha if self.alpha is not None else 0.25

      # 计算 focal loss
      ce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
      pt = inputs * targets + (1 - inputs) * (1 - targets)  # p_t
      focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - pt) ** self.gamma
      loss = focal_weight * ce_loss

      if self.reduction == 'mean':
        return loss.mean()
      elif self.reduction == 'sum':
        return loss.sum()
      else:
        return loss

class BCEFocalLoss(torch.nn.Module):

  def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction


  def forward(self, _input, target):

    pt = _input#torch.sigmoid(_input)
    # pt = _input
    alpha = self.alpha
    loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
           (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
    if self.reduction == 'mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)
    return loss




class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    # print(self.crit)
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.py_crit = DMLoss(type='smooth_l1')
    self.py_crit2 = torch.nn.functional.smooth_l1_loss
    self.opt = opt
    self.cla_loss=FocalLossv2()

    # 获取加权交叉熵损失函数
    self.criterion = build_ce_with_fixed_weights()



  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0


    for s in range(1):
      output = outputs
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

      hm_loss=hm_loss*1.0#+corner_loss#+ms_loss*10.0

      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
          # print(opt.cat_spec_wh,'here')
        else:
          wh_loss += self.crit_reg(
            output['wh'][:,:2,:,:], batch['reg_mask'],
            batch['ind'], batch['wh'][:,:,:2]) / opt.num_stacks
          wh_loss=0.1*wh_loss


      py_loss_all = torch.tensor(0.0).cuda()

      off_loss = torch.tensor(0.0).cuda()

      off_loss += self.crit_reg(output['reg'], batch['reg_mask'],batch['ind'], batch['reg']) / opt.num_stacks
      off_loss=off_loss*1


    key_masks = batch['keymasks'].view(-1, 64, 1)

    key_masks_weight=1

    # try:
    if 1:
      for i in range(3):
        py_loss_all += self.py_crit2(output['py_pred'][0][i] * key_masks_weight,output['i_gt_py1']* key_masks_weight)
        py_loss_all += _neg_loss(output['py_pred'][1][i], key_masks.cuda())
      for i in range(6):
        pt = output['py_pred'][2][i]

        if i%2==0:
          py_loss_all += self.criterion(pt.view(-1,10), batch['ct_cls'].view(-1).long().cuda())

        else:
          continue

          py_loss_all += self.criterion(pt.view(-1, 10), batch['ct_cls'].view(-1).long().cuda()) *0.1



    loss = hm_loss + wh_loss + off_loss + py_loss_all
    loss_stats = {'loss': loss, 'hm_ls': hm_loss, 'wh_ls': wh_loss,'py_ls':py_loss_all}#'off_loss':off_loss,,'ed_ls':edge_loss
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_ls', 'wh_ls','py_ls','loss_llm']#'off_loss','ed_ls','s2l','s2a'
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(output['hm'], output['wh'], reg=reg,cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]