from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import time
import torch
import torchvision
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from models.utils import _sigmoid
from lib.models.dance_lib.utils.snake import snake_decode
from lib.models.dance_lib.utils import data_utils
import numpy as np
import math
nms = torchvision.ops.nms
import random

def com_area(contour):
  n = len(contour)
  s = 0
  for i in range(n - 1):
    s = s + contour[i][0] * contour[i + 1][1] - contour[i + 1][0] * contour[i][1]
  s = s + contour[n - 1][0] * contour[0][1] - contour[0][0] * contour[n - 1][1]
  s = math.fabs(s) / 2
  return s

def positional_encoding_2d(d_model, height, width):
  """
  :param d_model: dimension of the model
  :param height: height of the positions
  :param width: width of the positions
  :return: d_model*height*width position matrix
  """
  if d_model % 4 != 0:
    raise ValueError("Cannot use sin/cos positional encoding with "
                     "odd dimension (got dim={:d})".format(d_model))
  pe = torch.zeros(d_model, height, width)
  # Each dimension use half of d_model
  d_model = int(d_model / 2)
  div_term = torch.exp(torch.arange(0., d_model, 2) *
                       -(math.log(10000.0) / d_model))
  pos_w = torch.arange(0., width).unsqueeze(1)
  pos_h = torch.arange(0., height).unsqueeze(1)
  pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
  pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
  pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
  pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

  return pe



def get_pixel_features(image_size, d_pe=128):
  all_pe = positional_encoding_2d(d_pe, image_size, image_size)
  pixels_x = np.arange(0, image_size)
  pixels_y = np.arange(0, image_size)

  xv, yv = np.meshgrid(pixels_x, pixels_y)
  all_pixels = list()
  for i in range(xv.shape[0]):
    pixs = np.stack([xv[i], yv[i]], axis=-1)
    all_pixels.append(pixs)
  pixels = np.stack(all_pixels, axis=0)

  pixel_features = all_pe[:, pixels[:, :, 1], pixels[:, :, 0]]
  pixel_features = pixel_features.permute(1, 2, 0)
  return pixels, pixel_features


import torch
from fvcore.nn import FlopCountAnalysis


def decode_detection(output, h, w):

    ct_hm = output['hm']
    wh = output['wh']
    reg=output['reg']
    cts, detections = snake_decode.decode_ct_hm(_sigmoid(ct_hm), wh,reg=reg, K=64,cat_spec_wh=False)
    detections[..., :4] = data_utils.clip_to_image(detections[..., :4], h, w)
    keep = nms(detections[0,:, 0:4].cpu(), detections[0,:,4].cpu(), 0.3)
    index = keep.view(-1).long().cpu()
    #
    detections=detections[:,index,:].cuda()
    cts=cts[:,index,:]

    output.update({'ct': cts, 'detection': detections})

    return detections
class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss


  def decode_detection(self, output, h, w):

    ct_hm = output['hm']
    wh = output['wh']

    reg=output['reg']
    ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh,reg=reg, K=64,cat_spec_wh=False)
    detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)

    output.update({'ct': ct, 'detection': detection})
    return ct, detection

  def forward(self, batch):

    building_functions = ['dense residential', 'business', 'commercial', 'residential',
                          'factory', 'government', 'hospital', 'resort', 'public', 'school']

    B, N = batch['ct_cls'].shape
    target = batch['ct_cls'].clone()

    prompts=[]
    answers=[]
    for i in range(B):
        prompt = "<image>"
        prompt += "Can you segment the buildings with different functions in the image?"
        exist = [f"{name} [SEG{id}]" for id, name in enumerate(building_functions) if id in target[i]]
        not_exist = [f"{name} [SEG{id}]" for id, name in enumerate(building_functions) if id not in target[i]]

        answer = f"Sure, there are {', '.join(exist)}, there are no {', '.join(not_exist)}."
        # print(prompt)
        # print(answer)

        prompts.append(prompt)
        answers.append(answer)


    outputs,cnn_feature,xs, maskbackbone, all_feats,seg_embeddings,loss_llm= self.model(batch['input'],prompt=prompts,category=answers)



    outputs = outputs[0]


    with torch.no_grad():
      self.decode_detection(outputs, cnn_feature.size(2), cnn_feature.size(3))

    outputs = self.model.gcn(outputs, cnn_feature, batch, True,xs, maskbackbone, all_feats,seg_embeddings=seg_embeddings)



    loss, loss_stats = self.loss(outputs, batch)

    loss_stats.update({'loss_llm': loss_llm})
    loss_stats.update({'loss': loss+loss_llm})
    return outputs, loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      for k in batch:
        if k != 'meta' and k!='annots_corner'and k!='pt_corner':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      if phase == 'train':
        output, loss, loss_stats = model_with_loss(batch)
      else:
        with torch.no_grad():
            output, loss, loss_stats = model_with_loss(batch)

      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        # print(loss_stats[l])
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)