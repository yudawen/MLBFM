from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from skimage import measure
import os
import numpy as np
import numpy
from progress.bar import Bar
import time
import torch
import random
from models.utils import _sigmoid
import glob

# try:
#   from external.nms import soft_nms,nms
# except:
#   print('NMS not imported! If you need it,'
#         ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger
from lib.models.dance_lib.utils.snake import snake_decode
from lib.models.dance_lib.utils import data_utils
from .base_detector import BaseDetector



from lib.models.dance_lib.utils.snake import snake_decode
import torchvision
nms = torchvision.ops.nms

import math

import torch.nn.functional as F





def com_area(contour):
    n = len(contour)
    s = 0
    for i in range(n-1):
        s = s + contour[i][0]*contour[i+1][1]-contour[i+1][0]*contour[i][1]
    s = s + contour[n-1][0]*contour[0][1]-contour[0][0]*contour[n-1][1]
    s = math.fabs(s)/2
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

class DouglasClosedFast:
    def __init__(self, epsilon=1.0):
      self.D = epsilon

    def _dp(self, i1, i2, poly, mask):
      if i2 <= i1 + 1:
        return

      a = poly[i1]
      b = poly[i2]
      pts = poly[i1 + 1:i2]

      ab = b - a
      ap = pts - a
      denom = np.dot(ab, ab)

      if denom < 1e-8:
        dists = np.linalg.norm(ap, axis=1)
      else:
        t = np.dot(ap, ab) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dists = np.linalg.norm(pts - proj, axis=1)

      max_idx_local = np.argmax(dists)
      dmax = dists[max_idx_local]

      if dmax > self.D:
        max_idx = i1 + 1 + max_idx_local
        mask[max_idx] = 1
        self._dp(i1, max_idx, poly, mask)
        self._dp(max_idx, i2, poly, mask)

    def _make_path(self, start, end, N):
      if start <= end:
        return np.arange(start, end + 1)
      else:
        return np.concatenate([np.arange(start, N), np.arange(0, end + 1)])

    def _farthest_pair_bbox(self, poly):
      min_xy = poly.min(axis=0)
      max_xy = poly.max(axis=0)
      i0 = np.argmax(np.sum((poly - min_xy) ** 2, axis=1))
      i1 = np.argmax(np.sum((poly - max_xy) ** 2, axis=1))
      return i0, i1

    def sample(self, poly):
      N = len(poly)
      mask = np.zeros(N, dtype=int)

      i0, i1 = self._farthest_pair_bbox(poly)

      for start, end in ((i0, i1), (i1, i0)):
        path = self._make_path(start, end, N)
        sub_poly = poly[path]
        sub_mask = np.zeros(len(sub_poly), dtype=int)
        sub_mask[0] = 1
        sub_mask[-1] = 1

        self._dp(0, len(sub_poly) - 1, sub_poly, sub_mask)
        mask[path[sub_mask == 1]] = 1

      return mask

    def sample_batch(self, polys):
      """
      polys: np.ndarray, shape (P, N, 2)
      return: np.ndarray, shape (P, N), int mask
      """
      P, N, _ = polys.shape
      masks = np.zeros((P, N), dtype=int)

      for i in range(P):
        masks[i] = self.sample(polys[i])

      return masks
import torch


def decode_detection(output, h, w):

    ct_hm = output['hm']
    wh = output['wh']
    reg=output['reg']
    cts, detections = snake_decode.decode_ct_hm(_sigmoid(ct_hm), wh,reg=reg, K=300,cat_spec_wh=False)
    detections[..., :4] = data_utils.clip_to_image(detections[..., :4], h, w)
    keep = nms(detections[0,:, 0:4].cpu(), detections[0,:,4].cpu(), 0.3)
    index = keep.view(-1).long().cpu()
    #
    detections=detections[:,index,:].cuda()
    cts=cts[:,index,:]

    output.update({'ct': cts, 'detection': detections})

    return detections

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)

    self.id=0

  def _sigmoid(self,x):
      y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
      return y
  def decode_detection(self, output, h, w):

    ct_hm = output['hm']
    wh = output['wh']
    reg=output['reg']
    cts, detections = snake_decode.decode_ct_hm(self._sigmoid(ct_hm), wh,reg=reg, K=300,cat_spec_wh=False)
    detections[..., :4] = data_utils.clip_to_image(detections[..., :4], h, w)
    keep = nms(detections[0,:, 0:4].cpu(), detections[0,:,4].cpu(), 0.3)
    index = keep.view(-1).long().cpu()
    #
    detections=detections[:,index,:].cuda()
    cts=cts[:,index,:]
    # print(detections.size())

    output.update({'ct': cts, 'detection': detections})

    return detections

  def process(self, images, return_time=False):
        images=images[:,:3,:,:]


        building_functions = [ 'dense residential', 'business','commercial','residential',
                     'factory','government','hospital','resort','public','school']



        with torch.no_grad():

             all_polygons = []

             prompts = []
             answers = []
             for i in range(1):

                 prompt = "<image>"
                 prompt += "Can you segment the buildings with different functions in the image?"
                 answer = 'Sure, they are '
                 for id, name in enumerate(building_functions):
                     if id < len(building_functions) - 1:
                         answer += f"{name} [SEG{id}],"
                     else:
                         answer += f"{name} [SEG{id}]."
                 answers.append(answer)
                 prompts.append(prompt)

             outputs, cnn_feature, xs, maskbackbone, all_feats, seg_embeddings, _ = self.model(images,
                                                                                               prompt=prompts,
                                                                                               category=answers)



             outputs = outputs[0]
             outputs.update({'hm': outputs['hm'], 'wh': outputs['wh'], 'reg': outputs['reg']})  # [0:1]
             cnn_feature, maskbackbone = cnn_feature, maskbackbone

             xs.update({"0": xs["0"], "1": xs["1"], "2": xs["2"]})  #

             all_feats.update({'layer0': all_feats[f"layer0"], 'layer1': all_feats[f"layer1"],
                               'layer2': all_feats[f"layer2"], 'layer3': all_feats[f"layer3"],
                               'x_original': all_feats[f"x_original"]})

             dets = self.decode_detection(outputs, cnn_feature.size(2), cnn_feature.size(3))
             outputs = self.model.gcn(outputs, cnn_feature, None, False, xs, maskbackbone, all_feats,
                                      seg_embeddings=seg_embeddings)

             poly_num = outputs['py'][0][-1].size(0)
             polygons = outputs['py'][0][-1].cpu().detach().clone().numpy()

             scores = (F.sigmoid(outputs['py'][1][-1])).cpu().detach().clone().numpy()
             claes = (F.softmax(outputs['py'][2][-2], dim=-1)).cpu().detach().clone().numpy()





             outputs['detection'] = outputs['detection'].cpu()
             claes = numpy.argmax(claes, axis=2)#[0]


             outputs['claes'] = claes[0]  # outputs['detection'][0].cpu()

             for pi in range(poly_num):

                 pred_poly_ = polygons[pi]
                 score = scores[pi][:, 0]
                 pred_poly = []
                 for pid, p in enumerate(pred_poly_):
                     if score[pid] >= 0.05:
                         pred_poly.append(p)
                 pred_poly = np.array(pred_poly)
                 if len(pred_poly) == 0:
                     pred_poly = pred_poly_
                 all_polygons.append(pred_poly)
                 continue

             assert len(all_polygons) == poly_num
             forward_time = time.time()
             outputs.update({'all_polygons': all_polygons})



        if return_time:
         return outputs, dets, forward_time
        else:
         return outputs, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], self.opt.num_classes)

    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
      # if len(self.scales) > 1 or self.opt.nms:
      #    soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
    # print(self.max_per_image)
    self.max_per_image=300
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    # debugger.show_all_imgs(pause=self.pause)
