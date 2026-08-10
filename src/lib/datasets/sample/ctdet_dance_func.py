from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .douglas import DouglasClosedFast
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from itertools import product
import torch.utils.data as data
import torch
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import math
from shapely.geometry import Polygon
import random
import cv2
import numpy as np



def com_area(contour):
    n = len(contour)
    s = 0
    for i in range(n-1):
        s = s + contour[i][0]*contour[i+1][1]-contour[i+1][0]*contour[i][1]
    s = s + contour[n-1][0]*contour[0][1]-contour[0][0]*contour[n-1][1]
    s = math.fabs(s)/2
    return s

def generate_random_color():
    r = random.randint(0, 255)  # Red component
    g = random.randint(0, 255)  # Green component
    b = random.randint(0, 255)  # Blue component
    return (r, g, b)

def reorder_perloss(oct_sampled_targets, oct_sampled_pts):
  """
  Adaptively adjust the penalty, concept-wise the loss is much more reasonable.
  :param oct_sampled_targets: (\sum{k}, num_sampling, 2) for all instances
  :param oct_sampled_pts: same~
  :return:
  """
  assert oct_sampled_targets.size() == oct_sampled_pts.size()
  n = len(oct_sampled_targets)
  num_locs = oct_sampled_pts.size(1)
  ind1 = torch.arange(num_locs, device=oct_sampled_targets.device)
  ind2 = ind1.expand(num_locs, -1)
  enumerated_ind = torch.fmod(ind2 + ind1.view(-1, 1),
                              num_locs).view(-1).long()
  enumerated_targets = oct_sampled_targets[:, enumerated_ind, :].view(
    n, -1, num_locs, 2)
  diffs = enumerated_targets - oct_sampled_pts[:, None, ...]
  diffs_sum = diffs.pow(2).sum(3).sum(2)
  tt_idx = torch.argmin(diffs_sum, dim=1)
  re_ordered_gt = enumerated_targets[torch.arange(n), tt_idx]
  return re_ordered_gt

def uniform_sample_1d(pts, new_n):
  if new_n == 1:
    return pts[:1]
  n = pts.shape[0]
  if n == new_n + 1:
    return pts[:-1]
  # len: n - 1
  segment_len = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))

  # down-sample or up-sample
  # n
  start_node = np.cumsum(np.concatenate([np.array([0]), segment_len]))
  total_len = np.sum(segment_len)

  new_per_len = total_len / new_n

  mark_1d = ((np.arange(new_n - 1) + 1) * new_per_len).reshape(-1, 1)
  locate = (start_node.reshape(1, -1) - mark_1d)
  iss, jss = np.where(locate > 0)
  cut_idx = np.cumsum(np.unique(iss, return_counts=True)[1])
  cut_idx = np.concatenate([np.array([0]), cut_idx[:-1]])

  after_idx = jss[cut_idx]
  before_idx = after_idx - 1

  after_idx[after_idx < 0] = 0

  before = locate[np.arange(new_n - 1), before_idx]
  after = locate[np.arange(new_n - 1), after_idx]

  w = (-before / (after - before)).reshape(-1, 1)

  sampled_pts = (1 - w) * pts[before_idx] + w * pts[after_idx]

  return np.concatenate([pts[:1], sampled_pts], axis=0)

def single_uniform_multisegment_matching(dense_targets, sampled_pts, ext_idx,up_rate, poly_num):

  min_idx = ext_idx

  ch_pts = dense_targets[min_idx]  # characteristic points

  diffs = ((ch_pts[:, np.newaxis, :] -
            sampled_pts[np.newaxis]) ** 2).sum(axis=2)
  ext_idx = np.argmin(diffs, axis=1)
  if ext_idx[0] != 0:
    ext_idx[0] = 0
  if ext_idx[-1] < ext_idx[1]:
    ext_idx[-1] = poly_num - 2
  ext_idx = np.sort(ext_idx)

  aug_ext_idx = np.concatenate([ext_idx, np.array([poly_num])], axis=0)

  # diff = np.sum((ch_pts[:, np.newaxis, :] - dense_targets[np.newaxis, :, :]) ** 2, axis=2)
  # min_idx = np.argmin(diff, axis=1)

  aug_min_idx = np.concatenate(
    [min_idx, np.array([poly_num * up_rate])], axis=0)

  if aug_min_idx[-1] < aug_min_idx[1]:
    # print("WARNING: Last point not matching!")
    # print(aug_ext_idx)
    # print(aug_min_idx)
    aug_min_idx[
      -1] = poly_num * up_rate - 2  # enforce matching of the last point

  if aug_min_idx[0] != 0:
    # TODO: This is crucial, or other wise the first point may be
    # TODO: matched to near 640, then sorting will completely mess
    # print("WARNING: First point not matching!")
    # print(aug_ext_idx)
    # print(aug_min_idx)
    aug_min_idx[0] = 0  # enforce matching of the 1st point

  aug_ext_idx = np.sort(aug_ext_idx)
  aug_min_idx = np.sort(aug_min_idx)

  # === error-prone ===

  # deal with corner cases

  if aug_min_idx[-2] == poly_num * up_rate - 1:
    # print("WARNING: Bottom extreme point being the last point!")
    # print("WARNING: Bottom extreme point being the last point!")
    # hand designed remedy
    aug_min_idx[-2] = poly_num * up_rate - 3
    aug_min_idx[-1] = poly_num * up_rate - 2

  if aug_min_idx[-1] == poly_num * up_rate - 1:
    # print("WARNING: Right extreme point being the last point!")
    # print("WARNING: Right extreme point being the last point!")
    # print(aug_ext_idx)
    # print(aug_min_idx)
    aug_min_idx[-1] -= 1
    aug_min_idx[-2] -= 1

  segments = []
  try:
    for i in range(len(ext_idx)):
      if aug_ext_idx[i + 1] - aug_ext_idx[i] == 0:
        continue  # no need to sample for this segment

      if aug_min_idx[i + 1] - aug_min_idx[i] <= 0:
        # overlap due to quantization, negative value is due to accumulation of overlap
        aug_min_idx[i + 1] = aug_min_idx[i] + 1  # guarantee spacing

      if i == len(ext_idx) - 1:  # last, complete a circle
        pts = np.concatenate(
          [dense_targets[aug_min_idx[i]:], dense_targets[:1]],
          axis=0)
      else:
        pts = dense_targets[aug_min_idx[i]:aug_min_idx[i + 1] +
                                           1]  # including
      new_sampled_pts = uniform_sample_1d(
        pts, aug_ext_idx[i + 1] - aug_ext_idx[i])
      segments.append(new_sampled_pts)
    # segments.append(dense_targets[-1:]) # close the loop
    segments = np.concatenate(segments, axis=0)
    if len(segments) != poly_num:
      # print("WARNING: Number of points not matching!")
      print("WARNING: Number of points not matching!",
            len(segments))
      raise ValueError(len(segments))
  except Exception as err:  # may exist some very tricky corner cases...
    # print("WARNING: Tricky corner cases occurred!")
    print("WARNING: Tricky corner cases occurred!")
    print(err)
    print(aug_ext_idx)
    print(aug_min_idx)
    # raise ValueError('TAT')
    segments = reorder_perloss(
      torch.from_numpy(dense_targets[::up_rate][None]),
      torch.from_numpy(sampled_pts)[None])[0]
    segments = segments.numpy()

  return segments

def get_aux_extreme_points(pts):
  num_pt = pts.shape[0]

  aux_ext_pts = []

  l, t = min(pts[:, 0]), min(pts[:, 1])
  r, b = max(pts[:, 0]), max(pts[:, 1])
  # 3 degrees
  thresh = 0.02
  band_thresh = 0.02
  w = r - l + 1
  h = b - t + 1

  t_band = np.where((pts[:, 1] - t) <= band_thresh * h)[0].tolist()
  while t_band:
    t_idx = t_band[np.argmin(pts[t_band, 1])]
    t_idxs = [t_idx]
    tmp = (t_idx + 1) % num_pt
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
      t_idxs.append(tmp)
      tmp = (tmp + 1) % num_pt
    tmp = (t_idx - 1) % num_pt
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
      t_idxs.append(tmp)
      tmp = (tmp - 1) % num_pt
    tt = (max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2
    aux_ext_pts.append(np.array([tt, t]))
    t_band = [item for item in t_band if item not in t_idxs]

  b_band = np.where((b - pts[:, 1]) <= band_thresh * h)[0].tolist()
  while b_band:
    b_idx = b_band[np.argmax(pts[b_band, 1])]
    b_idxs = [b_idx]
    tmp = (b_idx + 1) % num_pt
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
      b_idxs.append(tmp)
      tmp = (tmp + 1) % num_pt
    tmp = (b_idx - 1) % num_pt
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
      b_idxs.append(tmp)
      tmp = (tmp - 1) % num_pt
    bb = (max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2
    aux_ext_pts.append(np.array([bb, b]))
    b_band = [item for item in b_band if item not in b_idxs]

  l_band = np.where((pts[:, 0] - l) <= band_thresh * w)[0].tolist()
  while l_band:
    l_idx = l_band[np.argmin(pts[l_band, 0])]
    l_idxs = [l_idx]
    tmp = (l_idx + 1) % num_pt
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
      l_idxs.append(tmp)
      tmp = (tmp + 1) % num_pt
    tmp = (l_idx - 1) % num_pt
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
      l_idxs.append(tmp)
      tmp = (tmp - 1) % num_pt
    ll = (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2
    aux_ext_pts.append(np.array([l, ll]))
    l_band = [item for item in l_band if item not in l_idxs]

  r_band = np.where((r - pts[:, 0]) <= band_thresh * w)[0].tolist()
  while r_band:
    r_idx = r_band[np.argmax(pts[r_band, 0])]
    r_idxs = [r_idx]
    tmp = (r_idx + 1) % num_pt
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
      r_idxs.append(tmp)
      tmp = (tmp + 1) % num_pt
    tmp = (r_idx - 1) % num_pt
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
      r_idxs.append(tmp)
      tmp = (tmp - 1) % num_pt
    rr = (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2
    aux_ext_pts.append(np.array([r, rr]))
    r_band = [item for item in r_band if item not in r_idxs]

  # assert len(aux_ext_pts) >= 4
  pt0 = aux_ext_pts[0]

  # collecting
  aux_ext_pts = np.stack(aux_ext_pts, axis=0)

  # ordering
  shift_idx = np.argmin(np.power(pts - pt0, 2).sum(axis=1))
  re_ordered_pts = np.roll(pts, -shift_idx, axis=0)

  # indexing
  ext_idxs = np.argmin(np.sum(
    (aux_ext_pts[:, np.newaxis, :] - re_ordered_pts[np.newaxis, ...]) ** 2,
    axis=2),
    axis=1)
  ext_idxs[0] = 0

  ext_idxs = np.sort(np.unique(ext_idxs))

  return re_ordered_pts, ext_idxs

def get_extreme_points(pts):
  l, t = min(pts[:, 0]), min(pts[:, 1])
  r, b = max(pts[:, 0]), max(pts[:, 1])
  # 3 degrees
  thresh = 0.02
  w = r - l + 1
  h = b - t + 1

  t_idx = np.argmin(pts[:, 1])
  t_idxs = [t_idx]
  tmp = (t_idx + 1) % pts.shape[0]
  while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
    t_idxs.append(tmp)
    tmp = (tmp + 1) % pts.shape[0]
  tmp = (t_idx - 1) % pts.shape[0]
  while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
    t_idxs.append(tmp)
    tmp = (tmp - 1) % pts.shape[0]
  tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2, t]

  b_idx = np.argmax(pts[:, 1])
  b_idxs = [b_idx]
  tmp = (b_idx + 1) % pts.shape[0]
  while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
    b_idxs.append(tmp)
    tmp = (tmp + 1) % pts.shape[0]
  tmp = (b_idx - 1) % pts.shape[0]
  while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
    b_idxs.append(tmp)
    tmp = (tmp - 1) % pts.shape[0]
  bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2, b]

  l_idx = np.argmin(pts[:, 0])
  l_idxs = [l_idx]
  tmp = (l_idx + 1) % pts.shape[0]
  while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
    l_idxs.append(tmp)
    tmp = (tmp + 1) % pts.shape[0]
  tmp = (l_idx - 1) % pts.shape[0]
  while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
    l_idxs.append(tmp)
    tmp = (tmp - 1) % pts.shape[0]
  ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2]

  r_idx = np.argmax(pts[:, 0])
  r_idxs = [r_idx]
  tmp = (r_idx + 1) % pts.shape[0]
  while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
    r_idxs.append(tmp)
    tmp = (tmp + 1) % pts.shape[0]
  tmp = (r_idx - 1) % pts.shape[0]
  while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
    r_idxs.append(tmp)
    tmp = (tmp - 1) % pts.shape[0]
  rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2]

  return np.array([tt, ll, bb, rr])

def get_rectangle(x_min, y_min, x_max, y_max):
  rectangle = [
    x_min,
    y_min,  # top-left
    x_min,
    y_max,  # bottom-left
    x_max,
    y_max,  # bottom-right
    x_max,
    y_min,  # top-right
  ]
  return np.array(rectangle).reshape(-1, 2)

def uniformsample(pgtnp_px2, newpnum):
  pnum, cnum = pgtnp_px2.shape
  assert cnum == 2

  idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
  pgtnext_px2 = pgtnp_px2[idxnext_p]
  edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
  edgeidxsort_p = np.argsort(edgelen_p)

  # two cases
  # we need to remove gt points
  # we simply remove shortest paths
  if pnum > newpnum:
    edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
    edgeidxsort_k = np.sort(edgeidxkeep_k)
    pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
    assert pgtnp_kx2.shape[0] == newpnum
    return pgtnp_kx2
  # we need to add gt points
  # we simply add it uniformly
  else:
    edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(
      np.int32)
    for i in range(pnum):
      if edgenum[i] == 0:
        edgenum[i] = 1

    # after round, it may has 1 or 2 mismatch
    edgenumsum = np.sum(edgenum)
    if edgenumsum != newpnum:

      if edgenumsum > newpnum:

        id = -1
        passnum = edgenumsum - newpnum
        while passnum > 0:
          edgeid = edgeidxsort_p[id]
          if edgenum[edgeid] > passnum:
            edgenum[edgeid] -= passnum
            passnum -= passnum
          else:
            passnum -= edgenum[edgeid] - 1
            edgenum[edgeid] -= edgenum[edgeid] - 1
            id -= 1
      else:
        id = -1
        edgeid = edgeidxsort_p[id]
        edgenum[edgeid] += newpnum - edgenumsum

    assert np.sum(edgenum) == newpnum

    psample = []
    for i in range(pnum):
      pb_1x2 = pgtnp_px2[i:i + 1]
      pe_1x2 = pgtnext_px2[i:i + 1]

      pnewnum = edgenum[i]
      wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(
        -1, 1) / edgenum[i]

      pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
      psample.append(pmids)

    psamplenp = np.concatenate(psample, axis=0)
    return psamplenp

def filter_tiny_polys(polys):
  return [poly for poly in polys if Polygon(poly).area > 5]

def get_cw_polys(polys):
    return [
      poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys
    ]


def do_lines_intersect(p1, q1, p2, q2):
  """Check if line segments (p1, q1) and (p2, q2) intersect."""

  def orientation(p, q, r):
    """Find orientation of ordered triplet (p, q, r).
    0 --> p, q and r are collinear
    1 --> Clockwise
    2 --> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
      return 0
    return 1 if val > 0 else 2

  def on_segment(p, q, r):
    """Check if point q lies on line segment pr."""
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
      return True
    return False

  o1 = orientation(p1, q1, p2)
  o2 = orientation(p1, q1, q2)
  o3 = orientation(p2, q2, p1)
  o4 = orientation(p2, q2, q1)

  # General case
  if o1 != o2 and o3 != o4:
    return True

  # Special cases
  if o1 == 0 and on_segment(p1, p2, q1): return True
  if o2 == 0 and on_segment(p1, q2, q1): return True
  if o3 == 0 and on_segment(p2, p1, q2): return True
  if o4 == 0 and on_segment(p2, q1, q2): return True

  return False


def check_self_intersection(polygon):
        # print(polygon[0])

        poly = Polygon(polygon[0].tolist())

        # 使用 shapely 的 is_valid 方法检查多边形是否有效，若无自交，则返回 True
        return poly.is_valid
  # """Check if a polygon (list of points) is self-intersecting."""
  # n = len(polygon)
  # for i in range(n):
  #   for j in range(i + 2, n):
  #     # Avoid checking consecutive edges and the first & last edges in a closed polygon
  #     if (i == 0 and j == n - 1):
  #       continue
  #     p1, q1 = polygon[i], polygon[(i + 1) % n]
  #     p2, q2 = polygon[j], polygon[(j + 1) % n]
  #     if do_lines_intersect(p1, q1, p2, q2):
  #       return True
  # return False
class CTDetDataset(data.Dataset):

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def read_txt(self, txt_path):
        f = open(txt_path)
        line = f.readline()
        data_list = []
        while line:
            num = list(map(float, line.split()))
            data_list.append(num)
            line = f.readline()
        f.close()
        # data_array = np.array(data_list)
        return data_list

  def get_valid_polys(self, instance_polys, inp_out_hw,category_gt):
    output_h, output_w = inp_out_hw[2:]
    instance_polys_ = []
    category_gt_=[]
    for id,instance in enumerate(instance_polys):
      instance = [poly for poly in instance if len(poly) >= 4]

      for poly in instance:
        poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
      polys = filter_tiny_polys(instance)
      polys = get_cw_polys(polys)
      polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
      if polys is None or len(polys) == 0:
          continue
      category_gt_.append(category_gt[id])
      instance_polys_.append(polys)

    return instance_polys_,category_gt_




  def prepare_dance_evolution(self, poly, box, img_init_polys, whs, img_h, img_w, wh, ct_cls, ct_ind,key_masks,target_corner_,poly_ori,targ_poly1):
        if Polygon(poly).exterior.is_ccw:
            # print('poly not in cycle order, discard this poly')
            wh.pop()
            ct_cls.pop()
            ct_ind.pop()
            return

        x_min, y_min, x_max, y_max = box


        # use this as a scaling
        ws = x_max - x_min
        hs = y_max - y_min

        point_num=64
        rectangle = get_rectangle(x_min, y_min, x_max, y_max)
        img_init_poly = uniformsample(rectangle, point_num)

        img_init_poly[:, 0] = np.clip(img_init_poly[:, 0], 0, img_w - 1)
        img_init_poly[:, 1] = np.clip(img_init_poly[:, 1], 0, img_h - 1)


        # 2) deformation target
        img_gt_poly =uniformsample(poly,len(poly) * (point_num) * 5)


        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]  # still over-sampled by up_rate

        img_gt_poly, aux_ext_idxs = get_aux_extreme_points(img_gt_poly)

        tt_idx = np.argmin(np.power(img_init_poly - img_gt_poly[0], 2).sum(axis=1))
        img_init_poly = np.roll(img_init_poly, -tt_idx, axis=0)


        img_gt_poly = single_uniform_multisegment_matching(img_gt_poly, img_init_poly, aux_ext_idxs, 5,point_num)

        # # === 起始点从左上角开始 ===
        lt_idx_init = np.argmin(img_init_poly[:, 0] + img_init_poly[:, 1])
        # lt_idx_init = np.lexsort((img_gt_poly[:, 1], img_gt_poly[:, 0]))[0]

        img_init_poly = np.roll(img_init_poly, -lt_idx_init, axis=0)
        img_gt_poly = np.roll(img_gt_poly, -lt_idx_init, axis=0)


        def is_clockwise(poly):
            """
            判断多边形顶点序列是否为顺时针
            """
            s = 0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                s += (x2 - x1) * (y2 + y1)
            return s > 0  # 如果大于0，则逆时针

        # 保证顺时针
        if is_clockwise(img_init_poly):
            img_init_poly = np.flip(img_init_poly, axis=0)
            img_gt_poly = np.flip(img_gt_poly, axis=0)

            # # === 起始点从左上角开始 ===
            lt_idx_init = np.argmin(img_init_poly[:, 0] + img_init_poly[:, 1])
            # lt_idx_init = np.lexsort((img_gt_poly[:, 1], img_gt_poly[:, 0]))[0]

            img_init_poly = np.roll(img_init_poly, -lt_idx_init, axis=0)
            img_gt_poly = np.roll(img_gt_poly, -lt_idx_init, axis=0)

        img_gt_poly_1 = img_gt_poly#.copy()

        for p in poly:
            # 找到 img_gt_poly 中距离当前 poly 点最近的索引
            dists = np.linalg.norm(img_gt_poly_1 - p, axis=1)
            nearest_idx = np.argmin(dists)
            # 用 poly 中的点替换最近点
            img_gt_poly_1[nearest_idx] = p



        img_gt_poly_1[:, 0] = np.clip(img_gt_poly_1[:, 0], 0, img_w - 1)
        img_gt_poly_1[:, 1] = np.clip(img_gt_poly_1[:, 1], 0, img_h - 1)

        key_mask=DouglasClosedFast().sample(img_gt_poly_1*4)
        key_masks.append(key_mask)


        target_corner_.append(key_mask)
        img_init_polys.append(img_init_poly)
        targ_poly1.append(img_gt_poly_1)
        whs.append(np.array([ws, hs]))

  def __getitem__(self, index):
    img_path = os.path.join(self.data_root, "image", self.img_paths[index])
    gttxt_path = os.path.join(self.data_root, "gt_txt", self.img_paths[index][:-4]+'.txt')

    data_augmentation_flag=1


    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img=img[:,:,::-1]




    image_height,image_width=img.shape[0],img.shape[1]
    margin=0
    crop_size = 512+margin*2
    length_h = max(image_height, crop_size)
    length_w = max(image_width, crop_size)

    img_ = np.zeros((length_h, length_w, 3), dtype=np.uint8)
    img_[margin:image_height+margin, margin:image_width+margin, :] = img
    img = img_
    image_height,image_width=img.shape[0],img.shape[1]


    scale_h=1
    scale_w=1



    input_h, input_w = img.shape[0], img.shape[1]
    output_h, output_w = input_h // 4,input_w // 4


    # gt label
    gt_txt = self.read_txt(gttxt_path)

    polygons_gt = []  # x1,y1,x2,y2,x3,y3.....
    category_gt = []
    polygons_gt_ = []  # x1,y1,x2,y2,x3,y3.....

    for line in gt_txt:
      cla = int(line[0]) - 1

      category_gt.append(cla)

      tem = [[int(line[2 * i + 1]*scale_w+margin), int(line[2 * i + 2]*scale_h+margin)] for i in range(len(line[1:]) // 2)]
      temp = np.array(tem, dtype=np.float32)
      polygons_gt.append([temp])

      for i in range(len(line[1:]) // 2):
        polygons_gt_.append([int(line[2 * i + 1]*scale_w+margin), int(line[2 * i + 2]*scale_h+margin)])

    if len(polygons_gt_)<3:
      index=np.random.randint(0,len(self.img_paths) - 1)
      return self.__getitem__(index)


    scale=1
    if data_augmentation_flag==1:

      c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
      s1 = max(img.shape[0], img.shape[1]) * 1.0
      s2=np.random.choice(np.arange(0.7, 1.30, 0.05))
      w_border = self._get_border(128, img.shape[1])
      h_border = self._get_border(128, img.shape[0])
      c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
      c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      rot = np.random.choice([0, 90, 180, 270, 360])
      scale=s2
      s = s1 * s2

      trans_input = get_affine_transform(c, s, rot, [input_w, input_h])#,shift=np.array([x_shift/input_w, y_shift/input_h], dtype=np.float32)
      #

      inp = cv2.warpAffine(img, trans_input,
                           (input_w, input_h),
                           flags=cv2.INTER_LINEAR)


      img = (inp.astype(np.float32) / 255.)
    else:
      img = (img.astype(np.float32) / 255.)


    if 'test' not in img_path:
      _data_rng = np.random.RandomState(123)
      _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
      _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
      ], dtype=np.float32)
      img=color_aug(_data_rng, img, _eig_val, _eig_vec)

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 3)


    img = (img - mean) / std


    # input: all
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)


    inp=img

    img_id=index

    if data_augmentation_flag == 1:
      new_polygons_gt=[]
      for poly in polygons_gt:
        poly=[affine_transform(np.array(point), trans_input).tolist() for point in poly[0]]
        new_polygons_gt.append([np.array(poly)/4])
      polygons_gt=new_polygons_gt
    else:
      new_polygons_gt=[]
      for poly in polygons_gt:
        poly=[np.array(point).tolist() for point in poly[0]]
        new_polygons_gt.append([np.array(poly)/4])
      polygons_gt=new_polygons_gt

    try:
      assert len(polygons_gt)==len(category_gt)
    except:
      index=np.random.randint(0,len(self.img_paths) - 1)
      print('error 2!')

      return self.__getitem__(index)

    instance_polys,category_gt = self.get_valid_polys(polygons_gt, [image_height,image_width,image_height//4,image_width//4],category_gt)

    try:
      assert len(category_gt)==len(instance_polys)
    except:
      index=np.random.randint(0,len(self.img_paths) - 1)
      print(len(instance_polys))
      print(len(category_gt))

      print('error 1!')
      return self.__getitem__(index)

    num_objs = len(instance_polys)
    if num_objs == 0:
      index=np.random.randint(0,len(self.img_paths) - 1)
      return self.__getitem__(index)


    max_objs=400
    class_num=1


    hm = np.zeros((class_num, output_h, output_w), dtype=np.float32)
    wh = np.zeros((max_objs, 3), dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)

    height_map = np.zeros((output_h, output_w), dtype=np.float32)

    contour_map=np.zeros((output_h, output_w),dtype=np.float32)



    # polygon evolution
    init_box = []
    targ_poly1 = []
    key_masks=[]
    whs = []
    ct_cls = []
    ct_ind = []
    ct_wh=[]

    draw_gaussian = draw_umich_gaussian

    num_objs = min(num_objs, max_objs)

    gt_det = []
    target_corner = []

    instance_polys = [instance_polys[k][0] for k in range(len(instance_polys))]



    for k in range(num_objs):

      poly = instance_polys[k]#[0]
      poly_np=np.array(poly)

      key_mask = DouglasClosedFast().sample(poly_np * 4)
      poly = poly_np[key_mask==1]



      x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
      x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
      bbox = [x_min, y_min, x_max, y_max]

      bbox = np.array(bbox,dtype=np.float32)

      cls_id = category_gt[k]

      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      # bbox = np.array(bbox,dtype=np.float32)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      poly_np=np.array([poly],np.int32)

      ploy_area=com_area(poly)
      if h > 2 and w > 2 and ploy_area>=4:#:
        poly_ori=[]
        for pid in range(len(poly)):
          height_map[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = cls_id

          x=int(round(poly[pid][0]*4))
          y=int(round(poly[pid][1]*4))
          x=x if x<output_w*4 else output_w*4 - 1
          y=y if y<output_h*4 else output_h*4 - 1

          poly_ori.append([x,y])

        poly_np = np.array([poly], np.int32)
        cv2.polylines(contour_map[:,:],poly_np,1,1,1)
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))

        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[0], ct_int, int(radius))


        wh[k] = 1. * w, 1. * h, 1.*cls_id
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1



        if k<32:
            ct_ind.append(ct_int[1] * output_w + ct_int[0])
            ct_cls.append(cls_id)

            ct_wh.append([w, h])
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

            self.prepare_dance_evolution(
              poly,
              bbox,
              init_box,  # decode_boxes[j] is slightly shifted
              whs,
              output_h,
              output_w,
              ct_wh,
              ct_cls,
              ct_ind,
              key_masks,target_corner,poly_ori,targ_poly1)



    ct_num = len(targ_poly1)
    if ct_num<4 or len(targ_poly1)!=len(target_corner):
      index = np.random.randint(0, len(self.img_paths) - 1)
      return self.__getitem__(index)

    fix_box_num=32
    box_num=len(init_box)
    if box_num<fix_box_num:
        for i in range(box_num,fix_box_num):
            init_box.append(init_box[i%box_num])
            whs.append(whs[i%box_num])
            key_masks.append(key_masks[i%box_num])
            targ_poly1.append(targ_poly1[i%box_num])
            target_corner.append(target_corner[i%box_num])
            ct_cls.append(ct_cls[i%box_num])
    n = len(init_box)
    assert all(len(x) == n for x in [
        whs, key_masks, targ_poly1, target_corner, ct_cls
    ])
    if box_num>=fix_box_num:

        idx =random.sample(range(box_num), fix_box_num)  # 随机选 fix_box_num 个索引
        data = list(zip(
            init_box,  whs, key_masks,
            targ_poly1, target_corner, ct_cls
        ))

        data = [data[i] for i in idx]
        (
            init_box,  whs, key_masks,
            targ_poly1, target_corner, ct_cls
        ) = map(list, zip(*data))




    dance_evolution = {
      'init_box': init_box,
      'ct_cls': ct_cls,
      'whs': whs,
      'keymasks':key_masks,
      'targ_poly1':targ_poly1,

    }




    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,'reg': reg}
    ret.update(dance_evolution)


    gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)
    c = np.array([inp.shape[2]/ 2., inp.shape[1] / 2.], dtype=np.float32)
    s = max(inp.shape[1], inp.shape[2]) * 1.0
    meta = {'c': c, 's': s,  'img_id': img_id, 'ct_num': len(init_box),'sc':scale}
    ret['meta'] = meta
    ret.update({'target_corner': target_corner})

    return ret

