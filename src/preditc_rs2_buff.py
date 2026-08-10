
import glob

import os
import numpy as np
import torch.utils.data
import torchvision as tv
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from skimage import measure

import _init_paths
import os
import cv2
from opts import opts
from detectors.detector_factory import detector_factory
import argparse
import time
import numpy as np
# import gdal
import copy
import imageio
import shapefile
from itertools import chain
import cv2
import random
import time
import datetime
import imageio
import shapefile
# from src.polygon_hebing import run_hebing
import torchvision
nms = torchvision.ops.nms
# import nvidia_smi
import math

from torch import Tensor
import torch


def box_area(boxes ):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(box1 , box2):
    area1 = box_area(box1) + 1e-6# N
    area2 = box_area(box2) + 1e-6# M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM

def numpy_nms(boxes , scores , iou_threshold ):
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep

def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)
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


def com_area(contour):
    n = len(contour)
    s = 0
    for i in range(n - 1):
        s = s + contour[i][0] * contour[i + 1][1] - contour[i + 1][0] * contour[i][1]
    s = s + contour[n - 1][0] * contour[0][1] - contour[0][0] * contour[n - 1][1]
    s = math.fabs(s) / 2
    return s



def nms_filter(bboxes, rate_thre = 0.1,IoU_thre=0.5) :
   # [[xmin, ymin, xmax, ymax, conf], [xmin, ymin, xmax, ymax, conf], ...]
    bboxes=np.array(bboxes)


    bboxes = bboxes.reshape((-1, 4))
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)+1e-6
    scores = areas/np.max(areas)  # np.ones((bboxes.shape[0]))
    order = scores.argsort()

    picked_boxes_index = []
    while order.size > 0:
        index = order[-1]
        picked_boxes_index.append(index)
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])

        # 检测框之间的面积比
        rate = areas[index] / areas[order[:-1]]
        # print(rate)
        rate[rate > 1] = 1 / rate[rate > 1]

        w = np.maximum(0.0, x22 - x11)
        h = np.maximum(0.0, y22 - y11)
        intersection = w * h

        # IoU
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        # print(ratio)

        # 相交的检测框里IoU与面积比相近（<rate_thre）的认为存在包含或者高度重叠关系
        # 过滤掉上述检测框以及IoU>IoU_threde
        order = order[:-1][(~((abs(ratio - rate) < rate_thre) * (ratio > 0.))) ]
        # order = order[:-1][(~((abs(ratio - rate) < rate_thre) * (ratio > 0.))) * (ratio <= IoU_thre)]

    return picked_boxes_index


from sklearn.cluster import DBSCAN



def PredictImg(detector,image_path,save_path):

    os.makedirs(save_path, exist_ok=1)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image=image[:,:,::-1]


    width, height = image.shape[1],image.shape[0]
    margin=0
    crop_size=512+margin*2
    length_h = max(height, crop_size)
    length_w = max(width, crop_size)

    img_ = np.zeros((length_h, length_w, 3), dtype=np.uint8)
    img_[margin:height+margin, margin:width+margin, :] = image

    image = img_

    os.makedirs(save_path,exist_ok=1)
    basename=os.path.basename(image_path)

    cut_image_size=crop_size
    img_per_box_num=0

    boxes_all = []
    labels_all = []
    scores_all = []
    masks_all = []
    boxes_nms=[]
    for h in range(0, length_h, cut_image_size):
        for w in range(0, length_w, cut_image_size):

            start_h = h
            start_w = w

            end_h = start_h + cut_image_size
            end_w = start_w + cut_image_size

            if end_h >= length_h:
                end_h = length_h
                start_h = end_h - cut_image_size

            end_h = end_h if end_h >= 0 else 0
            start_h = start_h if start_h >= 0 else 0

            if end_w >= length_w:
                end_w = length_w
                start_w = end_w - cut_image_size

            end_w = end_w if end_w >= 0 else 0
            start_w = start_w if start_w >= 0 else 0

            img=image[start_h:end_h,start_w:end_w,:]
            img=np.array(img,dtype=np.uint8)

            img = (img.astype(np.float32) / 255.)

            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 3)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean) / std

            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)



            with torch.no_grad():
                ret = detector.run(img)
            outputs = ret['outputs']

            result=outputs['detection']#[0]
            polygons=outputs['all_polygons']
            claes=outputs['claes']

            polygons=np.array(polygons,dtype=object)#.numpy()
            result=np.array(result)#.numpy()


            if polygons.shape[0]==0:
                continue

            img_per_box_num+=result.shape[0]

            print(result.shape[0],polygons.shape[0])

            for idx in range(result.shape[0]):
                name=claes[idx]+1

                score=result[idx][4]


                if  score<0.3:
                    continue

                ratio=4

                x1,y1,x2,y2=result[idx][0]*ratio+start_w,result[idx][1]*ratio+start_h,result[idx][2]*ratio+start_w,result[idx][3]*ratio+start_h

                polygon_=polygons[idx]
                if len(polygon_) <= 3:
                    continue
                polygon=[[int(round(polygon_[i][0]))+start_w,int(round(polygon_[i][1]))+start_h] for i in range(len(polygon_))]

                mask=DouglasClosedFast(epsilon=1).sample(np.array(polygon))
                polygon=(np.array(polygon)[mask == 1]).tolist()

                if len(polygon) <= 3:
                    continue

                x1=x1 if x1>=margin else margin
                y1=y1 if y1>=margin else margin
                x2=x2 if x2<=length_w-margin-1 else length_w-margin-1
                y2=y2 if y2<=length_h-margin-1 else length_h-margin-1

                a = [x1, y1, x2, y2]
                boxes_all.append(a)
                polygon.append(polygon[0])

                contours_np = np.array(polygon)
                contours_np[:,0] = np.clip(contours_np[:,0], margin, length_w - margin-1)
                contours_np[:,1] = np.clip(contours_np[:,1], margin, length_h - margin-1)


                polygon=contours_np.tolist()

                x1 = np.min(contours_np[:, 0])
                y1 = np.min(contours_np[:, 1])
                x2 = np.max(contours_np[:, 0])
                y2 = np.max(contours_np[:, 1])

                x1=x1 if x1>=margin else margin
                y1=y1 if y1>=margin else margin
                x2=x2 if x2<=length_w-margin-1 else length_w-margin-1
                y2=y2 if y2<=length_h-margin-1 else length_h-margin-1

                a = [x1, y1, x2, y2]
                b = polygon
                c = score
                d = [name]

                boxes_nms.append(a)
                masks_all.append(b)
                scores_all.append(c)
                labels_all.append(d)

    boxes_nms = np.array(boxes_nms)

    if boxes_nms.shape[0] > 0:


        boxes_save = [boxes_all[i] for i in range(len(boxes_all))]
        scores_save = [scores_all[i].item() for i in range(len(boxes_all))]
        labels_save = [labels_all[i] for i in range(len(boxes_all))]
        masks_save = [masks_all[i] for i in range(len(boxes_all))]

        txt_file_path = save_path + '/' + basename[:-4] + '.txt'
        txt_file = open(txt_file_path, 'w')

        indx = sorted(range(len(scores_save)), key=lambda k: -scores_save[k])


        assert len(boxes_save)==len(masks_save)

        for id, mask in enumerate(masks_save[:]):
            idx = indx[id]
            info = ''
            name = int(labels_save[idx][0])

            info = info + str(name) + ' '
            for point_id in range(len(masks_save[idx])):
                info = info + str(masks_save[idx][point_id][0] - margin) + ' ' + str((masks_save[idx][point_id][1] - margin)) + ' '



            info = info + '\n'
            txt_file.write(info)
        txt_file.close()
    else:

        txt_file_path = save_path + '/' + basename[:-4] + '.txt'

        txt_file = open(txt_file_path, 'w')
        txt_file.close()
#

def get_index(lst=None, item=None):
    return [i for i in range(len(lst)) if lst[i] == item]
def area_element_process_with_hollow(points):

    end_point_index = points.index(points[0], 1)
    is_hollow = end_point_index < len(points) - 1

    if is_hollow:  # 含有空洞

        start_idx = 0
        indexes = get_index(points, points[start_idx])  # 同一个点有多少个索引
        idx = indexes[-1]  # 拿到最后一个
        polygon_points = points[start_idx:idx + 1]
        return polygon_points

    else:
        return points

import torch
import time
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count




if __name__ == "__main__":

    opt = opts().init()



    model_path=r""
    save_path=r""
    image_path=r''
    image_paths=glob.glob(image_path+'/*.jpg')

    # dector
    opt.load_model = model_path
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    start_time = time.time()



    for image_id,image_path in enumerate(image_paths[:]):

        print('process:', image_path,' ',(image_id+1),'/',len(image_paths))

        PredictImg(detector,image_path,save_path)

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f'Prediction time {minutes}:{seconds:04.1f} (mm:ss.s)')

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Prediction time {}'.format(total_time_str))

