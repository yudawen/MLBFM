import math
import numpy as np

class DouglasClosed:
    def __init__(self, epsilon=1.0):
        self.D = epsilon

    @staticmethod
    def point_line_distance(p, a, b):
        ab = b - a
        ap = p - a
        denom = np.dot(ab, ab)
        if denom < 1e-8:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def _dp(self, i1, i2, poly, mask):
        if i2 <= i1 + 1:
            return

        a = poly[i1]
        b = poly[i2]

        max_dist = -1.0
        max_idx = -1

        for i in range(i1 + 1, i2):
            d = self.point_line_distance(poly[i], a, b)
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > self.D:
            mask[max_idx] = 1
            self._dp(i1, max_idx, poly, mask)
            self._dp(max_idx, i2, poly, mask)

    def _make_path(self, start, end, N):
        """生成 start -> end 的环路径（含两端）"""
        if start <= end:
            return list(range(start, end + 1))
        else:
            return list(range(start, N)) + list(range(0, end + 1))

    def sample(self, poly):
        """
        poly: (N,2) 闭合多边形（不重复首尾）
        return: mask (N,)  1=保留, 0=移除
        """
        N = len(poly)
        mask = np.zeros(N, dtype=int)

        # 1. 选两个最远点作为断点
        dist_mat = np.sum((poly[:, None] - poly[None]) ** 2, axis=2)
        i0, i1 = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)

        # 2. 两条弧
        path1 = self._make_path(i0, i1, N)
        path2 = self._make_path(i1, i0, N)

        # 3. 对两条路径分别做 DP
        for path in (path1, path2):
            sub_poly = poly[path]
            sub_mask = np.zeros(len(sub_poly), dtype=int)
            sub_mask[0] = 1
            sub_mask[-1] = 1

            self._dp(0, len(sub_poly) - 1, sub_poly, sub_mask)

            for k, idx in enumerate(path):
                if sub_mask[k]:
                    mask[idx] = 1

        return mask

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

class Douglas:
    D = 1
    def sample(self, poly):
        mask = np.zeros((poly.shape[0],), dtype=int)
        mask[0] = 1
        endPoint = poly[0: 1, :] + poly[-1:, :]
        endPoint /= 2
        poly_append = np.concatenate([poly, endPoint], axis=0)
        self.compress(0, poly.shape[0], poly_append, mask)
        return mask

    def compress(self, idx1, idx2, poly, mask):
        p1 = poly[idx1, :]
        p2 = poly[idx2, :]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        m = idx1
        n = idx2
        if (n == m + 1):
            return
        d = abs(A * poly[m + 1: n, 0] + B * poly[m + 1: n, 1] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2) + 1e-4)
        max_idx = np.argmax(d)
        dmax = d[max_idx]
        max_idx = max_idx + m + 1

        if dmax > self.D:
            mask[max_idx] = 1
            self.compress(idx1, max_idx, poly, mask)
            self.compress(max_idx, idx2, poly, mask)
