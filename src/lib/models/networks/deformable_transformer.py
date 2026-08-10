import copy
import torch
from torch import nn, Tensor
from lib.models.networks.ops.modules import MSDeformAttn,MSDeformAttnv2
import torch.nn.functional as F
dim=256

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.models.networks.relative_position import MultiHeadAttentionLayer,RelativeMultiHeadAttention

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src1,src2=src[0],src[1]
        src=src1
        src2 = self.self_attn(self.with_pos_embed(src2, pos), reference_points, src1, spatial_shapes, level_start_index,padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return [src,src]


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device='cuda')
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableAttnDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,flag=False,posend_type=0):
        super().__init__()
        # cross attention
        self.flag = flag
        if flag:
            self.cross_attn = MSDeformAttnv2(d_model, n_levels, n_heads=n_heads, n_points=n_points,dir=True)

        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads=n_heads, n_points=n_points,flag=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None,
                key_padding_mask=None,dir=None,posend_type=0,lid=0):
        # cross attention
        bs = reference_points.size(0)
        polygon_num = tgt.size(0)
        tgt = tgt.view(bs, -1, dim)
        query_pos = query_pos.view(bs, -1, dim)
        src_padding_mask = src_padding_mask.view(bs, -1)
        if self.flag:
            dir = dir.view(bs, -1, dim)
            # if posend_type==1:
            # tgt=self.with_pos_embed(tgt, query_pos)
            tgt2 = self.cross_attn(tgt,
                                   reference_points,
                                   src, src_spatial_shapes, level_start_index, src_padding_mask,direction_vector=dir)
        else:
            # if posend_type==1:
            # tgt=self.with_pos_embed(tgt, query_pos)
            tgt2 = self.cross_attn(tgt,
                                   reference_points,
                                   src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = tgt.view(polygon_num, -1, dim)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt



class RelPosAttentionCircular(nn.Module):
    """
    Multi-head attention with circular relative positional encoding.
    Input: query/key/value = [B, L, C]
    Special for closed-loop structures (polygon vertices).
    """

    def __init__(self, d_model, n_head, d_head, max_rel_dist=32, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.max_rel_dist = max_rel_dist

        # === linear projections ===
        self.q_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.o_proj = nn.Linear(n_head * d_head, d_model, bias=False)

        # === positional embeddings (circular distance) ===
        self.rel_emb = nn.Embedding(2 * max_rel_dist , d_head)

        # Transformer-XL content & position bias
        self.u_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.v_bias = nn.Parameter(torch.Tensor(n_head, d_head))

        nn.init.normal_(self.u_bias, std=0.02)
        nn.init.normal_(self.v_bias, std=0.02)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------
    # Circular relative distance
    # ------------------------------------------
    def _rel_pos(self, Lq, Lk, device):

        # 计算 j-i（Lq × Lk）
        q_pos = torch.arange(Lq, device=device)[:, None]
        k_pos = torch.arange(Lk, device=device)[None, :]

        diff = k_pos - q_pos   # (Lq, Lk)

        # 循环绝对距离
        # abs_diff = diff.abs()
        # abs_diff = torch.minimum(abs_diff, torch.abs(abs_diff - Lk))

        # 保留符号
        signed_circ = diff#abs_diff * diff.sign()

        signed_circ = signed_circ.clamp(-self.max_rel_dist, self.max_rel_dist)
        # print(torch.max((signed_circ + self.max_rel_dist).long()))

        return (signed_circ + self.max_rel_dist).long()  # shift to [0, 2D]

    def _circular_rel_pos(self, Lq, Lk, device):
        """
        Compute circular relative positions for a closed loop:
        clockwise direction is positive, counter-clockwise is negative
        """
        # query 和 key 的位置索引
        q_pos = torch.arange(Lq, device=device)[:, None]  # (Lq, 1)
        k_pos = torch.arange(Lk, device=device)[None, :]  # (1, Lk)

        # 直接计算顺时针和逆时针距离
        cw_dist = (k_pos - q_pos) % Lk  # 顺时针距离
        ccw_dist = (q_pos - k_pos) % Lk  # 逆时针距离

        # 符号赋值：顺时针为正，逆时针为负
        signed_circ = torch.where(cw_dist <= ccw_dist, cw_dist, -ccw_dist)
        # signed_circ= torch.minimum(cw_dist, ccw_dist)

        # 截断到 [-max_rel_dist, max_rel_dist]
        signed_circ = signed_circ.clamp(-self.max_rel_dist, self.max_rel_dist)
        # print(torch.max((signed_circ + self.max_rel_dist).long()))

        # 平移到 [0, 2*max_rel_dist] 便于 embedding lookup
        # print(signed_circ)
        # print(torch.max(signed_circ))
        return (signed_circ + self.max_rel_dist).long()
    # ------------------------------------------
    # Forward
    # ------------------------------------------
    def forward(self, query, key, value, key_padding_mask=None):
        """
        query: [B, Lq, C]
        key:   [B, Lk, C]
        value: [B, Lk, C]
        """

        B, Lq, _ = query.shape
        _, Lk, _ = key.shape
        device = query.device

        # === projection ===
        q = self.q_proj(query).view(B, Lq, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.n_head, self.d_head).transpose(1, 2)

        # === circular relative position (Lq, Lk) ===
        rel_index = self._circular_rel_pos(Lq, Lk, device)  # (Lq, Lk)
        R = self.rel_emb(rel_index)                         # (Lq, Lk, d_head)

        # === Transformer-XL four terms ===
        # (1) Q * K^T
        score_ac = torch.einsum("bnqd,bnkd->bnqk", q, k)  # [B, n_head, Lq, Lk]

        # (2) Q * R^T
        score_bd = torch.einsum("bnqd,qkd->bnqk", q, R)  # [B, n_head, Lq, Lk]

        # (3) u_bias * K^T
        # 先 expand u_bias，直接加到 Q 上再 matmul
        q_u = q + self.u_bias[None, :, None, :]  # [B, n_head, Lq, d_head]
        score_u = torch.einsum("bnqd,bnkd->bnqk", q_u, k)  # [B, n_head, Lq, Lk]

        # (4) v_bias * R^T
        # 先 expand v_bias
        v_bias_expand =self. v_bias[None, :, None, :]  # [1, n_head, 1, d_head]
        score_v = torch.einsum("bnqd,qkd->bnqk", v_bias_expand, R)  # [B, n_head, Lq, Lk]

        # 最终 attention score
        score = (score_ac + score_bd + score_u + score_v) / (self.d_head ** 0.5)

        # === optional mask ===
        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask == 0, float('-inf'))

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # === attention output ===
        out = torch.einsum("bnij,bnjd->bnid", attn, v)

        out = out.transpose(1, 2).reshape(B, Lq, self.n_head * self.d_head)
        out = self.o_proj(out)

        return out#, attn


def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # 映射到闭环：0 → 0 rad, max_len → 2π rad
    # angle = 2 * math.pi * position / max_len

    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position  * theta  # 即公式里的：pos / (10000^(2i/d))
    # print(embeddings)

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)



    return embeddings
def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k
class RoPEMultiHeadAttentionV2(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        # self.dropout = nn.Dropout(dropout)
        # self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, use_RoPE=True,key_padding_mask=None):
# def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)
        B, L, D = query.shape
        # print(query.size())

        # project Q,K,V
        # qkv = self.qkv_proj(torch.cat([query, key, value], dim=1))  # concat to project simultaneously
        # q, k, v = torch.chunk(qkv, 3, dim=-1)
        # print(q.size())
        q = self.q_proj(query)  # .view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # reshape to multi-head
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,L,Dh]
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if use_RoPE:
            q, k = RoPE(q, k)

        d_k = k.size()[-1]

        att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
        att_logits /= math.sqrt(d_k)

        if key_padding_mask is not None:
            att_logits = att_logits.masked_fill(key_padding_mask == 0, -1e9)  # mask掉为0的部分，设为无穷大

        att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)
        out=torch.matmul(att_scores, v).transpose(1, 2).reshape(B,L,D).contiguous()
        #
        # if dropout is not None:
        #     att_scores = dropout(att_scores)

        # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
        out=self.out_proj(out)

        return out#, att_scores



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,flag=False,posend_type=0):
        super().__init__()
        # cross attention
        self.flag = flag
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads=n_heads, n_points=n_points,flag=flag)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        #
        # self.self_attn=RoPEAttention(d_model, n_heads)
        if posend_type==2:
            self.posend_type=posend_type
            # self.self_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            # self.self_linear1 = nn.Linear(d_model, d_model//2)
            # self.self_linear2 = nn.Linear(d_model, d_model//2)
            # self.self_norm = nn.LayerNorm(d_model//2)
            # self.fuse_linear = nn.Linear(d_model, d_model)
            #
            self.self_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout=dropout)#Mamba_Head(d_model)#

        elif posend_type==1:
            self.posend_type=posend_type
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.self_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout=dropout)  # Mamba_Head(d_model)#
        #
        # self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)#RelativeMultiHeadAttention(d_model, n_heads, dropout=dropout)#Mamba_Head(d_model)#

        # self.self_attn2 =  RelativeMultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self.alpha = nn.Parameter(torch.tensor(0.0))


    # @staticmethod
    def with_pos_embed(self,tensor, pos):
        # print(tensor.shape,pos.shape)
        return tensor if pos is None else tensor + pos

        # if self.posend_type==1:
        #     return tensor if pos is None else tensor + pos
        # if self.posend_type==2:
        #     # print(self.alpha)
        #     return tensor if pos is None else tensor + self.alpha*pos


    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # def precompute_rope_matrix(self,max_len, d_model):
    #     """
    #     生成旋转位置编码矩阵
    #     参数：
    #         max_len: 最大position embedding长度，默认为32768。
    #         d_model: 嵌入维度，默认为512。
    #     返回：
    #         旋转位置编码矩阵
    #     """
    #     theta = 10000.0
    #     # 频率,决定不同维度的旋转速度 shape: (d_model//2,)
    #     freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
    #     # 位置,[0, 1, 2, ..., max_len-1]  shape: (max_len, 1)
    #     t = torch.arange(max_len, dtype=torch.float32)
    #     # 计算位置和频率的外积，得到每个位置在各维度的角度值 freqs[m,i] = t[m] * freqs[i]，表示位置 m 在维度 i的旋转角度
    #     # freqs[m,i] 即旋转位置编码矩阵, 表示位置和频率的乘积 [max_len, d_model//2]
    #     freqs = torch.outer(t, freqs).float()
    #     # 计算余弦和正弦值，freqs_sin 和 freqs_cos 是预计算的正弦和余弦矩阵，用于对查询（query）和键（key）向量进行旋转操作。
    #     # freqs_sin 和 freqs_cos 具体含义：存储每个位置在不同维度上的旋转角度的正弦和余弦值
    #     # freqs_sin: [max_len, d_model]
    #     # cat的原因：保持freqs_sin的前半部分和后半部分存储相同的正弦/余弦值
    #     freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    #     freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    #     return freqs_sin, freqs_cos

    # def rotate_half(self,x):
    #         # 假设q的最后一个维度是2d，这里（x_i,x_i+d)是共享同一个旋转角度的向量对
    #         # RoPE介绍是相邻旋转,（x_i,x_i+1)一对，本质一样，都可以实现Rope的核心思想
    #         x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    #         return torch.cat([-x2, x1], dim=-1)

    # def apply_rope(self,q, k, sin, cos):
    #     """
    #     应用旋转位置编码
    #     参数：
    #         q: 查询张量，形状为 [batch_size, seq_len, num_heads, head_dim]
    #         k: 键张量，形状为 [batch_size, seq_len, num_heads, head_dim]
    #         cos: 预计算的余弦值张量，形状为 [seq_len, head_dim]
    #         sin: 预计算的正弦值张量，形状为 [seq_len, head_dim]
    #         position_ids: 可选的位置索引张量，形状为 [seq_len]
    #     """
    #
    #
    #
    #     batch_size, seq_len, _, _ = q.shape
    #     # 截取前 seq_len 个位置的余弦值
    #     cos = cos[:seq_len]
    #     # 截取前 seq_len 个位置的正弦值
    #     sin = sin[:seq_len]
    #     # 调整cos和sin的形状以匹配q和k的广播需求
    #     cos = cos.unsqueeze(1)  # shape: (seq_len, 1, d_model)
    #     sin = sin.unsqueeze(1)  # shape: (seq_len, 1, d_model)
    #
    #     # 实现旋转
    #     q = q * cos + self.rotate_half(q) * sin
    #     k = k * cos + self.rotate_half(k) * sin
    #     return q, k


    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None,
                key_padding_mask=None,
                get_image_feat=True,dir=None,posend_type=0,lid=0):


        if posend_type==2:
            if get_image_feat==True:
                q = k = self.with_pos_embed(tgt, query_pos)
            else:
                q = k = tgt

            v=tgt

            tgt2 = self.self_attn(q, k, v, key_padding_mask=key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)



        elif posend_type==1:
            if get_image_feat==True:
                q = k = self.with_pos_embed(tgt, query_pos)

            else:
                q = k = tgt

            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  key_padding_mask=key_padding_mask)[0].transpose(0, 1)

            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if get_image_feat==True:
             tgt=self.with_pos_embed(tgt, query_pos)

        else:
            tgt=tgt

        if get_image_feat:
            # cross attention

            # print('here!')
            # print(reference_points.size())
            # print(tgt.size())
            # print(query_pos.size())
            # print(src_padding_mask.size())

            bs = reference_points.size(0)
            polygon_num = tgt.size(0)
            tgt = tgt.view(bs, -1, dim)
            # query_pos = query_pos.view(bs, -1, dim)
            src_padding_mask = src_padding_mask.view(bs, -1)

            tgt1 = self.cross_attn(tgt,
                                   reference_points,
                                   src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt = tgt + self.dropout1(tgt1)# tgt1
            tgt = self.norm1(tgt)

            tgt = tgt.view(polygon_num, -1, dim)
            # query_pos = query_pos.view(polygon_num, -1, dim)
            # src_padding_mask = src_padding_mask.view(bs, -1)


        # q = k =  self.with_pos_embed(tgt,query_pos)
        # q= k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn2(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
        #                            key_padding_mask=key_padding_mask)[0].transpose(0, 1)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)


        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, with_sa=True,posend_type=0):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.with_sa = with_sa
        self.posend_type=posend_type

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, key_padding_mask=None, get_image_feat=True,dir=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.with_sa:
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                               src_padding_mask, key_padding_mask, get_image_feat,dir,self.posend_type,lid=lid)
            else:
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                               src_level_start_index,
                               src_padding_mask, key_padding_mask,dir,self.posend_type,lid=lid)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
