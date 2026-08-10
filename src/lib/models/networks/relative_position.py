
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position=63):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class CircularRelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position=32):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 , num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        # assume circular structure, so lengths must match
        L = length_k

        # [L, 1], [1, L]
        q_pos = torch.arange(L)[:, None]
        k_pos = torch.arange(L)[None, :]

        # circular distance: between -L/2 ... L/2
        diff = (k_pos - q_pos + L // 2) % L - L // 2

        # clip to max relative range
        diff = torch.clamp(diff, -self.max_relative_position, self.max_relative_position)

        idx = diff + self.max_relative_position
        idx = idx.long().cuda()

        return self.embeddings_table[idx]

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.1, device='cuda'):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.relative_position_k = CircularRelativePosition(self.head_dim)#RelativePosition(self.head_dim)#
        self.relative_position_v = CircularRelativePosition(self.head_dim)#RelativePosition(self.head_dim)#CircularRelativePosition(self.head_dim)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, key_padding_mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x

class RelativePositionv2(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2+1 , num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k, device=None):
        # 生成位置索引
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)

        # 普通相对位置
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]  # (Lq, Lk)

        # ✅ 环形最短距离（对称到 [-L//2, L//2]）
        seq_len = max(length_q, length_k)
        half = seq_len // 2
        distance_mat = ((distance_mat + half) % seq_len) - half
        # print(torch.max(distance_mat))
        # return
        #
        # 限制在 [-max_relative_position, max_relative_position]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )

        # 偏移到正数区间，作为 embedding 索引
        final_mat = distance_mat_clipped + self.max_relative_position
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        "Take in model size and number of heads."
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        self.linears = nn.Sequential(nn.Linear(d_model, d_model),nn.Linear(d_model, d_model),nn.Linear(d_model, d_model),nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.relative_position_k = RelativePositionv2(self.head_dim, max_relative_position=32)
        self.relative_position_v = RelativePositionv2(self.head_dim, max_relative_position=32)
        self.batch_size=1
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, query, key, value,key_padding_mask=None, value_padding_mask=None):
        # embedding
        # query, key, value = [batch_size, len, hid_dim]
        self.batch_size = query.size(0)
        query, key, value = [l(x).view(self.batch_size, -1, self.d_model) for l, x in zip(self.linears, (query, key, value))]

        len_k = query.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        # Self-Attention
        # r_q1, r_k1 = [batch_size, len, n_heads, head_dim]
        r_q1 = query.view(self.batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(self.batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, self.batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(self.batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale
        # attn = attn1/ self.scale

        attn = self.dropout(torch.softmax(attn, dim=-1))
        # attn = [batch_size, n_heads, len, len]
        r_v1 = value.view(self.batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, self.batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(self.batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(self.batch_size * len_q, self.d_model)
        # x = [batch size * query len, hid dim]
        x=self.linears[-1](x).view(self.batch_size,len_q,self.d_model)
        # print(x.size())

        return x