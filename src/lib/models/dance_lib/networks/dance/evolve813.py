
from lib.models.dance_lib.utils.dance import dance_config, dance_gcn_utils, dance_decode
from lib.models.dance_lib.utils.snake import snake_config, snake_decode

import numpy as np
from lib.models.networks.mlp import MLP
from lib.models.networks.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DeformableAttnDecoderLayer
from lib.models.networks.ops.modules import MSDeformAttn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import torch.nn.functional as F
from lib.models.networks.utils.misc import NestedTensor

import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d

dim=256
class_num=10
import copy
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).bool().to(x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def flatten(feats):
    """
    feats: list of (B,C,H,W)
    return: (B,N,C)
    """
    outs = []

    for f in feats:
        B, C, H, W = f.shape
        outs.append(f.flatten(2).transpose(1,2))

    return torch.cat(outs, dim=1)



def generate_uniform_points_in_bbox(polygons, N, scores):
    """
    polygons: Tensor, shape [polygon_num, point_num, 2]
    N: int, 每个多边形生成的点数量
    return: Tensor, shape [polygon_num, N, 2]
    """
    polygon_num = polygons.shape[0]

    # 计算每个多边形的包围矩形
    x_min = polygons[:, :, 0].min(dim=1)[0].unsqueeze(1)
    x_max = polygons[:, :, 0].max(dim=1)[0].unsqueeze(1)
    y_min = polygons[:, :, 1].min(dim=1)[0].unsqueeze(1)
    y_max = polygons[:, :, 1].max(dim=1)[0].unsqueeze(1)


    # 在矩形范围内生成均匀分布随机点
    rand_points = torch.rand(polygon_num, N, 2).cuda()
    rand_points[:, :, 0] = scores[:,:,0] * (x_max - x_min) + x_min
    rand_points[:, :, 1] = scores[:,:,1] * (y_max - y_min) + y_min

    return rand_points

class Corner_regression(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_feature_levels, backbone_strides, backbone_num_channels):
        super(Corner_regression, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)

        self.transformer = CornerTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=2,
                                           num_decoder_layers=1, dim_feedforward=256, dropout=0.1)





    @staticmethod
    def get_ms_feat(xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def forward(self, image_feats, pixel_features, feat_mask, corner_coords, poly_ind,cnn_feature,gt_poly_points=None,seg_embeddings=None):

        # cnn_feature_4s=cnn_feature
        features = self.get_ms_feat(image_feats, feat_mask)

        srcs = []
        masks = []
        all_pos = []

        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            srcs.append(self.input_proj[l](src))
            pos = self.img_pos(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_pos(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)

        batch_size = pixel_features.size(0)
        num_polygon = corner_coords.size(0)
        num_point = corner_coords.size(1)


        h,w=pixel_features.size(2),pixel_features.size(3)
        geo_pos = dance_gcn_utils.get_gcn_feature(pixel_features, corner_coords, poly_ind, h, w)
        geo_pos=geo_pos.permute(0,2,1).contiguous()

        corner_coords_norm = corner_coords /feat_mask.shape[1]
        normalized_size=feat_mask.shape[1]
        corner_inputs = geo_pos#self.corner_input_fc(geo_pos)




        pred_location_vertex_ ,pred_classification_vertex_,pred_classification_function_= self.transformer(srcs,masks,all_pos,corner_inputs,corner_coords_norm,batch_size,num_polygon,num_point,poly_ind,corner_coords,pixel_features,normalized_size,seg_embeddings)


        return pred_location_vertex_,pred_classification_vertex_,pred_classification_function_


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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class CornerTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=3, dec_n_points=4, enc_n_points=4,
                 ):
        super(CornerTransformer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)


        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          3, nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, 3)



        posend_type=2
        decoder_vertex = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, n_heads=nhead, n_points=dec_n_points,posend_type= posend_type)

        self.repeat_num=3
        vertex_decoder=[]


        for i in range(self.repeat_num):
            vertex_decoder.append(DeformableTransformerDecoder(decoder_vertex, num_decoder_layers,return_intermediate_dec, with_sa=True,posend_type=posend_type))


        self.vertex_decoder=nn.ModuleList(vertex_decoder)


        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))



        self.output_fc_vertex =nn.ModuleList([MLP(input_dim=d_model, hidden_dim=d_model , output_dim=2, num_layers=2,predict=True) for i in range(self.repeat_num)])
        self.output_cl_vertex =nn.ModuleList([MLP(input_dim=d_model, hidden_dim=d_model , output_dim=1, num_layers=2,predict=True) for i in range(self.repeat_num)])


        self.vertex_embed = nn.Embedding(64, 256)

        self.vertex_embed2 = nn.Embedding(1, 256)

        encoder_layer2 = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          3, nhead, enc_n_points)

        self.encoder2 = DeformableTransformerEncoder(encoder_layer2, 3)

        posend_type = 1
        decoder_vertex2 = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                           dropout, activation,
                                                           num_feature_levels, n_heads=nhead, n_points=dec_n_points,
                                                           posend_type=posend_type)

        self.repeat_num2 = 3
        vertex_decoder2 = []

        for i in range(self.repeat_num2):
            vertex_decoder2.append(
                DeformableTransformerDecoder(decoder_vertex2, num_decoder_layers, return_intermediate_dec, with_sa=True,
                                             posend_type=posend_type))

        self.vertex_decoder2 = nn.ModuleList(vertex_decoder2)

        self.level_embed2 = nn.Parameter(torch.Tensor(num_feature_levels, d_model))


        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU())
        self.output_fc_vertex2 = nn.ModuleList(
            [MLP(input_dim=d_model, hidden_dim=d_model, output_dim=class_num, num_layers=2, predict=True) for i in
             range(self.repeat_num2)])
        self.conv_out = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=class_num, out_channels=256, kernel_size=3,
                              bias=False, padding=1),nn.ReLU()) for _ in range(3)
        ])




    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)
    def get_query_pos_embed(self, ref_points):
            """
            Generates sinusoidal positional embeddings for the reference points.
            """
            num_pos_feats = 128
            temperature = 10000
            scale = 2 * math.pi

            dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
            dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # [128]
            ref_points = ref_points * scale
            pos = ref_points[:, :, :, None] / dim_t
            pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
            return pos
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def inverse_sigmoid(self,x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def cartesian_to_relative_polar(self,polygons):
        """
        将多边形坐标转换为以多边形中心为原点的极坐标形式

        参数:
            polygons: torch.Tensor, shape [polygon_num, point_num, 2]
                      最后一维是 (x, y)

        返回:
            polar_polygons: torch.Tensor, shape [polygon_num, point_num, 2]
                            最后一维是 (r, theta)
        """
        # 1. 计算每个多边形的中心
        center = polygons.mean(dim=1, keepdim=True)  # [polygon_num, 1, 2]

        # 2. 计算相对坐标
        rel_coords = polygons - center  # [polygon_num, point_num, 2]
        x = rel_coords[..., 0]
        y = rel_coords[..., 1]

        # 3. 转换为极坐标
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)  # [-pi, pi]

        # 4. 拼成极坐标
        polar_polygons = torch.stack([r, theta], dim=-1)

        return polar_polygons

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def cyclic_sinusoidal_position_encoding(self,seq_len, d_model):
        """
        生成循环序列版本的正弦位置编码 (Sinusoidal Positional Encoding)

        参数:
            seq_len: 序列长度
            d_model: 嵌入维度
        返回:
            pe: [seq_len, d_model] 的位置编码矩阵
        """
        # 位置索引 (seq_len, 1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]

        # 频率索引 (1, d_model//2)
        div_term = 2 * math.pi * torch.arange(0, d_model, 2).float() / seq_len  # [d_model//2]

        # 初始化位置编码矩阵
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        return pe


    def forward(self, srcs,masks,pos_embeds,query_embed,reference_points,batch_size,num_polygon,num_point,poly_ind,corner_coords,pixel_features,normalized_size=128,seg_embeddings=None):

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []

        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes_=spatial_shapes
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)


        h, w = pixel_features.size(2),pixel_features.size(3)

        if seg_embeddings !=[None,None] and 1:

            seg_embeddings_,seg_embeddings_ori=seg_embeddings[0],seg_embeddings[1]
            seg_embeddings_ori=seg_embeddings_ori.view(bs, class_num, 64, 64)


            seg_embeddings_ori_4s=self.conv_out[0](F.interpolate(seg_embeddings_ori,spatial_shapes_[0]))
            seg_embeddings_ori_8s=self.conv_out[1](F.interpolate(seg_embeddings_ori,spatial_shapes_[1]))
            seg_embeddings_ori_16s=self.conv_out[2](F.interpolate(seg_embeddings_ori,spatial_shapes_[2]))


            seg_embeddings_ori_list=torch.cat([seg_embeddings_ori_4s.flatten(2).transpose(1, 2),seg_embeddings_ori_8s.flatten(2).transpose(1, 2),seg_embeddings_ori_16s.flatten(2).transpose(1, 2)],dim=1)

            memory = self.encoder([src_flatten,seg_embeddings_ori_list], spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,mask_flatten)[0]
        else:
            memory = self.encoder([src_flatten,src_flatten], spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,mask_flatten)[0]

        bs, _, c = memory.shape

        """
        memory: (B, N, C)
        spatial_shapes: [(H1,W1), (H2,W2), (H3,W3)]
        """

        B, N, C = memory.shape



        box_num_each_batch=num_polygon//batch_size
        batchsize=batch_size



        reference_points=reference_points.view(batchsize, box_num_each_batch*num_point,  2)
        geo_pos = dance_gcn_utils.get_gcn_feature(pixel_features,corner_coords , poly_ind, h, w)
        pos_embed=geo_pos.permute(0,2,1).contiguous()

        vertex_embed =self.vertex_embed.weight.unsqueeze(0).expand(num_polygon, -1, -1).contiguous()#.repeat(num_polygon,1,1)#.expand(num_polygon, -1, -1)  # (B, num_queries, C).repeat(num_polygon,1,1)#

        pred_location_vertex_=[]

        pred_classification_vertex_=[]
        pred_classification_function_=[]

        #vertex decoder
        tgt_vertex=vertex_embed
        pred_location_vertex=corner_coords*snake_config.ro

        # vertex
        for i in range(self.repeat_num):
            tgt_vertex = tgt_vertex
            tgt_vertex, _ = self.vertex_decoder[i](tgt_vertex, reference_points, memory,
                                                spatial_shapes, level_start_index, valid_ratios, pos_embed,
                                                mask_flatten,key_padding_mask=None, get_image_feat=True)
            tgt=tgt_vertex
            # if i>1:

            pred_location_vertex_offset = self.output_fc_vertex[i](tgt)
            pred_location_vertex_class_=self.output_cl_vertex[i](tgt)
            if i>0:
                pred_location_vertex_class=pred_location_vertex_class+pred_location_vertex_class_
            else:
                pred_location_vertex_class=pred_location_vertex_class_
            pred_classification_vertex_.append(pred_location_vertex_class)
            pred_location_vertex = pred_location_vertex_offset + pred_location_vertex
            pred_location_vertex_.append(pred_location_vertex)

            reference_points = pred_location_vertex /snake_config.ro / normalized_size

            reference_points = reference_points.contiguous().view(batchsize, box_num_each_batch * num_point, 2)
            geo_pos = dance_gcn_utils.get_gcn_feature(pixel_features,pred_location_vertex/snake_config.ro , poly_ind, h, w)
            pos_embed=geo_pos.permute(0,2,1).contiguous()

        #function

        h, w = pixel_features.size(2), pixel_features.size(3)


        memory2 = self.encoder2([src_flatten,src_flatten], spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,mask_flatten)[0]

        bs, _, c = memory2.shape


        box_num_each_batch = num_polygon // batch_size
        batchsize = batch_size

        pred_location_center=torch.cat([torch.mean(pred_location_vertex[:,:,0],dim=1).unsqueeze(-1).unsqueeze(-1),torch.mean(pred_location_vertex[:,:,1],dim=1).unsqueeze(-1).unsqueeze(-1)],dim=-1)
        reference_points = pred_location_center / snake_config.ro / normalized_size
        reference_points = reference_points.contiguous().view(batchsize, box_num_each_batch * 1, 2)

        geo_pos = dance_gcn_utils.get_gcn_feature(pixel_features, pred_location_center / snake_config.ro, poly_ind, h, w)

        pos_embed = geo_pos.permute(0, 2, 1).contiguous()
        pos_embed=pos_embed.contiguous().view(batchsize, box_num_each_batch * 1, 256)


        tgt_vertex =self.vertex_embed2.weight.unsqueeze(0).expand(num_polygon, -1, -1).contiguous()#.repeat(num_polygon,1,1)#.expand(num_polygon, -1, -1)  # (B, num_queries, C).repeat(num_polygon,1,1)#


        tgt_vertex=tgt_vertex.view(batchsize, box_num_each_batch * 1, 256)

        for i in range(self.repeat_num2):
            tgt_vertex, _ = self.vertex_decoder2[i](tgt_vertex, reference_points, memory2,
                                                   spatial_shapes, level_start_index, valid_ratios, pos_embed,
                                                   mask_flatten, key_padding_mask=None, get_image_feat=True)
            tgt = tgt_vertex

            pred_location_vertex_function = self.output_fc_vertex2[i](tgt)
            pred_classification_function_.append(pred_location_vertex_function)

            if seg_embeddings != [None, None] and 1:
                seg_embeddings_, seg_embeddings_ori = seg_embeddings[0], seg_embeddings[1]

                tgt2=self.out_proj(tgt)
                tgt_norm = F.normalize(tgt2, dim=-1)
                cla_vertex = seg_embeddings_

                cla_norm = F.normalize(cla_vertex, dim=-1)
                logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
                pred_location_vertex_function_ = logit_scale*torch.matmul(tgt_norm, cla_norm.transpose(1, 2))
                pred_classification_function_.append(pred_location_vertex_function_)
            else:
                pred_classification_function_.append([])





        return pred_location_vertex_,pred_classification_vertex_,pred_classification_function_











import math
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
class ResdiualBlock(nn.Module):
    """
    实现子module：Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResdiualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class Dance(nn.Module):
    def __init__(self):
        super(Dance, self).__init__()

        self.corner_regression_model = Corner_regression(input_dim=dim, hidden_dim=dim, num_feature_levels=3, backbone_strides=[1, 2, 4],backbone_num_channels=[256,512,768])



    def prepare_training(self, output, batch):
        # batch all the polys && label the their indices (to each image)
        init = dance_gcn_utils.prepare_training(output, batch)
        output.update({
            'init_box': init['init_box'],
            'targ_poly': init['targ_poly']
        })
        return init

    def prepare_training_evolve(self, output, batch, init):
        ct_num = batch['meta']['ct_num'].sum()
        evolve = dance_gcn_utils.prepare_training_evolve(
            output['ex_pred'], init, ct_num)
        output.update({
            'i_it_py': evolve['i_it_py'],
            'c_it_py': evolve['c_it_py'],
            'i_gt_py': evolve['i_gt_py']
        })
        evolve.update({'ind': init['ind'][:evolve['i_gt_py'].size(0)]})
        return evolve

    def remove_rectangle_corners_torch(self,points: torch.Tensor):
        """
            输入:
                points: (B, N, 2) torch.Tensor，沿矩形边缘均匀采样，顺时针或逆时针
            输出:
                new_points: (B, N-4, 2) 去掉四个角点后的张量
                corner_indices: list，长度为B，每个元素是4个角点索引的张量
            """
        B, N, _ = points.shape

        # 相邻方向向量
        next_points = torch.roll(points, shifts=-1, dims=1)  # (B, N, 2)
        dirs = next_points - points  # (B, N, 2)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # 相邻方向点积
        prev_dirs = torch.roll(dirs, shifts=1, dims=1)
        dot_vals = (dirs * prev_dirs).sum(dim=-1)  # (B, N)

        # 拐角点：方向变化接近90°
        corner_mask = torch.abs(dot_vals) < 0.1  # 阈值可调

        corner_indices = []
        new_points_list = []

        for b in range(B):
            inds = torch.nonzero(corner_mask[b], as_tuple=False).squeeze(-1)

            # 只保留4个角点
            if inds.numel() > 4:
                step = max(1, inds.numel() // 4)
                inds = inds[::step][:4]

            corner_indices.append(inds)

            # 去掉角点
            mask = torch.ones(N, dtype=torch.bool, device=points.device)
            mask[inds] = False
            new_points_list.append(points[b][mask])

        new_points = torch.stack(new_points_list, dim=0)  # (B, N-4, 2)

        return new_points, corner_indices
    def prepare_testing_locations(self, output):
        # if len(output['cp_box']) == 0:
        # print(output['detection'].shape)

        box = output['detection'][..., :4]
        score = output['detection'][..., 4]
        ind = score > 0.05
        # ind = score > snake_config.ct_score
        i_it_4py = dance_decode.get_init(box)

        i_it_4py = i_it_4py[ind]
        if len(i_it_4py) == 0:
            init = {
                'i_it_4py':
                i_it_4py,
                'ind':
                torch.cat([
                    torch.full([ind[i].sum()], i) for i in range(ind.size(0))
                ],
                          dim=0)
            }
            output.update({'it_location': init['i_it_4py']})
            tmp = {'boxes': i_it_4py}
            init.update(tmp)
            return init

        i_it_4py = i_it_4py[None]

        # print('i_it_4py after', i_it_4py.shape)

        tmp = {'boxes': i_it_4py[0]}
        # print(i_it_4py.shape)
        i_it_4py = dance_gcn_utils.uniform_upsample(i_it_4py,64)
        # c_it_4py = dance_gcn_utils.img_poly_to_can_poly(i_it_4py)

        i_it_4py = i_it_4py[0]
        # print(i_it_4py.shape)
        # i_it_4py,_=self.remove_rectangle_corners_torch(i_it_4py)
        # print(i_it_4py.shape)
        # print(ind.shape)

        # c_it_4py = c_it_4py[0]
        ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
        # print(ind)
        init = {'i_it_4py': i_it_4py, 'ind': ind}
        init.update(tmp)
        output.update({'it_location': init['i_it_4py']})

        # output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        # output['detection']=output['detection'].squeeze(0)
        # output['ct']=output['ct'].squeeze(0)
        mask=output['detection'][..., 4] > 0.05

        output['detection'] = output['detection'][mask]
        output['ct'] = output['ct'][mask]
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w - 1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h - 1)
        evolve = dance_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve

    def de_location(self, locations):
        # de-location (spatial relationship among locations; translation invariant)
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        x_max = torch.max(locations[..., 0], dim=-1)[0]
        y_max = torch.max(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()

        new_locations[..., 0] = (new_locations[..., 0] - x_min[..., None]) / \
                                (x_max[..., None] - x_min[..., None])
        new_locations[..., 1] = (new_locations[..., 1] - y_min[..., None]) / \
                                (y_max[..., None] - y_min[..., None])
        return new_locations

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = dance_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly,ind, h, w)
        c_it_poly = c_it_poly * snake_config.ro
        init_input = torch.cat(
            [init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = dance_gcn_utils.get_adj_ind(snake_config.adj_num,
                                          init_input.size(2),
                                          init_input.device)
        i_poly = i_it_poly * snake_config.ro + snake(init_input, adj).permute(
            0, 2, 1)
        return i_poly


    def forward(self, output, cnn_feature, batch=None,raining_state=False, xs=None,maskbackbone=None, all_feats=None,pixel_features=None,seg_embeddings=None):
        # Attentive
        b=cnn_feature.size(0)
        h,w=cnn_feature.size(2), cnn_feature.size(3)



        pe = positional_encoding_2d(dim, h, w)
        geo_feature = pe.unsqueeze(0).expand(b, dim, h, w).cuda()

        if batch is not None and raining_state:
            # output.update({'corner': corner_map})
            ct_01 = batch['ct_01'].byte()
            locations = dance_gcn_utils.collect_training(
                batch['init_box'], ct_01)  # 1/4 scale
            output.update({'init_box_1s':locations*snake_config.ro})

            targ_poly1 = dance_gcn_utils.collect_training(
                batch['targ_poly1'], ct_01)  # 1/4 scale
            whs = dance_gcn_utils.collect_training(batch['whs'], ct_01)



            keymasks= dance_gcn_utils.collect_training(batch['keymasks'], ct_01)
            output.update({'batched_whs': whs})
            ct_num = batch['meta']['ct_num']
            poly_ind = torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))],dim=0)


            pred_location_vertex_,pred_classification_vertex_,pred_classification_function_=self.corner_regression_model(xs, geo_feature,maskbackbone,locations, poly_ind,cnn_feature,targ_poly1,seg_embeddings)
            location_preds=[pred_location_vertex_,pred_classification_vertex_,pred_classification_function_]


            output.update({
                'py_pred': location_preds,
                'i_gt_py1': targ_poly1 * snake_config.ro,
                'keymasks':keymasks,

                # 'att_map':pred_edge

            })
        if not raining_state:
                init = self.prepare_testing_locations(output)
                locations = output['it_location']

                poly_ind = init['ind']

                pred_location_vertex_,pred_classification_vertex_,pred_classification_function_ = self.corner_regression_model(xs, geo_feature,
                                                                                                       maskbackbone,
                                                                                                       locations,
                                                                                                       poly_ind,cnn_feature,None,seg_embeddings)

                location_preds = [pred_location_vertex_,pred_classification_vertex_,pred_classification_function_]




                output.update({'py': location_preds})


        return output
