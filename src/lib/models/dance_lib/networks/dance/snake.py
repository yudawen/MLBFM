import torch.nn as nn
import torch


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim,
                            out_state_dim,
                            kernel_size=self.n_adj * 2 + 1)

    def forward(self, input, adj):
        input = torch.cat(
            [input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircularConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircularConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim,
                            out_state_dim,
                            kernel_size=self.n_adj * 2 + 1,
                            dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([
                input[..., -self.n_adj * self.dilation:], input,
                input[..., :self.n_adj * self.dilation]
            ],
                              dim=2)
        return self.fc(input)


class SnakeBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
        super(SnakeBlock, self).__init__()

        self.conv = DilatedCircularConv(state_dim, out_state_dim, n_adj,
                                        dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)

        return x

# class SnakeBlock_(nn.Module):
#     def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
#         super(SnakeBlock_, self).__init__()
#
#         self.conv = DilatedCircularConv(state_dim, out_state_dim, n_adj,
#                                         dilation)
#         self.relu = nn.ReLU(inplace=True)
#         self.norm = nn.BatchNorm1d(out_state_dim)
#
#     def forward(self, x):
#         x_in=x
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.norm(x)
#         x=x+x_in
#
#         return x
# _conv_factory = {'grid': CircConv, 'dgrid': DilatedCircConv}

# class BasicBlock(nn.Module):
#     def __init__(self,
#                  state_dim,
#                  out_state_dim,
#                  conv_type,
#                  n_adj=4,
#                  dilation=1):
#         super(BasicBlock, self).__init__()

#         self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj,
#                                              dilation)
#         self.relu = nn.ReLU(inplace=True)
#         self.norm = nn.BatchNorm1d(out_state_dim)

#     def forward(self, x, adj=None):
#         x = self.conv(x, adj)
#         x = self.relu(x)
#         x = self.norm(x)
#         return x

# class Snake(nn.Module):
#     def __init__(self, state_dim, feature_dim, conv_type='dgrid'):
#         super(Snake, self).__init__()

#         self.head = BasicBlock(feature_dim, state_dim, conv_type)

#         self.res_layer_num = 7
#         dilation = [1, 1, 1, 2, 2, 4, 4]
#         for i in range(self.res_layer_num):
#             conv = BasicBlock(state_dim,
#                               state_dim,
#                               conv_type,
#                               n_adj=4,
#                               dilation=dilation[i])
#             self.__setattr__('res' + str(i), conv)

#         fusion_state_dim = 256
#         self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1),
#                                 fusion_state_dim, 1)
#         self.prediction = nn.Sequential(
#             nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim,
#                       256, 1), nn.ReLU(inplace=True), nn.Conv1d(256, 64, 1),
#             nn.ReLU(inplace=True), nn.Conv1d(64, 2, 1))

#     def forward(self, x, adj):
#         states = []

#         x = self.head(x, adj)
#         states.append(x)
#         for i in range(self.res_layer_num):
#             x = self.__getattr__('res' + str(i))(x, adj) + x
#             states.append(x)

#         state = torch.cat(states, dim=1)
#         global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
#         global_state = global_state.expand(global_state.size(0),
#                                            global_state.size(1), state.size(2))
#         state = torch.cat([global_state, state], dim=1)
#         x = self.prediction(state)

#         return x


class SnakeNet_ori(nn.Module):
    def __init__(self, state_dim, edge_feature_dim,out_channel=2):
        super(SnakeNet_ori, self).__init__()

        self.head = SnakeBlock(edge_feature_dim, state_dim)

        self.res_layer_num = 7  # TODO: decrease
        # dilation = [1, 2, 4, 8, 12, 18, 24]
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = SnakeBlock(state_dim,
                              state_dim,
                              n_adj=4,
                              dilation=dilation[i])
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = 256

        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1),fusion_state_dim, 1)
        # print(state_dim)

        # self.self_attn = nn.MultiheadAttention(state_dim, 8, 0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.norm = nn.LayerNorm(state_dim)

        # self.self_attn2 = nn.MultiheadAttention(fusion_state_dim, 8)
        # self.dropout2 = nn.Dropout(0.1)
        # self.norm2 = nn.LayerNorm(fusion_state_dim)

        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim,256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_channel, 1))

    def forward(self, x):
        states = []

        x = self.head(x)
        # x_in=x

        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        back_out = self.fusion(state)
        # x_in=states[-1]
        # # print(back_out.size())
        # x = x_in.transpose(1, 2)
        # q = k = x
        # tgt = self.self_attn(q, k, x, key_padding_mask=None)[0]
        # tgt = x + self.dropout(tgt)
        # global_state2 = self.norm(tgt)
        # global_state2 = global_state2.transpose(1, 2)

        # print(back_out.size())

        global_state = torch.max(back_out, dim=2, keepdim=True)[0]

        global_state = global_state.expand(global_state.size(0),
                                           global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x
class SnakeNet(nn.Module):
    def __init__(self, state_dim, edge_feature_dim, out_channel=2):
        super(SnakeNet, self).__init__()
        # print(edge_feature_dim)

        self.self_attn = nn.MultiheadAttention(edge_feature_dim, 8, dropout=0.01)
        self.dropout = nn.Dropout(0.01)
        self.norm = nn.LayerNorm(edge_feature_dim)

        self.self_attn2 = nn.MultiheadAttention(edge_feature_dim, 8, dropout=0.01)
        self.dropout2 = nn.Dropout(0.01)
        self.norm2 = nn.LayerNorm(edge_feature_dim)

        self.self_attn3 = nn.MultiheadAttention(edge_feature_dim, 8, dropout=0.01)
        self.dropout3 = nn.Dropout(0.01)
        self.norm3 = nn.LayerNorm(edge_feature_dim)

        self.prediction = nn.Sequential(
            nn.Conv1d(edge_feature_dim,64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_channel, 1))

    def forward(self, x,pos=None):

        # self attention
        x=x.transpose(1, 2)
        q = k = x if pos==None else x+pos.transpose(1, 2)#self.with_pos_embed(tgt, query_pos)

        tgt = self.self_attn(q, k, x,key_padding_mask=None)[0]
        tgt = x + self.dropout(tgt)
        tgt = self.norm(tgt)

        q = k = x if pos==None else x+pos.transpose(1, 2)#self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, tgt,key_padding_mask=None)[0]
        tgt2 = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt2)

        q = k = x if pos==None else x+pos.transpose(1, 2)#self.with_pos_embed(tgt, query_pos)

        tgt3 = self.self_attn(q, k, tgt, key_padding_mask=None)[0]
        tgt3 = tgt + self.dropout(tgt3)
        tgt = self.norm(tgt3)

        tgt=tgt.permute(0,2,1)

        tgt_f=tgt
        cor = self.prediction(tgt)

        return cor#,tgt_f
# class SnakeNet_(nn.Module):
#     def __init__(self, state_dim, edge_feature_dim,out_channel=2):
#         super(SnakeNet_, self).__init__()
#
#         self.head = SnakeBlock(edge_feature_dim, state_dim, n_adj=0, dilation=1)
#
#         self.res_layer_num = 4  # TODO: decrease
#         dilation = [1, 1, 1, 7]
#         # n_adjs=[1,1,3,9]
#         n_adjs=[0,0,1,4]
#
#         for i in range(self.res_layer_num):
#             conv = SnakeBlock(state_dim,
#                               state_dim,
#                               n_adj=n_adjs[i],
#                               dilation=dilation[i])
#             self.__setattr__('res' + str(i), conv)
#
#         fusion_state_dim = 128
#
#         self.fusion = nn.Conv1d(state_dim * (self.res_layer_num),fusion_state_dim, 1)
#
#         self.prediction = nn.Sequential(
#             nn.Conv1d(fusion_state_dim, 64, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(64, out_channel, 1)
#         )
#             # nn.ReLU(inplace=True),
#             # nn.Conv1d(64, out_channel, 1))
#
#     def forward(self, x):
#         states = []
#
#         x = self.head(x)
#         # states.append(x)
#         for i in range(self.res_layer_num):
#             x = self.__getattr__('res' + str(i))(x) + x
#             states.append(x)
#
#         state = torch.cat(states, dim=1)
#
#         back_out = self.fusion(state)
#         # global_state = torch.max(back_out, dim=2, keepdim=True)[0]
#         #
#         # global_state = global_state.expand(global_state.size(0),
#         #                                    global_state.size(1), state.size(2))
#         # state = torch.cat([global_state, state], dim=1)
#         x = self.prediction(back_out)
#
#         return x


class SnakeNet_(nn.Module):
    def __init__(self, state_dim, edge_feature_dim,out_channel=2):
        super(SnakeNet_, self).__init__()

        self.head = SnakeBlock(edge_feature_dim, state_dim, n_adj=0, dilation=1)

        self.res_layer_num = 5  # TODO: decrease
        dilation = [1, 1, 3, 5, 7]
        # n_adjs=[1,1,3,9]
        n_adjs=[0, 1, 2, 3, 4]

        for i in range(self.res_layer_num):
            conv = SnakeBlock(state_dim,
                              state_dim,
                              n_adj=n_adjs[i],
                              dilation=dilation[i])
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = 256

        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num),fusion_state_dim, 1)

        self.prediction = nn.Sequential(
            nn.Conv1d(fusion_state_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_channel, 1)
        )
            # nn.ReLU(inplace=True),
            # nn.Conv1d(64, out_channel, 1))

    def forward(self, x):
        states = []

        x = self.head(x)
        # states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        back_out = self.fusion(state)
        # global_state = torch.max(back_out, dim=2, keepdim=True)[0]
        #
        # global_state = global_state.expand(global_state.size(0),
        #                                    global_state.size(1), state.size(2))
        # state = torch.cat([global_state, state], dim=1)
        x = self.prediction(back_out)

        return x