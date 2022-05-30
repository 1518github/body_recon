import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                 SparseConvTranspose2d,
                                 SparseConvTranspose3d, SparseInverseConv2d,
                                 SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

from lib.config import cfg
from lib.networks.encoder import SpatialEncoder
import math
import time

class SpatialKeyValue(nn.Module):

    def __init__(self):
        super(SpatialKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(256, 256, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x))

class TemporalKeyValue(nn.Module):
    def __init__(self):
        super(TemporalKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(64, 32, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(64, 64, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x))

class MPS_net_atten(nn.Module):
    def __init__(self):
        super(MPS_net_atten, self).__init__()
        self.query_embed = nn.Conv1d(64, 32, kernel_size=1, stride=1)
        self.key_embed = nn.Conv1d(64, 32, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(64, 32, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.query_embed(x),
                self.key_embed(x),
                self.value_embed(x))
class MPS_net_HAFI(nn.Module):
    def __init__(self):
        super(MPS_net_HAFI, self).__init__()
        self.fc_0 = nn.Linear(64, 8)
        self.fc_1 = nn.Linear(24, 8)
        self.fc_2 = nn.Linear(8, 8)
        self.fc_3 = nn.Linear(8, 3)
        self.tanh = nn.Tanh()
    def forward(self, x):   # sctm_input_atten (6890*3, 64, 3)
        x0 = self.fc_0(x[..., 0])   # (6890*3, 64) -> (6890*3, 8)
        x1 = self.fc_0(x[..., 1])
        x2 = self.fc_0(x[..., 2])
        output = torch.cat((x0,x1,x2),dim=-1)    # (6890*3, 24)
        output = self.tanh(self.fc_1(output))         # (6890*3, 8)
        output = self.tanh(self.fc_2(output))         # (6890*3, 8)
        output = self.tanh(self.fc_3(output))         # (6890*3, 3)
        output = F.softmax(output,dim=-1).unsqueeze(1)  # (6890*3, 1, 3)
        output = torch.mul(x,output)    # (6890*3, 64, 3)*(6890*3, 1, 3) = (6890*3, 64, 3)
        output = torch.sum(output,dim=-1).unsqueeze(-1)   # (6890*3, 64, 1)
        return output   # (6890*3, 64)

def combine_interleaved(t, num_input=4, agg_type="average"):

    t = t.reshape(-1, num_input, *t.shape[1:])
    # torch.Size([1, 3, 256, 65536])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
        # torch.Size([1, 256, 65536])
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.temporal_view_layers = 1
        self.multi_view_layers = 1
        self.encoder = SpatialEncoder() # 图像特征提取

        if cfg.weight == 'cross_transformer':       # ✔
            self.cross_attention = nn.ModuleList([Multiview_Transformer() for _ in range(self.multi_view_layers)])
            # self.spatial_key_value_0 = SpatialKeyValue()
            # self.spatial_key_value_1 = SpatialKeyValue()
        if cfg.time_steps > 1:
            self.spatial_key_value_2 = TemporalKeyValue()
            self.spatial_key_value_3 = TemporalKeyValue()
        if cfg.use_mps_net:
            self.moca = nn.ModuleList([Temporal_Transformer() for _ in range(self.temporal_view_layers)])  # 2层self-attention
            # self.mps_net_atten = MPS_net_atten()
            # self.conv2d_0 = nn.Conv2d(2, 1, 1)
            # self.conv_wz = nn.Conv1d(32, 64, 1)
            self.mps_net_hafi = MPS_net_HAFI()
        self.xyzc_net = SparseConvNet()

        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(384, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)

        self.view_fc = nn.Conv1d(283, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

        self.fc_3 = nn.Conv1d(256, 256, 1)
        self.fc_4 = nn.Conv1d(128, 128, 1)

        self.alpha_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)

        self.rgb_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)
        self.rgb_res_1 = nn.Conv1d(cfg.img_feat_size, 128, 1)

        self.fc_5 = nn.Conv1d(64, 64, 1)
        self.fc_6 = nn.Conv1d(64, 64, 1)

    # def cross_attention(self, holder, pixel_feat):
    #     # 带参数网络，可学习
    #     key_embed, value_embed = self.spatial_key_value_0(
    #         pixel_feat.permute(2, 1, 0))    # (65536,256,3)
    #     # (65536,128,3)
    #     # (65536,256,3)
    #
    #     # 带参数网络，可学习
    #     query_key, query_value = self.spatial_key_value_1(
    #         holder.permute(2, 1, 0))
    #     # (65536,128,3)
    #     # (65536,256,3)
    #     k_emb = key_embed.size(1)
    #     A = torch.bmm(key_embed.transpose(1, 2), query_key)     # (65536,3,128),(65536,128,3)
    #     A = A / math.sqrt(k_emb)
    #     A = F.softmax(A, dim=1)
    #     # A=[65536, 3, 3]
    #     out = torch.bmm(value_embed, A) # (65536, 256, 3)
    #
    #     final_holder = query_value.permute(2, 1, 0) + out.permute(2, 1, 0)
    #     # (3, 256, 65536)
    #     return final_holder

    # lhh edit
    def self_attention(self,holder):    # len(holder) = timesteps , [shape:(3, 6890, 64)]
        sct = holder[0].permute(1, 0, 2)  # (L, C, d)=(6890, 3, 64)
        sctm = torch.stack(holder, dim=0).permute(2, 1, 3, 0)  # (L, C, d, M)=(6890, 3, 64, 3)
        sct_input_atten = sct.contiguous().view(-1, 1, sct.shape[2]).permute(0, 2, 1)  # (6890*3, 64, 1)
        sctm_input_atten = sctm.contiguous().view(sctm.shape[0] * sctm.shape[1], sctm.shape[2],
                                                  sctm.shape[3])  # (6890*3, 64, 3)
        # MPS-net目前只支持timesteps=3的情况，其他情况尚需修改

        if cfg.use_mps_net:
            for layer in self.moca:
                sctm_input_atten = layer(sctm_input_atten)  # (6890*3, 64, 3)
            # MOCA Module
            # nssm = torch.bmm(sctm_input_atten.transpose(1, 2),sctm_input_atten)  # (6890*3, 3, 3)
            # nssm = F.softmax(nssm,dim=1)
            #
            # query_embed, key_embed, value_embed = self.mps_net_atten(sctm_input_atten)
            # # (6890*3, 32, 3)
            # # (6890*3, 32, 3)
            # # (6890*3, 32, 3)
            # k_emb = key_embed.size(1)
            # atten_map = torch.bmm(query_embed.transpose(1, 2), key_embed)   # (6890*3, 3, 3)
            # atten_map = atten_map / math.sqrt(k_emb)
            # atten_map = F.softmax(atten_map, dim=1)
            #
            # moca_map = torch.stack((nssm,atten_map),dim=-1)     # (6890*3, 3, 3, 2)
            # moca_map = self.conv2d_0(moca_map.permute(0, 3, 1, 2))  # (6890*3, 1, 3, 3)
            # moca_map = moca_map.squeeze(1)
            # moca_map = F.softmax(moca_map, dim=1)   # # (6890*3, 3, 3)
            #
            # Y = torch.bmm(moca_map, value_embed.transpose(1, 2))    # (6890*3, 3, 32)
            # Y = self.conv_wz(Y.transpose(1,2))  # (6890*3, 64, 3)
            # sctm_input_atten = sctm_input_atten + Y     # (6890*3, 64, 3)

            # HAFI Module
            final_holder = self.mps_net_hafi(sctm_input_atten)    # (6890*3, 64, 1)
            final_holder = final_holder.squeeze(-1)                     # (6890*3, 64)
            final_holder = final_holder.view(sct.shape)                 # (6890, 3, 64)

            return final_holder.transpose(0, 1)    # (3, 6890, 64)
        else:
            query_key, _ = self.spatial_key_value_2(
                sct_input_atten) # (6890*3, 64, 1)
            # (6890*3, 32, 1)
            key_embed, value_embed = self.spatial_key_value_3(
                sctm_input_atten)   # (6890*3, 64, 3)
            # (6890*3, 32, 3)
            # (6890*3, 64, 3)
            k_emb = key_embed.size(1)
            A = torch.bmm(query_key.transpose(1, 2), key_embed)
            # (6890*3, 1, 3)

            A = A / math.sqrt(k_emb)
            A = F.softmax(A, dim=-1)

            out = torch.bmm(A,value_embed.permute(0, 2, 1))     # (20670, 1, 64)
            out = out.squeeze(1)        # (20670, 64)
            out = out.view(sct.shape)   # (6890, 3, 64)
            final_holder = sct + out    # (6890, 3, 64)

            return final_holder.transpose(0, 1)     # (3, 6890, 64)

    def forward(self, pixel_feat, sp_input, grid_coords, viewdir, light_pts,semantics,
                holder=None):
        # grid_coords: # (1, 1024*64, 3)
        # pixel_feat : (3, 256, 65536)
        # holder : (3, 6890, 64)
        # viewdir: (1, 1024*64, 27)
        # light_pts: (1, 1024*64, 63)
        # {feature:(6890, 3), coord: (6890, 4), out_sh:(1, 3), batch_size, i}
        feature = sp_input['feature']
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        p_features = grid_coords.transpose(1, 2)
        grid_coords = grid_coords[:, None, None]    # (1, 65536, 3, 1, 1)

        xyz = feature[..., :3]

        B = light_pts.shape[0]      # batch_size = 1
        n_input = int(pixel_feat.shape[0] / B)  # 3

        # self attention for temporal bank
        if cfg.time_steps > 1:
            holder = self.self_attention(holder)
        xyzc_features_list = []
        for view in range(n_input):
            xyzc = SparseConvTensor(holder[view], coord, out_sh, batch_size)
            xyzc_feature = self.xyzc_net(xyzc, grid_coords)
            xyzc_features_list.append(xyzc_feature)
            # (1, 384, 65536)
        # 特意将输出从（6890，64）调整到了（64+64+128+128 = 384，64*1024=65536）便于稍后输入transformer
        xyzc_features = torch.cat(xyzc_features_list,dim=0) # Sampled skeletal feature sctx
        # xyzc_features.shape : torch.Size([3, 384, 65536])
        net = self.actvn(self.fc_0(xyzc_features))  # fc_0: (384->256)  (3, 256, 65536)

        # transformer部分
        for layer in self.cross_attention:
            net = layer(net,self.actvn(self.alpha_res_0(pixel_feat)))    # (3, 256, 65536)
        # net = self.cross_attention(net,
        #                            self.actvn(self.alpha_res_0(pixel_feat)))    # (3, 256, 65536)
        net = self.actvn(self.fc_1(net))       # fc_1: (256->256)

        inter_net = self.actvn(self.fc_2(net))  # fc_2: (256->256)

        # NERF网络输入部分

        opa_net = combine_interleaved(
            inter_net, n_input, "average"       # (1, 256, 65536)   # 平均了n_input的值
        )
        opa_net = self.actvn(self.fc_3(opa_net))    # fc_3: (256->256)
        alpha = self.alpha_fc(opa_net)          # alpha_fc: (256 -> 1)
        # alpha.shape : (1, 1, 65536)       #

        features = self.feature_fc(inter_net)       # feature_fc: (256->256)
        features = features + self.rgb_res_0(pixel_feat)    # rgb_res_0: (256, 256)
        # features.shape : (3, 256, 65536)
        viewdir = repeat_interleave(viewdir,n_input)    # (3, 65536, 27)
        # 调整维度变为（27，65536）统一
        viewdir = viewdir.transpose(1, 2)   # (3, 27, 65536)

        features = torch.cat((features, viewdir), dim=1)    # (3, 256+27=283, 65536)
        net = self.actvn(self.view_fc(features))            # view_fc: (283->128)
        net = net + self.rgb_res_1(pixel_feat)              # rgb_res_1: (256->128)
        # num_input个输入作平均
        net = combine_interleaved(
            net, n_input, "average"
        )
        net = self.actvn(self.fc_4(net))                    # fc_4: (128->128)
        rgb = self.rgb_fc(net)                              # rgb_fc: (128->3)
        # rgb.shape: (1, 3, 65536)

        raw = torch.cat((rgb, alpha), dim=1)    # (1, 4, 65536)
        raw = raw.transpose(1, 2)               # (1, 65536, 4)
        return raw

class Temporal_Transformer(nn.Module):
    def __init__(self):
        super(Temporal_Transformer, self).__init__()
        self.mps_net_atten = MPS_net_atten()
        self.conv2d_0 = nn.Conv2d(2, 1, 1)
        self.conv_wz = nn.Conv1d(32, 64, 1)
    def forward(self,sctm_input_atten): # (6890*3, 64, 3)
        # MOCA Module
        nssm = torch.bmm(sctm_input_atten.transpose(1, 2), sctm_input_atten)  # (6890*3, 3, 3)
        nssm = F.softmax(nssm, dim=1)

        query_embed, key_embed, value_embed = self.mps_net_atten(sctm_input_atten)
        # (6890*3, 32, 3)
        # (6890*3, 32, 3)
        # (6890*3, 32, 3)
        k_emb = key_embed.size(1)
        atten_map = torch.bmm(query_embed.transpose(1, 2), key_embed)  # (6890*3, 3, 3)
        atten_map = atten_map / math.sqrt(k_emb)
        atten_map = F.softmax(atten_map, dim=1)

        moca_map = torch.stack((nssm, atten_map), dim=-1)  # (6890*3, 3, 3, 2)
        moca_map = self.conv2d_0(moca_map.permute(0, 3, 1, 2))  # (6890*3, 1, 3, 3)
        moca_map = moca_map.squeeze(1)
        moca_map = F.softmax(moca_map, dim=1)  # # (6890*3, 3, 3)

        Y = torch.bmm(moca_map, value_embed.transpose(1, 2))  # (6890*3, 3, 32)
        Y = self.conv_wz(Y.transpose(1, 2))  # (6890*3, 64, 3)
        sctm_input_atten = sctm_input_atten + Y  # (6890*3, 64, 3)
        return sctm_input_atten

class Multiview_Transformer(nn.Module):
    def __init__(self):
        super(Multiview_Transformer, self).__init__()
        self.spatial_key_value_0 = SpatialKeyValue()
        self.spatial_key_value_1 = SpatialKeyValue()
    def forward(self,holder, pixel_feat):
        # 带参数网络，可学习
        key_embed, value_embed = self.spatial_key_value_0(
            pixel_feat.permute(2, 1, 0))    # (65536,256,3)
        # (65536,128,3)
        # (65536,256,3)

        # 带参数网络，可学习
        query_key, query_value = self.spatial_key_value_1(
            holder.permute(2, 1, 0))
        # (65536,128,3)
        # (65536,256,3)
        k_emb = key_embed.size(1)
        A = torch.bmm(key_embed.transpose(1, 2), query_key)     # (65536,3,128),(65536,128,3)
        A = A / math.sqrt(k_emb)
        A = F.softmax(A, dim=1)
        # A=[65536, 3, 3]
        out = torch.bmm(value_embed, A) # (65536, 256, 3)

        final_holder = query_value.permute(2, 1, 0) + out.permute(2, 1, 0)
        # (3, 256, 65536)
        return final_holder

class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(64, 64, 'subm0')
        self.down0 = stride_conv(64, 64, 'down0')

        self.conv1 = double_conv(64, 64, 'subm1')
        self.down1 = stride_conv(64, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x, grid_coords):

        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()
        feature_4 = F.grid_sample(net4,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        '''

        '''

        features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                             dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     3,
                     2,
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
