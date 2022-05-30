import torch
import torch.nn as nn
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import matplotlib.pyplot as plt
import numpy as np
import gc
import math
import time
import pdb

from lib.datasets.light_stage.densepose_result import DensePose

class Renderer:

    def __init__(self, net):
        self.net = net
        if cfg.use_densepose:
            self.densepose = DensePose()

    def paint_neural_human(self, batch, t, holder_feat_map, holder_feat_scale,
                           prev_weight=None, prev_holder=None):

        smpl_vertice = batch['smpl_vertice'][t]     # (1, 6890, 3) 6890个点的真实3d坐标，未经过smpl坐标变化

        if cfg.rasterize:       # ✔
            vizmap = batch['input_vizmaps'][t]      # (1, 3, 6890) 三个视角下6890个点是否可见
            result = vizmap[0]

        image_shape = batch['input_imgs'][t].shape[-2:] # (512, 512)

        input_R = batch['input_R']      # (1, 3, 3, 3)
        input_T = batch['input_T']      # (1, 3, 3, 1)
        input_K = batch['input_K']      # (1, 3, 3, 3)
        input_R = input_R.reshape(-1, 3, 3)     # (3,3,3)
        input_T = input_T.reshape(-1, 3, 1)     # (3,3,1)
        input_K = input_K.reshape(-1, 3, 3)     # (3,3,3)



        # smpl_vertice : (1, 6890, 3) 6890个点的真实3d坐标，未经过smpl坐标变化
        vertice_rot = \
        torch.matmul(input_R[:, None], smpl_vertice.unsqueeze(-1))[..., 0]      # (3, 6890, 3)
        vertice = vertice_rot + input_T[:, None, :3, 0]
        vertice = torch.matmul(input_K[:, None], vertice.unsqueeze(-1))[..., 0]     # (3, 6890, 3)
        # 带有3个相机视角下的点集的3d坐标
        uv = vertice[:, :, :2] / vertice[:, :, 2:]  # (3, 6890, 2)
        # 对holder_feat_map进行双线性插值
        latent = self.sample_from_feature_map(holder_feat_map,
                                              holder_feat_scale, image_shape,
                                              uv)
        # (3, 64, 6890)
        latent = latent.permute(0, 2, 1)        # (3, 6890, 64)

        num_input = latent.shape[0]             # 3
        if cfg.use_viz_test:

            final_result = result   # (3, 6890)
            big_holder = torch.zeros((latent.shape[0], latent.shape[1],
                                      cfg.embed_size)).cuda()
            big_holder[final_result == True, :] = latent[final_result == True,
                                                  :]
            # (3, 6890, 64) 可见性
            if cfg.weight == 'cross_transformer':
                return final_result, big_holder

        else:

            holder = latent.sum(0)
            holder = holder / num_input
            return holder

    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):

        scale = feat_scale / image_shape        # 256/255*2/512 = [0.00392,0.00392]
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=torch.cuda.current_device())

        uv = uv * scale - 1.0
        uv = uv.unsqueeze(2)        # (3, 6890, 1, 2)
        # 双线性插值
        samples = F.grid_sample(
            feat_map,       # (3, 64, 256, 256)
            uv,             # uv ∈ [-1, 1]
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        # samples.shape : (3, 64, 6890, 1)
        return samples[:, :, :, 0]

    def get_pixel_aligned_feature(self, batch, xyz, pixel_feat_map,
                                  pixel_feat_scale, batchify=False):
        # xyz: 光线 (1,1024,64,3)
        image_shape = batch['input_imgs'][0].shape[-2:]     # (512, 512)
        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)     # (3, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)     # (3, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)     # (3, 3, 3)


        if batchify == False:
            xyz = xyz.view(xyz.shape[0], -1, 3) # (1,65536,3)
        xyz = repeat_interleave(xyz, input_R.shape[0])
        # 把xyz (1,65536,3)重复3遍得到(3, 65536, 3)，第一维度3个是一样的
        #     output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
        #     return output.reshape(-1, *input.shape[1:])
        # xyz.shape : (3, 65536, 3)

        xyz_rot = torch.matmul(input_R[:, None], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + input_T[:, None, :3, 0]
        xyz = torch.matmul(input_K[:, None], xyz.unsqueeze(-1))[..., 0]
        # 计算uv坐标
        uv = xyz[:, :, :2] / xyz[:, :, 2:]      # (3, 1024*64, 2)
        # 从特征图采样
        pixel_feat = self.sample_from_feature_map(pixel_feat_map,
                                                  pixel_feat_scale, image_shape,
                                                  uv)
        # pixel_feat.shape: (3, 256, 65536)
        return pixel_feat

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)       # cfg.N_samples=64 从0到1切分64份
        # t_vals.shape: (64,)
        
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals  # z_vals.shape=torch.size([1,1024,64])
        # [znear,zfar]中选取64个点
        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            noise = torch.cuda.FloatTensor(z_vals.shape)
            t_rand = torch.rand(z_vals.shape,out=noise)
            # t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand
        
        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]
        # pts = r0 + zd z ∈ [znear, zfar]
        return pts, z_vals

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)       # 出错！！！
        pts = pts.view(*sh)
        return pts

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts

    def prepare_sp_input(self, batch):
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape     # [1,6890,6]
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])
        # coordinate: [N, 4], batch_idx, z, y, x
        # 第一列是遍历一个batch的下标
        sh = batch['coord'].shape       # [1,6890,3]
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]        # [0, 0, 0, 0, ... 0]  torch.size([6890,])
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1) # (6890, 4)
        # [0, coord]
        out_sh, _ = torch.max(batch['out_sh'], dim=0)       # shape: (1, 3)
        sp_input['out_sh'] = out_sh.tolist()
        
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']

        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        # 把光线的xyz坐标转变为zyx坐标
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]  # 将bounding box左下角点变为zyx形式
        dhw = dhw - min_dhw[:, None, None]  # 光线转变为bounding box原点的坐标形式
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        # 将坐标系原点变为bounding box中心，且把坐标形式改为[-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]   # 又把zyx坐标改为了xyz
        return grid_coords

    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir,
                      light_pts,
                      semantics,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      pixel_feat_map=None,
                      pixel_feat_scale=None,
                      norm_viewdir=None,
                      holder=None,
                      embed_xyz=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []

        for i in range(0, grid_coords.shape[1], chunk):

            xyz_shape = xyz.shape
            xyz = xyz.reshape(xyz_shape[0], -1, 3)

            pixel_feat = self.get_pixel_aligned_feature(batch,
                                                        xyz[:, i:i + chunk],
                                                        pixel_feat_map,
                                                        pixel_feat_scale,
                                                        batchify=True)

            ret = self.net(pixel_feat, sp_input,
                           grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk],
                           light_pts[:, i:i + chunk],
                           semantics,
                           holder=holder)

            all_ret.append(ret)

        all_ret = torch.cat(all_ret, 1)

        return all_ret


    def render(self, batch):

        ray_o = batch['ray_o']      # torch.Size([1, 1024, 3])
        ray_d = batch['ray_d']      # torch.Size([1, 1024, 3])
        near = batch['near']        # torch.Size([1, 1024])
        far = batch['far']          # torch.Size([1, 1024])
        sh = ray_o.shape    # torch.Size([1, 1024, 3])
        pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        # 在znear和zfar之间设置64个采样点用于近似积分计算
        # pts.shape = torch.Size([1, 1024, 64, 3])
        # z_vals.shape = torch.Size([1, 1024, 64])      [2.x ........ 3.x]从小到大的长度

        xyz = pts.clone()       # 光线

        # 将输入数量增大，提高效果
        light_pts = embedder.xyz_embedder(pts)          # torch.Size([1, 1024, 64, 63])     N=10

        # 将光线各点坐标转换为smpl坐标
        pts = self.pts_to_can_pts(pts, batch)

        ray_d0 = batch['ray_d'] # 各光线观测方向


        viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)      # torch.Size([1, 1024, 3])  [0.1x , 0.8x , -0.x]
        # 观测方向输入增加到27，提高效果
        viewdir = embedder.view_embedder(viewdir)       # torch.Size([1, 1024, 27])         N=4
        # 只是把维度增大了，值没有改变，为了保证shape一致
        viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()     # torch.Size([1, 1024, 64, 27])
        # 调整输入shape
        light_pts = light_pts.view(sh[0], -1,
                                   embedder.xyz_dim)        # torch.Size([1, 65536, 63])
        viewdir = viewdir.view(sh[0], -1,
                               embedder.view_dim)           # torch.Size([1, 65536, 27])

        sp_input = self.prepare_sp_input(batch)
        # {feature:(6890, 3), coord: (6890, 4), out_sh:(1, 3), batch_size, i}

        # convert xyz to the voxel coordinate dhw
        # 把光线pts坐标变为bounding box坐标系形式，且原点在bounding box中心，大小也缩小到了[-1,1]

        grid_coords = self.get_grid_coords(pts, sp_input,
                                           batch)   # (1, 1024, 64, 3)
        # 统一shape形式便于输入
        grid_coords = grid_coords.view(sh[0], -1, 3)    # (1, 65536, 3)

        ### --- get feature maps from encoder
        image_list = batch['input_imgs']        # torch.Size([1, 3, 3, 512, 512])
        if cfg.use_densepose:
            img_pth_list = batch['input_imgpaths']
        weight = None
        holder = None

        temporal_holders = []
        temporal_weights = []
        temporal_semantics = []

        for t in range(cfg.time_steps):         # 0

            images = image_list[t].reshape(-1, *image_list[t].shape[2:])    # (3,3,512,512)
            if cfg.use_densepose:
                img_pths = img_pth_list[t]
                semantics = []
                for img_pth in img_pths:
                    semantic_mask, uvmap_mask = self.densepose.get_densepose_result(img_pth[0])
                    import cv2
                    semantic_mask = torch.from_numpy(cv2.resize(np.array(semantic_mask.cpu()), (512, 512))).to(device=torch.device('cuda:{}'.format(cfg.local_rank)))
                    semantics.append(semantic_mask)
                semantics = torch.stack(semantics,dim=0).to(device=torch.device('cuda:{}'.format(cfg.local_rank)))
            if t == 0:
                # 图像特征提取
                holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(
                    images)

            else:
                holder_feat_map, holder_feat_scale, _, _ = self.net.encoder(
                    images)

            # holder_feat_map = torch.Size([3, 64, 256, 256])
            # pixel_feat_map  = torch.Size([3, 256, 256, 256])

            ### --- paint the holder
            # smpl
            weight, holder = self.paint_neural_human(batch, t,
                                                     holder_feat_map,
                                                     holder_feat_scale,
                                                     weight, holder)

            # weight: (3, 6890) 3个视角下6890个点的可见性
            # holder: (3, 6890, 64) 只显示可见的点
            if cfg.weight == 'cross_transformer':

                if cfg.cross_att_mode == 'cross_att':
                    temporal_holders.append(holder)
                    temporal_weights.append(weight)
            if cfg.use_densepose:
                temporal_semantics.append(semantics)
        if cfg.time_steps == 1:
            holder = temporal_holders[0]
            if cfg.use_densepose:
                semantics = temporal_semantics[0]
        else:
            holder = temporal_holders
            if cfg.use_densepose:
                semantics = torch.stack(temporal_semantics, dim=0).to(device=torch.device('cuda:{}'.format(cfg.local_rank)))  # (3,3,512,512)

        if ray_o.size(1) <= 2048:
            # 光线
            # xyz:未经过坐标变换的pts光线点：(1,1024,64,3)
            pixel_feat = self.get_pixel_aligned_feature(batch, xyz,
                                                        pixel_feat_map,
                                                        pixel_feat_scale)
            # shape: torch.Size([3, 256, 65536])
            if cfg.use_densepose:
                raw = self.net(pixel_feat, sp_input, grid_coords, viewdir,
                               light_pts, semantics, holder=holder)
            else:
                raw = self.net(pixel_feat, sp_input, grid_coords, viewdir,
                               light_pts, semantics=None, holder=holder)
            # torch.Size([1, 65536, 4])
        else:
            if cfg.use_densepose:
                raw = self.batchify_rays(sp_input, grid_coords, viewdir,
                                         light_pts,semantics,
                                         chunk=1024 * 32, net_c=None,
                                         batch=batch, xyz=xyz,
                                         pixel_feat_map=pixel_feat_map,
                                         pixel_feat_scale=pixel_feat_scale,
                                         holder=holder)
            else:
                raw = self.batchify_rays(sp_input, grid_coords, viewdir,
                                         light_pts,semantics=None,
                                         chunk=1024 * 32, net_c=None,
                                         batch=batch, xyz=xyz,
                                         pixel_feat_map=pixel_feat_map,
                                         pixel_feat_scale=pixel_feat_scale,
                                         holder=holder)


        raw = raw.reshape(-1, z_vals.size(2), 4)    # (1024, 64, 4) 64个采样点
        z_vals = z_vals.view(-1, z_vals.size(2))    # (1024, 64)
        ray_d = ray_d.view(-1, 3)                   # (1024, 3)



        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)


        rgb_map = rgb_map.view(*sh[:-1], -1)    # (1, 1024, 3)
        acc_map = acc_map.view(*sh[:-1])
        depth_map = depth_map.view(*sh[:-1])
        ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map}

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()

        return ret
