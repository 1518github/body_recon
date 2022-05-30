import torch
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.datasets import get_human_info
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import pdb
import random
import time

from lib.datasets.light_stage.densepose_result import DensePose

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()
        self.split = split  # 'train'
        self.im2tensor = self.image_to_tensor()
        if self.split == 'train' and cfg.jitter:
            self.jitter = self.color_jitter()
        #pdb.set_trace()
        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {}
        data_name = cfg.virt_data_root.split('/')[-1]    # virt_data_root = 'data/zju_mocap'
        # cfg.human: 313
        human_info = get_human_info.get_human_info(self.split)
        human_list = list(human_info.keys())  # ['CoreView_313', 'CoreView_315', 'CoreView_377', 'CoreView_386', 'CoreView_390', 'CoreView_392', 'CoreView_396']

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]

            data_root = os.path.join(cfg.virt_data_root, human)         # data/zju_mocap/CoreView_313

            ann_file = os.path.join(cfg.virt_data_root, human, 'annots.npy')    # data/zju_mocap/CoreView_313/annots.npy
            annots = np.load(ann_file, allow_pickle=True).item()   # type: dict , keys: [cams, ims]
            self.cams[human] = annots['cams']
            # annots['cams']:
            # type: dict , keys: [K, R, T, D]
            # K.size() : [num_cams, 3, 3]   相机内参矩阵
            # [[fx, 0 , u0]
            #  [0 , fy, v0]
            #  [0 , 0 , 1 ]]
            # R.size() : [num_cams, 3, 3]   旋转矩阵
            # T.size() : [num_cams, 3, 1]   平移矩阵
            # D.size() : [num_cams, 5, 1]   畸变矩阵
            num_cams = len(self.cams[human]['K'])
            if cfg.run_mode == 'train':
                test_view = [i for i in range(num_cams)]     # [0, 1, 2, 3, 4 ... num_cams-1]

            elif cfg.run_mode == 'test':
                if cfg.test_sample_cam:
                    if human in ['CoreView_313', 'CoreView_315']:
                        test_view = cfg.zju_313_315_sample_cam
                    else:
                        test_view = cfg.zju_sample_cam
                else:
                    test_view = [i for i in range(num_cams) if
                                 i not in cfg.test_input_view]

            if len(test_view) == 0:
                test_view = [0]


            i = 0
            i = i + human_info[human]['begin_i']
            i_intv = human_info[human]['i_intv'] # i提取间隔
            ni = human_info[human]['ni']
            # annots['ims']:
            # type:list size():(1470,)
            # content:[dict1, dict2, dict3, ...... ]
            # dict keys():[ims, kpts2d]
            #   ims: ['Camera (1)/CoreView_313_Camera_(1)_0001_2019-08-23_16-08-50.592.jpg', ... ]      size(): num_cams
            #   kpts2d: ppt 25   size(): [num_cams, 25, 3]

            # (i+ni-i)/i_intv
            ims = np.array([ # size(): [(i+ni-i)/i_intv, cam_nums]
                np.array(ims_data['ims'])[test_view]
                for ims_data in annots['ims'][i:i + ni][::i_intv]    # annots['ims'] size(): [1470, ]  提取M帧记忆帧
                ]).ravel()          # size(): [1260, ]
            # ims:
            # type: np.array   size():1260 = 21*60 = cam_nums * (i+ni-i)/i_intv
            # content: ['Camera (1)/CoreView_313_Camera_(1)_0001_2019-08-23_16-08-50.592.jpg', ... ]

            cam_inds = np.array([
                np.arange(len(ims_data['ims']))[test_view]
                for ims_data in annots['ims'][i:i + ni][::i_intv]
                ]).ravel()      #size(): [1260, ]
            # cam_inds:
            # type: np.array    size():1260 = 21*60 = cam_nums * (i+ni-i)/i_intv
            # content: 从0到cam_nums-1重复共(i+ni-i)/i_intv次，作为下标  [0, 1, 2, ... , 20, 0, 1, 2, ... 20, ... 20]

            start_idx = len(self.ims)   # 从上一个human_list之后开始读取
            length = len(ims)
            self.ims.extend(ims)        # 每处理一个human把图片添加进ims
            self.cam_inds.extend(cam_inds)

            if human in ['CoreView_313', 'CoreView_315']:

                self.ims[start_idx:start_idx + length] = [
                    data_root + '/' + x.split('/')[0] + '/' +
                    x.split('/')[1].split('_')[4] + '.jpg' for x in
                    self.ims[start_idx:start_idx + length]]
            else:
                self.ims[start_idx:start_idx + length] = [
                    data_root + '/' + x for x in
                    self.ims[start_idx:start_idx + length]]

            self.start_end[human] = {}
            self.start_end[human]['start'] = int(
                self.ims[start_idx].split('/')[-1][:-4])
            self.start_end[human]['end'] = int(
                self.ims[start_idx + length - 1].split('/')[-1][:-4])
            self.start_end[human]['length'] = self.start_end[human]['end'] - \
                                              self.start_end[human]['start']
            self.start_end[human]['intv'] = human_info[human]['i_intv']
            # {'CoreView_313': {'start': 1, 'end': 60, 'length': 59, 'intv': 1}, 'CoreView_315': {'start': 1, 'end': 397, 'length': 396, 'intv': 6} ... }

        self.nrays = cfg.N_rand      # rays num: 1024
        self.num_humans = len(human_list)  # 7

    def image_to_tensor(self):

        ops = []

        ops.extend(
            [transforms.ToTensor(), ]
        )

        return transforms.Compose(ops)

    def color_jitter(self):
        ops = []

        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )

        return transforms.Compose(ops)

    def get_mask(self, index):

        data_info = self.ims[index].split('/')      # ['data', 'zju_mocap', 'CoreView_313', 'Camera (6)', '0006.jpg']
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        msk_exist = False
        msk_cihp_exist = False

        msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                camera, frame)[:-4] + '.png'
        # msk_path: data/zju_mocap/CoreView_392/mask/Camera_B15/000186.png
        msk_exist = os.path.exists(msk_path)
        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)
        # msk.shape : [1024,1024]
        msk_path = os.path.join(cfg.virt_data_root, human, 'mask_cihp',
                                camera, frame)[:-4] + '.png'
        msk_cihp_exist = os.path.exists(msk_path)
        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        # msk_cihp.shape :  (1024, 1024)
        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)
        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)
        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        data_name = cfg.virt_data_root.split('/')[-1]   # zju_mocap

        border = 5      # 核大小

        kernel = np.ones((border, border), np.uint8)    # 5x5的全1矩阵作为腐蚀（膨胀）核
        msk_erode = cv2.erode(msk.copy(), kernel)       # 腐蚀
        msk_dilate = cv2.dilate(msk.copy(), kernel)     # 膨胀
        msk[(msk_dilate - msk_erode) == 1] = 100        # 腐蚀≠膨胀的地方变为100
        return msk

    def get_input_mask(self, human, index, filename):
        # index is camera index

        msk_exist = False
        msk_cihp_exist = False

        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                    'Camera (' + str(index) + ')',
                                    filename[:-4] + '.png')
        else:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                    'Camera_B' + str(index),
                                    filename[:-4] + '.png')

        msk_exist = os.path.exists(msk_path)

        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask_cihp',
                                    'Camera (' + str(index) + ')',
                                    filename[:-4] + '.png')
        else:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask_cihp',
                                    'Camera_B' + str(index),
                                    filename[:-4] + '.png')
        msk_cihp_exist = os.path.exists(msk_path)

        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)

        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)

        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        return msk

    def get_smpl_vertice(self, human, frame):

        vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices',
                                     '{}.npy'.format(frame))
        # data/zju_mocap/CoreView_392/vertices/xxx.npy
        smpl_vertice = np.load(vertices_path).astype(np.float32)

        return smpl_vertice

    def prepare_input(self, human, i):

        vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices',
                                     '{}.npy'.format(i))
        # data/zju_mocap/CoreView_313/vertices/xxx.npy
        xyz = np.load(vertices_path).astype(np.float32)     # xyz.shape : (6890, 3)
        smpl_vertices = None
        if cfg.time_steps == 1:
            smpl_vertices = np.array(xyz)

        nxyz = np.zeros_like(xyz).astype(np.float32)        # 全0 xyz

        # 顶点采样边界
        # obtain the original bounds for point sampling
        data_name = cfg.virt_data_root.split('/')[-1]       # zju_mocap
        min_xyz = np.min(xyz, axis=0)
        # 找到6890个点中3个坐标里最小的 shape: [1, 3]
        max_xyz = np.max(xyz, axis=0)
        # 找到6890个点中3个坐标里最大的 shape: [1, 3]
        if cfg.big_box:             # cfg.big_box = False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        # bounding box的六角点（未经过smpl坐标变换）
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)
        # can_bounds.shape : [2, 3]  type() : np.array

        # transform smpl from the world coordinate to the smpl coordinate
        # 世界坐标系转变为smpl坐标系
        params_path = os.path.join(cfg.virt_data_root, human, cfg.params,
                                   '{}.npy'.format(i))
        # cfg.params = params_4views_5e-4
        params = np.load(params_path, allow_pickle=True).item()
        # type() : dict     keys(): [Rh, Th, poses, shapes]
        # Rh.shape : (1, 3)
        # Th.shape : (1, 3)
        # poses.shape : (1, 72)
        # shapes.shape : (1, 10)

        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)     # 罗德里格斯公式：将旋转向量转变为旋转矩阵
        # R.shape : (3, 3)
        Th = params['Th'].astype(np.float32)


        xyz = np.dot(xyz - Th, R)       # 点乘->转变为smpl坐标

        # 变换增广,
        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)    # 增加随机旋转和平移，加噪

        # 坐标形式的边界
        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:  # False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        # bounding box的六角点（经过smpl坐标变换）
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)       #
        nxyz = nxyz.astype(np.float32)      # nxyz = xyz.zeros_like()
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        # feature.shape : (6890, 6)

        # 构建坐标系
        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]     # 将[0,1,2]转为[2,1,0]形式的坐标，即z,y,x
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)       # voxel_size: [0.005, 0.005, 0.005]
        # 将坐标系的原点变为min_dhw所在位置，把坐标改成这种形式
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)     # 计算坐标

        # bounding box尺寸
        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)     # shape:[1, 3]

        # np.ceil() 不小于该数的最小整数(向上取整)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1     # | : 按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1。
        # can_bounds: 坐标变换前的边界
        # bounds: 坐标变换后的边界
        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, smpl_vertices


    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):

        prob = np.random.randint(9000000)

        data_name = cfg.virt_data_root.split('/')[-1]   # zju_mocap
        img_path = self.ims[index]  # data/zju_mocap/CoreView_315/Camera (14)/0367.jpg
        data_info = img_path.split('/')
        human = data_info[-3]       # CoreView_313
        camera = data_info[-2]      # Camera (1)
        frame = data_info[-1]       # 000930.jpg
        img = imageio.imread(img_path)

        if self.split == 'train' and cfg.jitter:
            img = Image.fromarray(img)
            torch.manual_seed(prob)
            img = self.jitter(img)
            img = np.array(img)
        # type(img) : np.array
        img = img.astype(np.float32) / 255.

        msk = self.get_mask(index)      # msk.shape : (1024,1024)       type(msk) : imageio.core.util.Array
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams[human]['K'][cam_ind])       # K.shape : [3,3]
        D = np.array(self.cams[human]['D'][cam_ind])       # D.shape : [5,1]

        img = cv2.undistort(img, K, D)  # 用内参和畸变矩阵去图像畸变
        # img.shape : (1024, 1024)
        msk = cv2.undistort(msk, K, D)  # mask去畸变
        # msk.shape : (1024, 1024)
        R = np.array(self.cams[human]['R'][cam_ind])            # R.shape : [3,3]
        # when generating the annots.py, 1000 is multiplied, so dividing back
        T = np.array(self.cams[human]['T'][cam_ind]) / 1000.    # T.shape : [3,1]

        # 按比例cfg.ratio降低图像分辨率
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        # interpolation : 插值采样
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)     # 像素面积关系重采样 : https://zhuanlan.zhihu.com/p/38493205
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)  # 离得最近的像素值作为结果
        # img.shape : (512, 512, 3)
        # msk.shape : (512, 512)
        if cfg.mask_bkgd:
            if cfg.white_bkgd:
                img[msk == 0] = 1
            else:
                img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        target_K = K.copy()         # 目标K矩阵
        target_R = R.copy()
        target_T = T.copy()

        num_inputs = len(cfg['test_input_view'])        # test_input_view: [0,7,15]

        ### process input images
        if cfg.run_mode == 'train':

            if human in ['CoreView_313', 'CoreView_315']:
                input_view = [i for i in range(len(self.cams[human]['K']))]
                cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 21, 22]
            else:
                input_view = [i for i in range(len(self.cams[human]['K']))]
            # input_view : [0, 1, 2, ... num_cams-1]
            random.shuffle(input_view)  # shuffle
            input_view = input_view[:num_inputs]        # 从num_cams个相机视角中选择打乱好的前num_inputs个作为实际输入,num_inputs=3
        else:
            if human in ['CoreView_313', 'CoreView_315']:
                cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 21, 22]
            input_view = cfg.test_input_view            # 在test中则直接使用规定好的3个相机视角作为输入

        input_vizmaps = []
        input_imgs = []
        input_msks = []
        input_imgpaths = []
        input_K = []
        input_R = []
        input_T = []
        smpl_vertices = []

        if cfg.time_steps == 1:     # cfg.time_steps = 1
            time_mult = [0]
        elif cfg.time_steps > 1:
            if self.split == 'train':

                total_intv = int(
                    self.start_end[human]['length'] / self.start_end[human][
                        'intv'])

                raw_mult = np.array([-4, -3, -2, -1, 1, 2, 3, 4])


                random.shuffle(raw_mult)
                raw_mult = raw_mult[:cfg.time_steps - 1]
                if cfg.time_steps > 2:
                    raw_mult.sort()
                time_mult = [0]
                time_mult.extend(raw_mult)

            elif self.split == 'test':
                time_mult = cfg.time_mult

        target_frame = frame[:-4]       # 000825
        frame_index = int(target_frame) # 825
        zfill = len(target_frame)       # 6

        for t in range(cfg.time_steps):     # cfg.time_steps = 3
            start = self.start_end[human]['start']
            end = self.start_end[human]['end']
            intv = self.start_end[human]['intv']
            length = self.start_end[human]['length']
            # 采样得到M帧记忆帧从而增大每帧时间间隔
            if self.split == 'train':
                if t == 0:
                    current_frame = int(target_frame)
                else:
                    current_frame = ((int(target_frame) + time_mult[
                        t] * intv - start) % length) + start
                    # intv是间隔
            elif self.split == 'test':
                if t == 0:
                    current_frame = int(target_frame)
                else:
                    current_frame = ((int(target_frame) + time_mult[
                        t] - start) % length) + start


            filename = str(current_frame).zfill(zfill) + '.jpg'     # 获取当前帧图像


            if cfg.time_steps > 1:
                smpl_vertices.append(
                    self.get_smpl_vertice(human, current_frame))    # 加载smpl骨骼点集

            tmp_vizmaps = []
            tmp_imgs = []   # 存储一个step 3个view下的输入图像
            tmp_msks = []
            if cfg.use_densepose:
                tmp_imgpaths = []

            for i in range(num_inputs):
                idx = input_view[i]         # view下标
                cam_idx = None
                if human in ['CoreView_313', 'CoreView_315']:
                    cam_idx = cam_idx_list[idx]

                if human in ['CoreView_313', 'CoreView_315']:
                    input_img_path = os.path.join(cfg.virt_data_root, human,
                                                  'Camera (' + str(
                                                      cam_idx + 1) + ')',
                                                  filename)
                else:
                    input_img_path = os.path.join(cfg.virt_data_root, human,
                                                  'Camera_B' + str(idx + 1),
                                                  filename)
                    # 输入图像路径
                input_img = imageio.imread(input_img_path)

                if self.split == 'train' and cfg.jitter:
                    input_img = Image.fromarray(input_img)
                    torch.manual_seed(prob)
                    input_img = self.jitter(input_img)
                    input_img = np.array(input_img)

                input_img = input_img.astype(np.float32) / 255.

                if cfg.rasterize:       # 一帧下某个img对应smpl模型每个点是否可见
                    vizmap_idx = str(current_frame).zfill(zfill)
                    if human in ['CoreView_313', 'CoreView_315']:
                        vizmap_path = os.path.join(cfg.rasterize_root, human,
                                                   'visibility',
                                                   'Camera (' + str(
                                                       cam_idx + 1) + ')',
                                                   '{}.npy'.format(vizmap_idx))
                    else:
                        vizmap_path = os.path.join(cfg.rasterize_root, human,
                                                   'visibility',
                                                   'Camera_B' + str(idx + 1),
                                                   '{}.npy'.format(vizmap_idx))
                    # vizmap_path : data/zju_rasterization/CoreView_390/visibility/Camera_B7/000754.npy
                    input_vizmap = np.load(vizmap_path).astype(np.bool)
                    # 可见为True，不可见为False
                    # type() : np.array     shape: (6890,)      6890为smpl顶点总数

                if human in ['CoreView_313', 'CoreView_315']:
                    input_msk = self.get_input_mask(human, cam_idx + 1,
                                                    filename)
                else:
                    input_msk = self.get_input_mask(human, idx + 1, filename)   # 输入mask
                in_K = np.array(self.cams[human]['K'][idx]).astype(np.float32)  # 输入图片内参矩阵
                in_D = np.array(self.cams[human]['D'][idx]).astype(np.float32)

                input_img = cv2.undistort(input_img, in_K, in_D)
                # input_img.shape : (1024, 1024, 3)
                input_msk = cv2.undistort(input_msk, in_K, in_D)
                # input_msk.shape : (1024, 1024)
                in_R = np.array(self.cams[human]['R'][idx]).astype(np.float32)
                in_T = (np.array(self.cams[human]['T'][idx]) / 1000.).astype(
                    np.float32)

                input_img = cv2.resize(input_img, (W, H),
                                       interpolation=cv2.INTER_AREA)
                # input_img.shape : (512, 512, 3)
                input_msk = cv2.resize(input_msk, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
                # input_msk.shape : (512, 512)
                if cfg.mask_bkgd:
                    if cfg.white_bkgd:
                        input_img[input_msk == 0] = 1
                    else:
                        input_img[input_msk == 0] = 0

                input_msk = (
                            input_msk != 0)  # bool mask : foreground (True) background (False)

                if cfg.use_viz_test and cfg.use_fg_masking:     # use_viz_test=True , use_fg_masking=False
                    if cfg.ratio == 0.5:
                        border = 5


                    kernel = np.ones((border, border), np.uint8)

                    input_msk = cv2.erode(input_msk.astype(np.uint8) * 255,
                                          kernel)

                # numpy转tensor，且范围变为[-1, 1]
                input_img = self.im2tensor(input_img)
                input_msk = self.im2tensor(input_msk).bool()
                # input_img.shape : (3,512,512)
                # input_msk.shape : (1,512,512)

                in_K[:2] = in_K[:2] * cfg.ratio

                tmp_imgs.append(input_img)      # 添加此输入图像
                tmp_msks.append(input_msk)
                if cfg.use_densepose:
                    tmp_imgpaths.append(input_img_path)

                if cfg.rasterize:
                    tmp_vizmaps.append(torch.from_numpy(input_vizmap))

                if t == 0:
                    input_K.append(torch.from_numpy(in_K))      # 第0个timestep存储相机参数矩阵
                    input_R.append(torch.from_numpy(in_R))
                    input_T.append(torch.from_numpy(in_T))

            input_imgs.append(torch.stack(tmp_imgs))

            input_msks.append(torch.stack(tmp_msks))
            if cfg.use_densepose:
                input_imgpaths.append(tmp_imgpaths)
            if cfg.rasterize:
                input_vizmaps.append(torch.stack(tmp_vizmaps))
        input_K = torch.stack(input_K)
        input_R = torch.stack(input_R)
        input_T = torch.stack(input_T)

        i = int(frame[:-4])     # 该图像对应第i帧

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, tmp_smpl_vertices = self.prepare_input(
            human, i)
        if cfg.time_steps == 1:
            smpl_vertices.append(tmp_smpl_vertices)

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)
        acc = if_nerf_dutils.get_acc(coord_, msk)

        ret = {
            'smpl_vertice': smpl_vertices,
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'acc': acc,
            'mask_at_box': mask_at_box
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        i = int(os.path.basename(img_path)[:-4])    # basename(): 返回最后一个文件名

        human_idx = 0
        if self.split == 'test':
            human_idx = self.human_idx_name[human]
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
            'cam_ind': cam_ind,
            'frame_index': frame_index,
            'human_idx': human_idx,
            'input_imgs': input_imgs,
            'input_msks': input_msks,
            'input_vizmaps': input_vizmaps,
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'target_K': target_K,
            'target_R': target_R,
            'target_T': target_T

        }
        ret.update(meta)
        if cfg.use_densepose:
            ret.update({'input_imgpaths': input_imgpaths})
        return ret

    def get_length(self):
        return self.__len__()

    def __len__(self):
        return len(self.ims)
