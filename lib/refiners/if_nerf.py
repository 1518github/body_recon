import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored
from lib.datasets.light_stage.densepose_result import DensePose

class Refiner:
    def __init__(self):
        self.densepose = DensePose()

        pred_root = 'data/refine/{}'.format(cfg.exp_name)
        os.system('mkdir -p {}'.format(pred_root))

        gt_root = 'data/refine/{}'.format(cfg.exp_name)
        os.system('mkdir -p {}'.format(gt_root))

        self.pred_root = os.path.join(pred_root, 'pred.png')
        self.gt_root = os.path.join(gt_root, 'gt.png')

    def refine(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        if cfg.white_bkgd:
            img_pred = np.ones((H, W, 3))
            img_gt = np.ones((H, W, 3))
        else:
            img_pred = np.zeros((H, W, 3))
            img_gt = np.zeros((H, W, 3))

        img_pred[mask_at_box] = rgb_pred
        img_gt[mask_at_box] = rgb_gt
        img_pred = img_pred[..., [2, 1, 0]]
        img_gt = img_gt[..., [2, 1, 0]]

        cv2.imwrite(self.pred_root, img_pred * 255)
        cv2.imwrite(self.gt_root, img_gt * 255)

        semantic_pred, uvmap_pred = self.densepose.get_densepose_result(self.pred_root)
        semantic_gt, uvmap_gt = self.densepose.get_densepose_result(self.gt_root)


