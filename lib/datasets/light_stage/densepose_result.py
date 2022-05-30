'''
    输入图片获取densepose得到的结果    edit by Luohaohao
'''

import argparse
import glob
import os
import pdb
import sys
from typing import Any, ClassVar, Dict, List
import torch
from termcolor import colored
import numpy as np
import cv2

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

def create_context():
    context = {"results": []}
    return context

def setup_config(
    config_fpath: str, model_fpath: str, opts: List[str]
):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    return cfg

class DensePose(object):
    def __init__(self):
        self.cfg = 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'
        self.model = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

        print(colored(f"Loading config from {self.cfg}", 'green'))
        self.opts = []
        self.cfg = setup_config(self.cfg, self.model, self.opts)
        print(colored(f"Loading model from {self.model}", 'green'))
        self.predictor = DefaultPredictor(self.cfg)
        self.input = ''


    def execute_on_outputs(
            self, context, entry, outputs
    ):
        image_fpath = entry["file_name"]
        # print(colored(f"Processing {image_fpath}", 'green'))
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    def _get_input_file_list(self, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    def execute(self):
        context = {"results": []}
        # print(colored(f"Loading data from {self.input}", 'green'))
        # file_list = self._get_input_file_list(self.input)
        # if len(file_list) == 0:
        #     print(colored(f"No input images for {self.input}", 'red'))
        #     return
        # for file_name in file_list:
        img = read_image(self.input, format="BGR")  # predictor expects BGR image.
        H, W = img.shape[0], img.shape[1]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            self.execute_on_outputs(context, {"file_name": self.input, "image": img}, outputs)
        # cls.postexecute(context)      # save the output result in pkl
        result = context["results"]
        bbox = result[0]['pred_boxes_XYXY'][0]
        input_uvmap = result[0]['pred_densepose'][0].uv   # tensor([2, H, W])
        input_semantic = result[0]['pred_densepose'][0].labels  # tensor([H, W])
        uvmap_mask = torch.zeros((2, H, W))
        uvmap_mask[:, int(bbox[1]):int(bbox[1]) + input_uvmap.shape[1],int(bbox[0]):int(bbox[0]) + input_uvmap.shape[2]] = input_uvmap
        semantic_mask = torch.zeros((H, W))
        semantic_mask[int(bbox[1]):int(bbox[1]) + input_semantic.shape[0],int(bbox[0]):int(bbox[0]) + input_semantic.shape[1]] = input_semantic
        return semantic_mask, uvmap_mask



    def get_densepose_result(self, input_image = 'examples/22097467bffc92d4a5c4246f7d4edb75.png'):
        self.input = input_image
        semantic_mask, uvmap_mask = self.execute()
        return semantic_mask, uvmap_mask

if __name__ == '__main__':
    dp = DensePose()
    img_list = glob.glob('examples/*.png')
    for index,img in enumerate(img_list):
        labels = dp.get_densepose_result(img)
        labels *= 10
        labels = labels.unsqueeze(-1)
        labels = labels.repeat(1, 1, 3)
        labels = np.array(labels.cpu())
        cv2.imwrite('output_{}.jpg'.format(str(index)), labels)