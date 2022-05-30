import os
import sys
import pdb
import numpy as np
import torch
sys.path.append("../")
from smplmodel.body_model import SMPLlayer

smpl_dir = '/data/NeuralHumanPerformer/Luohaohao/Neural_Human_Performer-main/data/zju_mocap/CoreView_392/new_params'
verts_dir = '/data/NeuralHumanPerformer/Luohaohao/Neural_Human_Performer-main/data/zju_mocap/CoreView_392/new_vertices'

# Previously, EasyMocap estimated SMPL parameters without pose blend shapes.
# The newly fitted SMPL parameters consider pose blend shapes.
new_params = False
if 'new' in os.path.basename(smpl_dir):
    new_params = True

smpl_path = os.path.join(smpl_dir, "0.npy")
verts_path = os.path.join(verts_dir, "0.npy")

## load precomputed vertices
verts_load = np.load(verts_path)
## create smpl model
model_folder = '/data/NeuralHumanPerformer/Luohaohao/Neural_Human_Performer-main/data/smplx'
device = torch.device('cpu')
body_model = SMPLlayer(os.path.join(model_folder, 'smpl'),
                       gender='neutral',
                       device=device,
                       regressor_path=os.path.join(model_folder,
                                                   'J_regressor_body25.npy'))
body_model.to(device)

## load SMPL zju
params = np.load(smpl_path, allow_pickle=True).item()
# poses  :  (1, 72)
# Rh  :  (1, 3)
# Th  :  (1, 3)
# shapes  :  (1, 10)
vertices = body_model(return_verts=True,
                      return_tensor=False,
                      new_params=new_params,
                      **params)
print(vertices)
print(verts_load)