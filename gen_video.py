# import os
# import cv2
# import glob
# import pdb
# perform = 'data/perform'
# exp_name = 'new_mps1_mt1_timestep=3'
# epoch = -1
# exp_folder_name = 'epoch_50'
# path = os.path.join(perform,exp_name,'epoch_{}'.format(str(epoch)),exp_folder_name)
# view_list = []
# for index, (root ,dirs, files) in enumerate(os.walk(path)):
#     if index != 0:
#         view_list.append(root)
#
#
# for i, img_pth in enumerate(view_list):
#     list = []
#     for root ,dirs, files in os.walk(img_pth):
#         for file in files:
#             list.append(file[:-4])
#     list = sorted(list,key=int)
#     img = cv2.imread(img_pth+'/'+list[0]+'.png')
#     H, W = img.shape[0], img.shape[1]
#     video = cv2.VideoWriter(path+'/video_{}.avi'.format(i),cv2.VideoWriter_fourcc(*'MJPG'),60,(H,W))
#
#     for i in range(1,len(list)):
#         img = cv2.imread(img_pth+'/'+list[i-1]+'.png')
#         # img = cv2.resize(img,(1280,720))
#         video.write(img)
#
#     video.release()
from  moviepy.editor import *

clips = [VideoFileClip('data/perform/mps_timestep=3/epoch_-1/debug/video_0.avi'),VideoFileClip('data/perform/new_mps1_mt1_timestep=3/epoch_-1/debug/epoch_150.avi')]
video = clips_array([clips])
video.write_videofile(r'compare.mp4')
