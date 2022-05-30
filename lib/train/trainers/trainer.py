import pdb
import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg
from lib.evaluators import make_evaluator
from lib.networks.renderer import make_renderer
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            if cfg.use_apex:
                from apex.parallel import DistributedDataParallel
                network = DistributedDataParallel(network,delay_allreduce=True)
            else:
                network = torch.nn.parallel.DistributedDataParallel(
                    network,
                    device_ids=[cfg.local_rank],
                    output_device=cfg.local_rank,
                    find_unused_parameters=True
                )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device
        self.step = 0

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):

        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                if isinstance(batch[k][0], list):
                    continue
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)

        return batch

    def train(self, epoch, data_loader, optimizer, recorder, max_iter):
        if cfg.use_prefetcher:
            batch = data_loader.next()
            iteration = 0
        self.network.train()
        end = time.time()
        # for iteration, batch in enumerate(data_loader):   # use dataloader
        while batch is not None:    # use prefetcher
            data_time = time.time() - end
            iteration = iteration + 1
            # batch type(dict):
            #   smpl_vertice : torch.Size([1, 6890, 3])         (list类型，受timesteps影响)未经变换成smpl坐标系下的所有smpl点坐标
            #   feature      : torch.Size([1, 6890, 6])         xyz和全0xyz的拼接
            #   coord        : torch.Size([1, 6890, 3])         经过5mm×5mm×5mm的voxel处理后得到的D' x H' x W'新坐标（变为了zyx）
            #   out_sh       : torch.Size([1, 3])               out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32) , out_sh = (out_sh | (32 - 1)) + 1  变为了zyx
            #   rgb          : torch.Size([1, 1024, 3])         1024条光线的rgb颜色
            #   ray_o        : torch.Size([1, 1024, 3])         1024条光线的起始点坐标（x,y,z）
            #   ray_d        : torch.Size([1, 1024, 3])         1024条光线的终止点坐标（x,y,z）
            #   near         : torch.Size([1, 1024])            最近距离
            #   far          : torch.Size([1, 1024])            最远距离
            #   acc          :  torch.Size([1, 1024])           对于图片的mask（512*512）,投射在身体和随机投射位置共1024处索引的在msk上的值
            #   mask_at_box  :  torch.Size([1, 1024])           和3d bounding box相交2次的射线,这里1024条光线都有
            #   bounds       :  torch.Size([1, 2, 3])           经过smpl坐标系变换后[min_xzy, max_xyz]
            #   R            :  torch.Size([1, 3, 3])           smpl参数计算得到旋转矩阵
            #   Th           :  torch.Size([1, 1, 3])           smpl参数平移变量
            #   center       :  torch.Size([1, 3])              中心点坐标
            #   rot          :  torch.Size([1, 2, 2])           随机旋转角
            #   trans        :  torch.Size([1, 3])              随机平移量
            #   i            :  torch.Size([1])                 当前index下对应图片是第i帧
            #   cam_ind      :  torch.Size([1])                 相机下标
            #   frame_index  :  torch.Size([1])                 当前index下对应图片是第frame_index帧,实际上与i一致
            #   human_idx    :  torch.Size([1])                 test中使用，表示human下标，在train中为0
            #   input_imgs   :  torch.Size([1, 3, 3, 512, 512]) (list类型，受timesteps影响)输入一帧3视角下的3张图片
            #   input_msks   :  torch.Size([1, 3, 1, 512, 512]) (list类型，受timesteps影响)输入一帧3视角下的3个mask
            #   input_vizmaps:  torch.Size([1, 3, 6890])        (list类型，受timesteps影响)输入一帧3视角下的smpl骨骼各顶点是否可见，存储bool类型
            #   input_uvmaps :  torch.Size([1, 3, 2, 512, 512]) (list类型，受timesteps影响)输入一帧3视角下的图片对应uvmap
            #   input_semantics :  torch.Size([1, 3, 512, 512]) (list类型，受timesteps影响)输入一帧3视角下的图片对应各部分semantic
            #   input_K      :  torch.Size([1, 3, 3, 3])        输入一帧3视角下的相机内参矩阵
            #   input_R      :  torch.Size([1, 3, 3, 3])        输入一帧3视角下的相机旋转矩阵
            #   input_T      :  torch.Size([1, 3, 3, 1])        输入一帧3视角下的相机平移矩阵
            #   target_K     :  torch.Size([1, 3, 3])           当前index下的相机内参矩阵
            #   target_R     :  torch.Size([1, 3, 3])           当前index下的相机R矩阵
            #   target_T     :  torch.Size([1, 3, 1])           当前index下的相机T矩阵

            '''for key in batch.keys():
                if key != 'smpl_vertice' and key != 'input_imgs' and key != 'input_msks' and key != 'input_vizmaps' :
                    print('\t%-13s: ' % (key), batch[key].shape)'''
                    # print('#\t%-13s: '%('smpl_vertice'),len(batch['smpl_vertice']))         # monitor the input shape

            batch = self.to_cuda(batch)

            output, loss, loss_stats, image_stats = self.network(batch)
            # training stage: loss; optimizer; scheduler
            optimizer.zero_grad()
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (
                            max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                training_state = '  '.join(
                    ['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string,
                                                       str(recorder), lr,
                                                       memory)
                print(training_state)

            if iteration % cfg.record_interval == 0 or iteration == (
                    max_iter - 1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')
            if cfg.use_prefetcher:
                batch = data_loader.next()

    def val(self, epoch, data_loader, renderer, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        self.step += 1
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                #output, loss, loss_stats, image_stats = self.network(batch)
                output = renderer.render(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch, self.step)
        evaluator.log(epoch)
        #     loss_stats = self.reduce_loss_stats(loss_stats)
        #     for k, v in loss_stats.items():
        #         val_loss_stats.setdefault(k, 0)
        #         val_loss_stats[k] += v
        #
        # loss_state = []
        # for k in val_loss_stats.keys():
        #     val_loss_stats[k] /= data_size
        #     loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        # print(loss_state)
        #
        # if evaluator is not None:
        #     result = evaluator.summarize()
        #     val_loss_stats.update(result)
        #
        # if recorder:
        #     recorder.record('val', epoch, val_loss_stats, image_stats)
