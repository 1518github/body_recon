import time
import pdb
from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, \
    make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
from lib.networks.renderer import make_renderer
from line_profiler import LineProfiler
from torch.cuda.amp import GradScaler

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ['OMP_NUM_THREADS'] = str(1)

from queue import Queue
from threading import Thread


class CudaDataLoader:
    """ 异步预先将数据从CPU加载到GPU中 """

    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里 """
        # The loop that will load into the queue in the background
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class DataPrefetcher():
    def __init__(self, loader, cfg):
        self.loader = iter(loader)
        self.cfg = cfg
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch.keys():
                if isinstance(self.batch[k], list):
                    if isinstance(self.batch[k][0], list):
                        continue
                    for i in range(len(self.batch[k])):
                        self.batch[k][i] = self.batch[k][i].to(device=torch.device('cuda:{}'.format(cfg.local_rank)),
                                                         non_blocking=True)
                else:
                    self.batch[k] = self.batch[k].to(device=torch.device('cuda:{}'.format(cfg.local_rank)), non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def train(cfg, network):
    optimizer = make_optimizer(cfg, network)
    if cfg.use_apex:
        from apex.parallel import convert_syncbn_model
        from apex.parallel import DistributedDataParallel
        from apex import amp
        network = convert_syncbn_model(network)
        network, optimizer = amp.initialize(network, optimizer, opt_level='O1')
    trainer = make_trainer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    renderer = make_renderer(cfg, network)
    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    print('train_loader done!')
    val_loader = make_data_loader(cfg, is_train=False)
    print('val_loader done!')
    # for epoch in range(begin_epoch, cfg.train.epoch):
    # train_loader = CudaDataLoader(train_loader, device=torch.device('cuda:{}'.format(cfg.local_rank)))
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        if cfg.use_prefetcher:
            prefetch_time = time.time()
            train_prefetcher = DataPrefetcher(train_loader, cfg)
            # val_prefetcher = DataPrefetcher(val_loader, cfg)
            print('data prefetcher done in {}s'.format(time.time()-prefetch_time))
            trainer.train(epoch, train_prefetcher, optimizer, recorder, max_iter=len(train_loader))
        else:
            trainer.train(epoch, train_loader, optimizer, recorder, max_iter=len(train_loader))

        scheduler.step()

        if (epoch + 1) % cfg.save_freq == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        if cfg.eval_when_train and (epoch + 1) % cfg.eval_ep == 0:
            evaluator = make_evaluator(cfg)
            trainer.val(epoch, val_loader, renderer, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    # MPS_net now only works when timesteps = 3 !!!
    # torch.multiprocessing.set_start_method('spawn')
    if cfg.eval_when_train:
        print('eval ep: ',cfg.eval_ep)
    else:
        print('train without eval')
    print('time_step: %d , view_num: %d'%(cfg.time_steps,len(cfg.training_view)))

    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()
    network = make_network(cfg)
    if cfg.distributed:
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
