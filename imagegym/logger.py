import logging
import os
import re
import sys
from sklearn.metrics import *
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

from imagegym.config import cfg
from imagegym.utils.device import get_current_gpu_usage
from imagegym.utils.io import dict_to_json, dict_to_tb
from imagegym.utils.plots_polar import (globe_plot_from_data_batch,
                                        globe_plot_from_data_batch2)
from imagegym.utils.plots_voxels import plot_voxels_batch_new


def setup_printing():
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    h_file = logging.FileHandler('{}/logging.log'.format(cfg.out_dir))
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)


class Logger(object):
    def __init__(self, name='train', task_type=None):
        assert task_type is not None
        self.name = name
        self.task_type = task_type

        self._epoch_total = cfg.optim.max_epoch
        self._time_total = 0  # won't be reset

        self.out_dir = os.path.join(cfg.out_dir, name)
        self.out_dir_img = os.path.join(self.out_dir, 'imgs')
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.out_dir_img, exist_ok=True)
        if cfg.tensorboard_each_run:
            self.tb_writer = SummaryWriter(self.out_dir)
        self.figure_types = cfg.plotting.figure_type
        self.reset()

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def reset(self):
        self._iter = 0
        self._size_current = 0 + 1e-10
        self._loss = 0
        self._lr = 0
        self._params = 0
        self._time_used = 0
        self._time_used_for_dl = 0
        self._custom_stats = {}

    # basic properties
    def basic(self):
        if self._size_current == 0:
            self._size_current = self._size_current + 1e-3
        if self._iter == 0:
            self._iter = self._iter + 1e-3
        stats = {'loss': round(self._loss / self._size_current, cfg.round) ,
                 'lr': round(self._lr, 6),
                #  'params': self._params,
                #  'time_iter': round(self.time_iter(), cfg.round),
                #  'time_iter_dl': round(self.time_iter_dl(), cfg.round),
                 }
        # gpu_memory = get_current_gpu_usage()
        # if gpu_memory > 0:
        #     stats['gpu_memory'] = gpu_memory
        # stats['gpu_memory'] = gpu_memory
        return stats

    # customized input properties
    def custom(self):
        if len(self._custom_stats) == 0:
            return {}
        out = {}
        for key, val in self._custom_stats.items():
            out[key] = val / self._size_current
        return out

    def _get_pred_int(self, pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > cfg.model.thresh).long()
        else:
            return pred_score.max(dim=1)[1]

    def generative(self):

        return {}


    def time_iter(self):
        return self._time_used / self._iter
    
    # time iter for data loading
    def time_iter_dl(self):
        return self._time_used_for_dl / self._iter

    def eta(self, epoch_current):
        epoch_current += 1  # since counter starts from 0
        time_per_epoch = self._time_total / epoch_current
        return time_per_epoch * (self._epoch_total - epoch_current)

    def update_stats(self, batch_size, loss, lr, time_used, time_used_for_dl, params, **kwargs):
        self._iter += 1
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        # self._params = params
        self._time_used += time_used #GPU
        self._time_used_for_dl += time_used_for_dl
        self._time_total += (time_used + time_used_for_dl) #total time
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                if key in ['recons','super']:
                    self._custom_stats[key] = val
                else:
                    self._custom_stats[key] = val * batch_size
            else:
                if key in ['recons','super']:
                    self._custom_stats[key] += val
                else:
                    self._custom_stats[key] += val * batch_size

    def write_iter(self):
        raise NotImplementedError

    def write_epoch(self, cur_epoch):
        basic_stats = self.basic()

        if self.task_type == 'generative':
            task_stats = self.generative()
        elif self.task_type in ['era5',"era5_polar","polar"]:
            task_stats = self.generative()
        elif self.task_type in ['chairs']:
            task_stats = self.generative()
        else:
            raise ValueError('Task not defined')

        epoch_stats = {'epoch': cur_epoch}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {**epoch_stats, **eta_stats, **basic_stats, **task_stats,
                     **custom_stats}
        else:
            stats = {**epoch_stats, **basic_stats, **task_stats, **custom_stats}

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()


    def save_image(self, x, name):
        grid = make_grid(x,nrow=8,normalize=False,scale_each=False)
        save_image(grid, os.path.join(self.out_dir_img, f'{name}{self.figure_types}'))


    def save_image_grid(self, x, name, nrow=8, normalize=False,scale_each=False):
        grid = make_grid(x,nrow=nrow,normalize=normalize,scale_each=scale_each)
        save_image(grid, os.path.join(self.out_dir_img, f'{name}{self.figure_types}'))

    
    def save_era(self,x, name, globe=False):

        fig_name = os.path.join(self.out_dir_img, f'{name}{self.figure_types}')        
        if globe==False:
            globe_plot_from_data_batch(x.cpu(), view=(100., 0.), globe=False,save_fig = fig_name)
        
        else:
            views = [(100., 0.)]# views = [(40,1),(160,1),(280,1)]
            for i in views:
                era_angle = '_' + re.sub(r'[^\w]', '', str(i))
                fig_name = os.path.join(self.out_dir_img, f'{name+era_angle}{self.figure_types}')
                globe_plot_from_data_batch2(x.cpu(), view=i, globe=True,save_fig = fig_name)

    def save_voxel(self,x, name):
        fig_name = os.path.join(self.out_dir_img, f'{name}{self.figure_types}')
        numpy_name = os.path.join(self.out_dir_img, f'{name}')
        plot_voxels_batch_new(voxels=x.cpu(),save_fig=fig_name)

    def close(self):
        if cfg.tensorboard_each_run:
            self.tb_writer.close()



def create_logger(datasets):
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(datasets):
        loggers.append(Logger(name=names[i], task_type=cfg.dataset.task_type))
    return loggers

def create_logger_sampling(datasets):
    loggers = []
    names = ['train_sample', 'val_sample', 'test_sample']
    for i, dataset in enumerate(datasets):
        loggers.append(Logger(name=names[i], task_type=cfg.dataset.task_type))
    return loggers
