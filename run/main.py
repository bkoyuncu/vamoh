import random
import numpy as np
import torch
import logging
from imagegym.cmd_args import parse_args
from imagegym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir)

from imagegym.logger import setup_printing, create_logger
from imagegym.optimizer import create_optimizer, create_scheduler
from imagegym.model_builder import create_model, create_scenerios, create_adjust_beta
from imagegym.train import train
from imagegym.utils.device import auto_select_device
from imagegym.contrib.train import *

from imagegym.loader import create_dataset, create_loader

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print("cuda is available:",torch.cuda.is_available())
    print("cuda device count :",torch.cuda.device_count())
    # Load cmd line args
    args = parse_args()
    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        assert_cfg(cfg)
        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        out_dir_parent = cfg.out_dir
        cfg.seed = cfg.seed + i + args.seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        update_out_dir(out_dir_parent, args)
        dump_cfg(cfg)
        setup_printing()
        auto_select_device()
        print("Device:", cfg.device)

        # Set learning environment

        datasets = create_dataset()
        loaders = create_loader(datasets)
        meters = create_logger(datasets)
        model = create_model(datasets)
        scenerios = create_scenerios()
        adjust_beta_c, adjust_beta_z = create_adjust_beta()
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)

        dict_model_tuners = {
            "scenerios": scenerios,
            "adjust_beta_c": adjust_beta_c,
            "adjust_beta_z": adjust_beta_z,
        }

        
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = model.print_params_count(logging)
        logging.info('Num parameters: {}'.format(cfg.params))

        # Start training
        if cfg.train.mode == 'standard':
            train(meters, loaders, model, optimizer, scheduler, dict_model_tuners)
        else:
            raise NotImplementedError