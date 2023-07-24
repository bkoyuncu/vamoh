import torch
import time
import logging

from imagegym.config import cfg
from imagegym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from imagegym.checkpoint import load_ckpt, save_ckpt, clean_ckpt

from imagegym.register import register_train



def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for x in loader:
        optimizer.zero_grad()
        x.to(torch.device(cfg.device))
        loss_dict = model(x)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        # custom_stats = add_custom_stats_from_batch(batch)
        # logger.update_stats(true=true.detach().cpu(),
        #                     pred=pred_score.detach().cpu(),
        #                     loss=loss.item(),
        #                     lr=scheduler.get_last_lr()[0],
        #                     time_used=time.time() - time_start,
        #                     params=cfg.params,
        #                     **custom_stats)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for x in loader:
        x.to(torch.device(cfg.device))
        loss_dict = model(x)
        loss = loss_dict['loss']
        # custom_stats = add_custom_stats_from_batch(batch)
        # logger.update_stats(true=true.detach().cpu(),
        #                     pred=pred_score.detach().cpu(),
        #                     loss=loss.item(),
        #                     lr=0,
        #                     time_used=time.time() - time_start,
        #                     params=cfg.params,
        #                     **custom_stats)
        time_start = time.time()


def train_example(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('example', train_example)
