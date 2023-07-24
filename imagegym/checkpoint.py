from imagegym.config import cfg

import torch
import os
import logging


def get_ckpt_dir():
    print(cfg.out_dir)
    return '{}/ckpt/'.format(cfg.out_dir)


def get_all_epoch():
    d = get_ckpt_dir()
    names = os.listdir(d) if os.path.exists(d) else []
    if len(names) == 0:
        return [0]
    epochs = [int(name.split('.')[0]) for name in names]
    return epochs


def get_last_epoch():
    return max(get_all_epoch())


def load_ckpt(model, optimizer=None, scheduler=None, ckpt_number = None):
    if cfg.train.epoch_resume < 0:
        epoch_resume = get_last_epoch()
    else:
        epoch_resume = cfg.train.epoch_resume

    if ckpt_number is not None:
        epoch_resume = ckpt_number
        
    ckpt_name = '{}{}.ckpt'.format(get_ckpt_dir(), epoch_resume)
    if not os.path.isfile(ckpt_name):
        return 0

    print(torch.cuda.is_available(),torch.cuda.device_count())

    print("ckpt name:",ckpt_name)
    if cfg.device == "cpu":
        ckpt = torch.load(ckpt_name, map_location=torch.device(cfg.device))
    else:
        ckpt = torch.load(ckpt_name, map_location=torch.device('cpu'))
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state'],strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    return epoch + 1


def save_ckpt(model, optimizer, scheduler, epoch):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    os.makedirs(get_ckpt_dir(), exist_ok=True)
    ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
    torch.save(ckpt, ckpt_name)
    logging.info('Check point saved: {}'.format(ckpt_name))


def clean_ckpt():
    epochs = get_all_epoch()
    epoch_last = max(epochs)
    for epoch in epochs:
        if epoch != epoch_last:
            ckpt_name = '{}/{}.ckpt'.format(get_ckpt_dir(), epoch)
            os.remove(ckpt_name)
