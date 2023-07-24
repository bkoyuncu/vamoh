import torch
from imagegym.config import cfg
from imagegym.models.vamoh import VAMOH
from imagegym.utils.scenerios import Scenerios, adjust_beta
from imagegym.contrib.network import *
import imagegym.register as register

network_dict = {
    'vamoh': VAMOH,
}
network_dict = {**register.network_dict, **network_dict}

def create_model(datasets=None,
                 to_device=True,
                 dim_in=None,
                 dim_out=None):

    model_kwargs = create_model_kwargs()
    model = network_dict[cfg.model.type](**model_kwargs)

    if to_device:
        model.to(torch.device(cfg.device))

    model.create_meta_model()
    print(f"fparams is cuda {model.fparams[0].is_cuda}")
    model.set_input_scaler(datasets[0])

    return model

def create_scenerios():

    scenerios = Scenerios(cfg.model.scenerio,cfg.model.scenerio_start,cfg.model.scenerio_end)

    return scenerios

def create_adjust_beta():

    adjust_beta_c = adjust_beta([cfg.model.beta_c_scheduler, cfg.model.start_scheduler, cfg.model.end_scheduler, 1, cfg.model.beta_c])

    adjust_beta_z = adjust_beta([cfg.model.beta_z_scheduler, cfg.model.start_scheduler, cfg.model.end_scheduler, 1, cfg.model.beta_z])

    return adjust_beta_c, adjust_beta_z
    

def create_model_kwargs():
    kwargs ={}
    if cfg.model.type in ['vamoh']:
        #device 
        kwargs['device'] = cfg.device
        #task
        kwargs['task'] = cfg.dataset.task

        #data
        kwargs['dims_x'] = cfg.dataset.dims #[ch, h, w]
        kwargs['coordinate_dim'] = cfg.dataset.coordinate_dim
        kwargs['feature_dim'] = cfg.dataset.dims[0]
        #model
        kwargs['distr_x'] = cfg.model.distr_x
        kwargs['name_encoding'] = cfg.model.name_encoding
        kwargs['params_encoding'] = cfg.params_encoding
        kwargs['params_hyper'] = cfg.params_hyper
        kwargs['params_fnrep'] = cfg.params_fnrep
        kwargs['params_pointconvnet'] = cfg.params_pointconvnet
        kwargs['dim_z'] = cfg.model.dim_z
        kwargs['distr_z'] = cfg.model.distr_z
        kwargs['drop_input'] = cfg.model.drop_input
        kwargs['encoder_type'] = cfg.model.encoder_type
        kwargs['params_convnet'] = None
        kwargs['params_cat_prior'] = cfg.params_cat_prior
        kwargs['params_cat_post'] = cfg.params_cat_post
        kwargs['params_cat_x'] = cfg.params_cat_x
        kwargs['K'] = cfg.params_k_mixture.K

        #loss function
        kwargs['beta_z'] = cfg.model.beta_z
        kwargs['beta_c'] = cfg.model.beta_c
        kwargs['beta_c_scheduler'] = cfg.model.beta_c_scheduler

        #optimizer
        kwargs['loss_fun'] = cfg.model.loss_fun

        #training
        kwargs['two_step_training'] = cfg.model.two_step_training 
        kwargs['first_step_ratio'] = cfg.model.first_step_ratio
        kwargs['learn_residual_posterior'] =cfg.model.learn_residual_posterior
        kwargs['post_cat_has_z'] = cfg.model.post_cat_has_z
        kwargs['fix_categorical_prior'] = cfg.model.fix_categorical_prior
        kwargs['learn_residual_posterior'] = cfg.model.learn_residual_posterior
        kwargs['distr_x_logscales'] = cfg.model.distr_x_logscales

    return kwargs
