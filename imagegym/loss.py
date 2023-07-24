import torch
import torch.nn as nn
import torch.nn.functional as F

from imagegym.contrib.loss import *
import imagegym.register as register
from imagegym.config import cfg

import numpy as np

def ELBO_mixture_observed(qz_x, pz, px_z, x, beta, beta_c, qc, pc, K, mask):
    '''
    ELBO for mixture of 
    qz_x: q(z|x) [bs,dim_z]
    pz: p(z) [dim_z]
    px_z: p(x|z) [bs,num_pix_full,channels,K]
    x: ground truth [bs,num_pix_full,channels]
    beta: beta for KL loss
    beta_c: beta for KL loss of categorical
    qc: q(c|x) [bs,num_pix,K]
    pc: p(c) [bs,num_pix,K]
    K: number of mixture components
    mask: [bs,num_pix_full] observed pixels
    '''

    x_repeat = x.unsqueeze(-1).expand(-1, -1, -1, K) #better memory [bs,num_pix,ch,K]

    # loglikelihood of full data 
    if cfg.model.distr_x in ['ber'] and cfg.dataset.threshold == 0: #TODO check this one for mnist now/ because it is binarized
        log_prob_x = -torch.nn.BCEWithLogitsLoss(reduction="none")(px_z.logits,x_repeat) #sigmoid of logit gives the mean of dist
    elif cfg.model.distr_x in ['logistic']:
        log_prob_x = px_z.log_prob(x_repeat)
        # torch.log(px_z.cdf)
    else:
        log_prob_x = px_z.log_prob(x_repeat)

    # Marginalization
    #PIs of categorical distribution
    qc_probs = qc.probs.unsqueeze(-2) #[bs,num_pix,1,K], qc.probs has shape [bs,num_pix,K]
    #logp*pi for each pixel,ch,K
    log_prob_x_obs = log_prob_x[mask,:,:].reshape(x_repeat.shape[0],-1,x_repeat.shape[2],x_repeat.shape[3]) #[bs,#points(obs),ch,K]
    log_prob_x_all = (log_prob_x_obs*qc_probs) #[bs,#points(obs),ch,K]
    #sum over pixs,ch, K and mean over bs
    log_prob_x = log_prob_x_all.sum(dim=(1,2,3)).mean() #takes mean over the batch after sums over last 3 dims

    # KL loss (z)
    kl = compute_kl(pz,qz_x)

    # KL loss (c)
    kl_cat = torch.distributions.kl.kl_divergence(qc, pc).sum(-1) #[bs,pixels] and then summed over pixels
    kl_cat = kl_cat.mean()

    assert torch.any(torch.isinf(log_prob_x))==False, "log prob x has infinity"
    assert torch.any(torch.isinf(kl))==False, "KL has infinity"
    assert torch.any(torch.isinf(kl_cat))==False, "KL_cat has infinity"
    # assert torch.any(kl_cat>=0), "KL_cat has negative"

    
    elbo = log_prob_x - (beta) * kl - (beta_c) * kl_cat 

    my_dict = {
        'elbo': elbo,
        'log_prob_x': log_prob_x,
        'kl': kl,
        'kl_cat': kl_cat,
        'beta_c': torch.tensor(beta_c),
        'beta_z': torch.tensor(beta)
    }
    return my_dict

def compute_kl(prior, posterior):
    if prior.name == "normal":
        # KL loss (z)
        kl = torch.distributions.kl.kl_divergence(posterior, prior.get_prior()).sum(-1) #[bs,dim_z] and then summed over dim_z
        kl = kl.mean()
        return kl
    
    if prior.name == "nf":
        if prior.params_nf_fixed:
            kl = torch.distributions.kl.kl_divergence(posterior, prior.get_prior()).sum(-1) #[bs,dim_z] and then summed over dim_z
            kl = kl.mean()
            return kl
        else:
            z_sample =posterior.sample((2**12,)) #num_sample, bs, dim_z
            z_sample = z_sample.reshape(-1, z_sample.shape[-1]) #all, dim_z
            kl = prior.get_prior().forward_kld(z_sample) #INFO kl: [] it is a scalar, mean over all
            return kl