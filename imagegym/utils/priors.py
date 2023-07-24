import torch.nn as nn
import torch.distributions as td
import torch
import torch.nn.functional as F
from torch.distributions.distribution import Distribution
from imagegym.utils.scaler import MyStandardScaler, MyStandardScalerFixed, MyMinMaxScaler, MyMinMaxScalerFixed
import numpy as np
from imagegym.config import cfg
from imagegym.contrib.utils.random import get_permutation
from imagegym.utils.loader import get_weight
from torch.utils.data import DataLoader
from torch.distributions import constraints
import normflows as nf


class PriorDistribution(nn.Module):
    def __init__(self, name=None, dim_z=None, device=None):
        super().__init__()
        self.name = name
        self.dim_z = dim_z
        self.device=device
        
        if self.name == 'normal':
            self.prior_distr_z = torch.distributions.Normal(torch.zeros(self.dim_z).to(self.device),
                                                            (torch.ones(self.dim_z)*1).to(self.device))
            self.params_nf_fixed = True
        elif self.name == 'nf':
            base = nf.distributions.base.DiagGaussian(self.dim_z)
            num_layers = cfg.params_nf.L
            flows = []

            if "planar" in cfg.params_nf.type:
                for i in range(num_layers):
                    flows += [nf.flows.Planar((self.dim_z,),act = cfg.params_nf.act)]
            else:
                raise NotImplementedError
            
            # Construct flow model
            self.prior_distr_z = nf.NormalizingFlow(q0=base, flows=flows) #q0 is base dist, #flows list of flows
            self.prior_distr_z_normal = torch.distributions.Normal(torch.zeros(self.dim_z).to(self.device),
                                                            (torch.ones(self.dim_z)*1).to(self.device))
            self.params_nf_fixed = False
        else:   
            raise NotImplementedError
    # def __getattribute__(self, __name: str) -> Any:
        # pass
    def get_prior(self):
        
        if self.name == 'nf':
            if self.params_nf_fixed:
                return self.prior_distr_z_normal
            else:
                return self.prior_distr_z
        elif self.name == 'normal':
            return self.prior_distr_z


    def sample(self,n):
        if self.name == 'normal':
            z = self.prior_distr_z.sample(n[0])[0]
        # IF USING NF  
        elif self.name == "nf":
            if self.params_nf_fixed:
                z=self.prior_distr_z_normal.sample(n)
            else:
                z =self.prior_distr_z.sample(n[0])[0] 
        return z #num_sample, dim_z
    
    def sample_normal(self,n):
        # pz_normal = torch.distributions.Normal(torch.zeros(self.dim_z).to(self.device),
                                                            # (torch.ones(self.dim_z)*1).to(self.device))
        z=self.prior_distr_z_normal.sample(n)
        # pz_normal.sample(n)
        return z
    


