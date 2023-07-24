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


class LogisticCustom(Distribution):
    """
        Logistic Distribution
    """
    arg_constraints = {"logits": constraints.real_vector}

    def __init__(self, logits = None, scales = None):
        if logits is None:
            raise ValueError("logits` must be specified")

        if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
        self.logits = logits
        self.scales = scales
        self._param = self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        super(LogisticCustom, self).__init__(batch_shape, validate_args=None)
        self.binsize = 1 / 512. #we use binsize to go one left one right
    
    @property
    def mean(self):
        return torch.sigmoid(self.logits)
    
    def sample(self):
        mean = torch.sigmoid(self.logits)
        # mean = self.logits
        sample = torch.round(mean*255)/255
        return sample
    
    def log_prob(self, sample):
        mean = torch.sigmoid(self.logits)
        scale = torch.clamp(torch.sigmoid(self.scales),min=None,max=1e-1)
        sample_r = ((sample + self.binsize) - mean) / scale
        sample_l = ((sample - self.binsize) - mean) / scale
        logp = torch.log(torch.sigmoid(sample_r) - torch.sigmoid(sample_l) + 1e-7 ) #cdf plus - cdf minus
        return logp



class BaseLikelihood(nn.Module):
    def __init__(self, domain_size):
        if isinstance(domain_size, list):
            domain_size = int(np.prod(domain_size))
        self._domain_size = domain_size
        super(BaseLikelihood, self).__init__()

    @property
    def domain_size(self):
        return self._domain_size
    @property
    def params_size(self):
        raise NotImplementedError()


    def get_scaler(self, dataset_tr):
        raise NotImplementedError()

    def fit_scaler(self, scaler, x):
        scaler.fit(x)

    def fit_scaler_with_loader(self, scaler, loader):
        print('Fitting scaler with loader')
        scaler.fit_with_loader(loader)


class BetaLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    @property
    def params_size(self):
        return self._domain_size * 2

    def forward(self, logits, return_mean=False,dim=1):
        logits = F.softplus(logits)
        latent_dim = logits.size(dim) // 2
        c0, c1 = torch.split(logits, split_size_or_sections=latent_dim, dim=dim)
        p = td.Beta(c0, c1)
        if return_mean:
            return p.mean, p
        else:
            return p

    def get_scaler(self, dataset_tr):
        bs = 64
        loader = DataLoader(dataset_tr,
                            batch_size=bs, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=False)
        scaler = MyMinMaxScaler(feature_range=(0, 1))
        for batch in iter(loader):
            x = batch[0]
            self.fit_scaler(scaler, x)
        return scaler



class BernoulliLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        p = td.Bernoulli(logits=logits)
        if return_mean:
            return p.mean, p
        else:
            return p


    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ["MNIST","PolyMNIST","celeba","shapenet","voxels"]:
            scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        else:
            bs = 64
            loader = DataLoader(dataset_tr,
                                batch_size=bs, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=False)
            scaler = MyMinMaxScaler(feature_range=(0, 1))
            for batch in iter(loader):
                x = batch[0]
                self.fit_scaler(scaler, x)
        return scaler



class CategoricalLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        p = td.Categorical(logits=logits)
        if return_mean:
            return F.softmax(logits,dim=dim), p
        else:
            return p

    def get_scaler(self, dataset_tr):
        raise NotImplementedError


class ContinousBernoulliLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        p = td.ContinuousBernoulli(logits=logits)
        if return_mean:
            return p.mean, p
        else:
            return p

    def get_scaler(self, dataset_tr):
        bs = 64
        loader = DataLoader(dataset_tr,
                            batch_size=bs, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=False)
        if cfg.dataset.name in ["MNIST","PolyMNIST","celeba","era5"]:
            scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        else:
            scaler = MyMinMaxScaler(feature_range=(0, 1))
        return scaler


class NormalLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    @property
    def params_size(self):
        return self._domain_size * 2

    def forward(self, logits, return_mean=False,dim=1):
        latent_dim = logits.size(dim) // 2
        mu, log_var = torch.split(logits, split_size_or_sections=latent_dim, dim=dim)
        log_var = torch.clamp(log_var, min=-70, max=70)
        std = torch.exp(log_var / 2)
        std = torch.clamp(std, min=-0.001, max=10)

        p = td.Normal(mu, std)
        if return_mean:
            return mu, p
        else:
            return p

    def get_scaler(self, dataset_tr):
        bs = 64
        loader = DataLoader(dataset_tr,
                            batch_size=bs, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=False)
        scaler = MyStandardScaler()
        self.fit_scaler_with_loader(scaler, loader)
        return scaler



class NormalMeanLikelihood(BaseLikelihood):
    def __init__(self, domain_size , variance):
        super().__init__(domain_size)
        self.variance = variance
        self.std  = np.sqrt(self.variance)



    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        # mu = torch.sigmoid(logits)
        mu = logits
        p = td.Normal(mu, self.std)
        if return_mean:
            return p.mean, p
        else:
            return p


    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ["celeba","era5"]:
            scaler = MyStandardScalerFixed()
            # scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        
        else:
            bs = 64
            loader = DataLoader(dataset_tr,
                                batch_size=bs, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=False)
            scaler = MyStandardScaler()
            self.fit_scaler_with_loader(scaler, loader)
        return scaler

class NormalMeanLikelihoodFixed(BaseLikelihood):
    std = None
    def __init__(self, domain_size):
        super().__init__(domain_size)

        self.std = NormalMeanLikelihoodFixed.std

    # a class method to create a Person object by birth year.
    @classmethod
    def create(cls, std):
        cls.std = std
        return cls

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        # mu = torch.sigmoid(logits)
        mu = logits
        p = td.Normal(mu, self.std)
        if return_mean:
            return p.mean, p
        else:
            return p


    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ["celeba"]:
            scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        return scaler
    
class NormalMeanLikelihoodTanh(BaseLikelihood):
    std = None
    def __init__(self, domain_size):
        super().__init__(domain_size)

        self.std = NormalMeanLikelihoodTanh.std

    # a class method to create a Person object by birth year.
    @classmethod
    def create(cls, std):
        cls.std = std
        return cls

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        mu = torch.tanh(logits)
        p = td.Normal(mu, self.std)
        if return_mean:
            return p.mean, p
        else:
            return p


    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ["celeba"]:
            scaler = MyStandardScalerFixed()
        else:
            bs = 64
            loader = DataLoader(dataset_tr,
                                batch_size=bs, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=False)
            scaler = MyStandardScaler()
            self.fit_scaler_with_loader(scaler, loader)
        return scaler


class NormalMeanLikelihoodSigmoid(BaseLikelihood):
    std = None
    def __init__(self, domain_size):
        super().__init__(domain_size)

        self.std = NormalMeanLikelihoodSigmoid.std

    # a class method to create a Person object by birth year.
    @classmethod
    def create(cls, std):
        cls.std = std
        return cls

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False, dim=1):
        mu = torch.sigmoid(logits)
        p = td.Normal(mu, self.std)
        if return_mean:
            return p.mean, p
        else:
            return p

    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ["MNIST","PolyMNIST","celeba"]:
            scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        else:
            scaler = MyMinMaxScaler(feature_range=(0, 1))
        return scaler

#create a torch dist
class Logistic(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)
        self.logits = None
        print("initializing our logistic in Logistic class")
    
    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits_scales, return_mean=True, dim=1):
        logits = logits_scales[0]
        scales = logits_scales[1]
        p = LogisticCustom(logits=logits,scales=scales)

        if return_mean:
            return p.mean, p
        else:
            return p

    def get_scaler(self, dataset_tr):
        if cfg.dataset.name in ["PolyMNIST","celeba","celebahq256","shapes3d","shapes3d_10","shapenet","shapes3d_50","MNIST"]:
            # scaler = MyStandardScalerFixed()
            scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        if cfg.dataset.name in ['era5']:
            scaler = MyMinMaxScalerFixed(feature_range=(0, 1))
        return scaler

likelihood_dict = {
    'beta': BetaLikelihood,
    'ber': BernoulliLikelihood,
    'cb': ContinousBernoulliLikelihood,
    'cat': CategoricalLikelihood,
    'normal': NormalLikelihood,
    'normal_mean': NormalMeanLikelihood,
    "logistic": Logistic
}

def set_likelihood(dist, in_channels):
    #INITIALIZE LL 
    if 'normal' in dist and len('normal') < len(dist):
        variance = float(dist.replace('normal',''))
        print(variance)
        likelihood_x = likelihood_dict['normal_mean'](in_channels, variance)
    elif "cb" in dist:
        likelihood_x = likelihood_dict['cb'](in_channels)
    elif "ber" in dist:
        likelihood_x = likelihood_dict['ber'](in_channels)
    elif dist == "logistic":
        print("initializing logistic LL")
        likelihood_x = likelihood_dict['logistic'](in_channels)
    elif dist == "logistic_nvae":
        likelihood_x = likelihood_dict['logistic_nvae'](in_channels)
    else:
        likelihood_x = likelihood_dict[dist](in_channels)

    return likelihood_x

