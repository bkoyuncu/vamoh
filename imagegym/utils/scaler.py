import numpy as np
from imagegym.config import cfg

class MinMaxScalerFixed:

    def __init__(self, feature_range):
        assert feature_range[0] == 0
        assert feature_range[1] == 1
        self.min_ = feature_range[0]
        self.max_ = feature_range[1]
        self.min_data = 0
        self.max_data = 1

    def fit(self, x):
        self.min_data = 0
        self.max_data = 1

    def transform(self, x):
        diff = self.max_data - self.min_data
        x_norm = (x - self.min_data) / diff  # [0,1]
        return x_norm

    def inverse_transform(self, x_norm):
        diff = self.max_data - self.min_data
        x = x_norm * diff + self.min_data
        return x

class MinMaxScaler:

    def __init__(self, feature_range):
        assert feature_range[0] == 0
        assert feature_range[1] == 1
        self.min_ = feature_range[0]
        self.max_ = feature_range[1]
        self.min_data = None
        self.max_data = None

    def fit(self, x):
        min_, max_ = x.min(), x.max()
        if self.min_data is None:
            self.min_data = x.min()
            self.max_data = x.max()
        else:
            self.min_data = min(self.min_data, min_)
            self.max_data = max(self.max_data, max_)
    
    def fit_manual(self):
        self.min_data = 0
        self.max_data = 1

    def transform(self, x):
        diff = self.max_data - self.min_data
        x_norm = (x - self.min_data) / diff  # [0,1]
        return x_norm

    def inverse_transform(self, x_norm):
        diff = self.max_data - self.min_data
        x = x_norm * diff + self.min_data
        return x

class StandardScalerFixed:

    def __init__(self):
        self.mu_ = None
        self.scale_ = None
        if cfg.dataset.name in ["celeba"]:
            #torchvision
            # self.mu_ = np.array([0.485, 0.456, 0.406])
            # self.scale_ = np.array([0.229, 0.224, 0.225])
            #Precomputed from dataset
            self.mu_ = 0.43173495
            self.scale_ = 0.2837438
        if cfg.dataset.name in ["era5"]:
            #this take them to [-1,1]
            # self.mu_ = 0.6352
            self.mu_ = 0.5
            # self.scale_ = 0.1825
            self.scale_ = 0.5
            # self.data.shape

    def fit(self, x):
        self.mu_ = x.mean()
        self.scale_ = x.std()

    def transform(self, x):
        return (x - self.mu_) / self.scale_

    def inverse_transform(self, x_norm):
        return x_norm * self.scale_ + self.mu_

class StandardScaler:

    def __init__(self):
        self.mu_ = None
        self.scale_ = None

    def fit(self, x):
        self.mu_ = x.mean()
        self.scale_ = x.std()

    def fit_with_loader(self, loader):
        mus, stds = [], []
        i = 0
        for batch in iter(loader):
            # print(f"i: {i} {batch[0].mean()}")
            i += 1
            mus.append(batch[0].mean())
            stds.append(batch[0].std())
            if i == 10: break

        self.mu_ = np.mean(mus)
        self.scale_ = np.mean(stds)

    def transform(self, x):
        return (x - self.mu_) / self.scale_

    def inverse_transform(self, x_norm):
        return x_norm * self.scale_ + self.mu_


class BaseScaler:

    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, x):
        self.scaler.fit(x)

    def fit_with_loader(self, loader):
        self.scaler.fit_with_loader(loader)

    def transform(self, x):
        x_uni = self.scaler.transform(x)
        return x_uni

    def inverse_transform(self, x_norm):
        x_uni = self.scaler.inverse_transform(x_norm)
        return x_uni

class MyStandardScaler(BaseScaler):
    def __init__(self):
        scaler = StandardScaler()
        super(MyStandardScaler, self).__init__(scaler=scaler)

class MyStandardScalerFixed(BaseScaler):
    def __init__(self):
        scaler = StandardScalerFixed()
        super(MyStandardScalerFixed, self).__init__(scaler=scaler)


class MyMinMaxScaler(BaseScaler):
    def __init__(self, feature_range):
        scaler = MinMaxScaler(feature_range=feature_range)
        super(MyMinMaxScaler, self).__init__(scaler=scaler)

class MyMinMaxScalerFixed(BaseScaler):
    def __init__(self, feature_range):
        scaler = MinMaxScalerFixed(feature_range=feature_range)
        super(MyMinMaxScalerFixed, self).__init__(scaler=scaler)
