from abc import abstractmethod
from imagegym.utils.likelihoods import set_likelihood
from imagegym.utils.mask import *
import torch.nn as nn
from imagegym.config import cfg
from functorch import combine_state_for_ensemble
from imagegym.datasets.converter import GridDataConverter, ERA5Converter
from imagegym.models.layer.mlp import MLP_dropout, Identity_layer, process_cat_encoder_prior_params, process_cat_encoder_post_params, process_cat_encoder_x_params
from imagegym.models.layer.point_conv_encoder import PointConvEncoder, process_pointconvnet_params
from imagegym.models.layer.function_representation import FunctionRepresentation, FourierFeatures, IdentityFeatures, process_fnrep_params
from imagegym.models.layer.hypernet import HyperNetwork, process_hypernet_params
import torch

class vamoh_base(nn.Module):
    def __init__(self,
                task,
                feature_dim,
                coordinate_dim,
                dims_x,
                distr_x,
                name_encoding,
                params_encoding,
                params_hyper,
                params_fnrep,
                dim_z,
                distr_z,
                beta_z,
                beta_c,
                loss_fun,
                drop_input,
                params_convnet,
                params_cat_prior,
                params_cat_post,
                params_cat_x,
                params_pointconvnet,
                encoder_type,
                K,
                device,
                name="vamoh",
                **kwargs):
        super(vamoh_base, self).__init__()

        assert cfg.model.encoder_type in ["pointconv"]

        self.name = name
        self.task = task # image or pointcloud or sth
        self.dim_z = dim_z
        self.distr_z = distr_z
        self.encoder_type = encoder_type
        self.parallize_in_batch = True #TODO CHANGE THIS for chair
        self.dims_x = dims_x
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.distr_x = distr_x
        self.device = device
        self.beta_z = beta_z
        self.beta_c = beta_c

        if self.task == "image":
            self.data_converter = GridDataConverter(device=device,
                                            data_shape=self.dims_x,
                                            normalize=True,
                                            normalize_features=False)
            self.parallize_in_batch = True

        if self.task == "era5_polar":
            self.coordinate_dim = 3
            # dims_x = [1,46,90]
            self.data_converter = ERA5Converter(device=cfg.device,
                                                data_shape=self.dims_x[1:],
                                                normalize=False,
                                                normalize_features=False)
            self.parallize_in_batch = True

        if self.task == "voxels_chairs":
            self.coordinate_dim = 3
            dims_x =  [1, 32, 32, 32]
            self.data_converter = GridDataConverter(device=cfg.device,
                                                data_shape=dims_x,
                                                normalize=True,
                                                normalize_features=False)
            self.parallize_in_batch = True

        self.K = K
        params_pointconvnet = params_pointconvnet
        params_cat_prior = params_cat_prior
        params_cat_post = params_cat_post
        params_cat_x = params_cat_x
        params_encoding = params_encoding

        #SET LIKELIHOOD_X
        self.lik_x = set_likelihood(dist = distr_x, in_channels=self.dims_x[0])
        if distr_x == "logistic":
            self.lik_x_logscales = nn.Parameter(torch.ones((self.dims_x[0], self.K))*kwargs['distr_x_logscales'],requires_grad=False)  # inside the LL + 1e-10
        #SET PRIOR_Z (always normal1)
        self.prior_z = set_likelihood(dist = "normal1", in_channels=dim_z)
        #SET LIKELIHOOD_Z (normal)
        self.lik_z = set_likelihood(dist = "normal", in_channels=dim_z)

        #FOURIER ENCODING FOR COORDINATES
        if name_encoding:
            num_frequencies = params_encoding['num_frequencies']
            input_dim = self.coordinate_dim
            mean = torch.zeros(num_frequencies,
                               input_dim)

            frequency_matrix = torch.normal(mean=mean,
                                            std=params_encoding['std']).to(self.device)

            encoding = FourierFeatures(frequency_matrix,
                                       learnable_features=params_encoding['learn_feats'])
            self.coordinate_dim = 2*num_frequencies
        else:
            encoding = IdentityFeatures(coordinate_dim=self.coordinate_dim)
            self.coordinate_dim = self.coordinate_dim
            assert cfg.params_pointconvnet.use_encoded_coors == False, "Cannot use encoded coordinates with Identity Features change it to False"
        
        
        #FUNCTION REPRESENTATION (DECODER)
        process_fnrep_params(encoding=encoding,
                             feature_dim= self.lik_x.params_size,
                             fnrep_params=params_fnrep)
        
        self.decoder = FunctionRepresentation(**params_fnrep).to(device)
        self.models_Kbs = [self.decoder.mlp.to(device) for _ in range(self.K*cfg.train.batch_size)]

        #HYPERNETWORK (GENERATES WEIGHTS OF FN)
        process_hypernet_params(input_dim=dim_z,
                                fn_representation=self.decoder,
                                hyper_params=params_hyper)
        
        self.hyper_list = nn.ModuleList([HyperNetwork(**params_hyper).to(device) for i in range(self.K)])
        
        self.encoder_z, self.prior_cat_encoder, self.post_cat_encoder, self.cat_encoder_x = self.set_encoders(params_pointconvnet, params_cat_prior, params_cat_post, params_cat_x, self.coordinate_dim)
        self.initial_same_coordinates=self.encoder_z.same_coordinates
        #other flags
        self.create_encoded_coords=True
        self.fix_categorical_prior = kwargs['fix_categorical_prior']
        self.freeze_categorical_posterior = False
        self.learn_residual_posterior = kwargs['learn_residual_posterior']

    @abstractmethod
    def forward(self, batch, **kwargs):
        """
        :param batch: a dict of tensors
        :param missingness: give the missingness of the batch (somehow)
        :return: elbo
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, sample_size, **kwargs):
        """
        :param sample_size: number of samples to generate
        :res resolution of generated samples
        :z latent z to condition on (if available)
        """
        raise NotImplementedError
    
    @abstractmethod
    def reconstruct(self, coordinates_features, resolution, out_coordinates, **kwargs):
        """
        :param coordinates_features: coordinates and features to encode
        :res resolution of reconstructions
        :out_coordinates: coordinates of the output
        """
        raise NotImplementedError
    
    @abstractmethod
    def elbo(self, **kwargs):
        """
        :param coordinates_features: coordinates and features to encode
        :res resolution of reconstructions
        :out_coordinates: coordinates of the output
        returns elbo for a given batch
        """
        raise NotImplementedError

    @abstractmethod
    def set_z_prior_distr(self, **kwargs):
        """
        """
        raise NotImplementedError

    @abstractmethod
    def set_pi_prior_distr(self, **kwargs):
        """
        """
        raise NotImplementedError

    def set_input_scaler(self, dataset):
        self.input_scaler = self.lik_x.get_scaler(dataset)
        return

    def create_meta_model(self):
        '''
        Creates meta model for using decoder (mixture of K) with VMAP 
        '''
        self.fmodel, self.fparams, self.fbuffers = combine_state_for_ensemble(self.models_Kbs)
        self.fmodel.requires_grad_(False)

    def preprocess_batch(self,x, **kwargs):
        """
        :param batch: a dict of tensors
        :param missingness: give the missingness of the batch (somehow)
        :return: elbo
        """
        assert self.input_scaler is not None
        x_norm = self.input_scaler.transform(x)

        if cfg.dataset.use_bn_initial:
            h = self.bn_initial(x_norm)
        else:
            h = x_norm

        return x_norm, h
    def postprocess_batch(self, x):
        return self.input_scaler.inverse_transform(x)
    
    def set_encoders(self,params_pointconvnet, params_cat_prior, params_cat_post, params_cat_x, coordinate_dim):
        #ENCODER (Z)
        if self.encoder_type == "pointconv":
            #coordinate_dim handled by process_pointconvnet_params
            process_pointconvnet_params(coordinate_dim=None, feature_dim=self.feature_dim, pointconvnet_params=params_pointconvnet)
            encoder = PointConvEncoder(**params_pointconvnet).to(self.device)
        else:
            raise ValueError("Unknown encoder type")
        
        #ENCODER (CATEGORICAL)
        process_cat_encoder_prior_params(mlp_params=params_cat_prior,coordinate_dim=coordinate_dim)
        process_cat_encoder_post_params(mlp_params=params_cat_post,coordinate_dim=coordinate_dim)
        process_cat_encoder_x_params(mlp_params= params_cat_x,coordinate_dim=coordinate_dim)

        prior_cat_encoder = MLP_dropout(**params_cat_prior).to(self.device)
        post_cat_encoder = MLP_dropout(**params_cat_post).to(self.device)
        if cfg.model.cat_encoder_x ==True:
            cat_encoder_x = MLP_dropout(**params_cat_x).to(self.device)
        else:
            cat_encoder_x = Identity_layer(**params_cat_x).to(self.device)
        
        return encoder, prior_cat_encoder, post_cat_encoder, cat_encoder_x

    def create_encoded_coordinates_fwd(self, coordinates:torch.Tensor):
        bs = coordinates.shape[0]
        if self.parallize_in_batch == False:
            self.encoded_coords_fwd_one = [self.decoder.position_encoding(coord) for coord in coordinates] #this makes [size*size,2] -> [size*size,256]
            self.create_encoded_coords=True
            encoded_coords_fwd = torch.stack(self.encoded_coords_fwd_one, 0)

        else:
            if self.create_encoded_coords:
                self.encoded_coords_fwd_one = [self.decoder.position_encoding(coord) for coord in coordinates[[0]]] #this makes [size*size,2] -> [size*size,256]
                self.create_encoded_coords=False
            encoded_coords_fwd = torch.stack(self.encoded_coords_fwd_one, 0).expand(bs,-1,-1)

            # if self.task == "voxels_chairs":
            #     self.create_encoded_coords=True

        return encoded_coords_fwd

    def prior_cat(self, coordinates, z):
        """
            Parameters:
            coordinates: encoded_coordinates
            z: latent variable
        """        
        bs = coordinates.shape[0]         
        number_of_pixels = coordinates.shape[1]
        coordinates = self.cat_encoder_x(coordinates.reshape(bs*number_of_pixels,-1)).reshape(bs, number_of_pixels, -1)
        if self.fix_categorical_prior ==True:
            pi_prior_single = torch.tensor([1/self.K]*self.K, requires_grad=False).to(self.device)
            pi_prior = pi_prior_single[None,None,:].repeat(bs,number_of_pixels,1)
        else:
            if (cfg.model.simple_cat ==True):
                prior_input = coordinates # p(c_d|x_d)  bs, #pixel, 256
            else:
                z_repeated = z.unsqueeze(1).expand(-1,number_of_pixels,-1)     #bs, pixel, dim_z   
                prior_input = torch.cat((coordinates,z_repeated),dim=2) # q(c_d|x_d,z)  bs, #pixel, 256+z

            prior_input_parallel = prior_input.reshape(bs*number_of_pixels, -1) #torch.Size([75264, 34])
            pi_prior = self.prior_cat_encoder(prior_input_parallel).reshape(bs, number_of_pixels, -1) #INFO these do not have dropout due to summation of pi s

        return pi_prior

    def post_cat(self, coordinates, features, z):
        """
            Parameters:
            coordinates: encoded_coordinates
            z: latent variable
        """        
        bs = coordinates.shape[0]
        number_of_pixels = coordinates.shape[1]
        coordinates = self.cat_encoder_x(coordinates.reshape(bs*number_of_pixels,-1)).reshape(bs, number_of_pixels, -1)
        z_repeated = z.unsqueeze(1).expand(-1,number_of_pixels,-1)

        if cfg.model.post_cat_has_z == True:
            if cfg.model.simple_cat ==True:
                post_input = torch.cat((coordinates, z_repeated),dim=2) # p(c_d|x_d,z) bs, #pixel, (2 or 256) 256+z
            else:
                post_input = torch.cat((coordinates, features, z_repeated),dim=2) # q(c_d|x_d,y_d,z) bs, #pixel, (256) +#channel+z
        else:
            raise NotImplemented
        post_input_parallel = post_input.reshape(bs*number_of_pixels, -1) #torch.Size([75264, 34])
        pi_post = self.post_cat_encoder(post_input_parallel).reshape(bs, number_of_pixels, -1)

        if self.freeze_categorical_posterior:
            pi_post = pi_post.detach()

        return pi_post
    
    def mask_to_input(self, input:torch.Tensor, mask:torch.Tensor)-> torch.Tensor:
        '''
        Args:
            input (torch.Tensor): Shape (batch_size, num_points, coordinate_dim or channel_dim).
            coor_mask (torch.Tensor): Shape (batch_size, num_points).
        Returns:
            missing_input (torch.Tensor): Shape (batch_size, num_points_not_masked,coordinate_dim or channel_dim)
        '''
        missing_input = input[mask,:].reshape(input.shape[0],-1,input.shape[2])
        return missing_input

    def create_outcoordinates(self, resolution:torch.Tensor, number_of_samples=1)-> torch.Tensor:
        '''
        Args:
        '''
        # out_coordinates = self.data_converter.coordinates.repeat(number_of_samples, 1, 1) #[bs,h*w,coord_dim]

        coordinates = self.data_converter.superresolve_coordinates(resolution) #[h*w,coord_dim]
        if self.task == "image":
            out_coordinates = coordinates.repeat(number_of_samples, 1, 1)
        elif self.task == "era5_polar":
            out_coordinates = coordinates.repeat(number_of_samples, 1, 1)
        elif self.task == "voxels_chairs":
            out_coordinates = coordinates.repeat(number_of_samples, 1, 1)
        
        return out_coordinates
    
    def _to_coordinates_and_features(self, batch:torch.Tensor):

        x = batch
        x_norm_im, x_norm_im_bn = self.preprocess_batch(x) #all of them (including input) [bs, 1, 28, 28] #if scaler is minmax1 they are same
        coors_point_all, x_norm_point_all = self.data_converter.batch_to_coordinates_and_features(data_batch=x_norm_im) #[bs, h*w, 2] #[bs,h*w,ch]
        return coors_point_all, x_norm_point_all

    def _encoder_to_inference(self, mode:str):
        if self.encoder_type == "pointconv":
            if mode=="inference":
                self.encoder_z.same_coordinates="batch"
                for i in self.encoder_z.layers:
                    try:
                        if isinstance(i.same_coordinates, str):
                            i.same_coordinates="batch"
                    except:
                        continue
            else:
                self.encoder_z.same_coordinates=self.initial_same_coordinates
                for i in self.encoder_z.layers:
                    try:
                        if isinstance(i.same_coordinates, str):
                            i.same_coordinates=self.initial_same_coordinates
                    except:
                        continue
        
        else:
            raise NotImplementedError
        
    def print_params_count(self, logging):
        
        # for part in [self.encoder_z, self.prior_cat_encoder, self.post_cat_encoder, self.cat_encoder_x]:
        total_params = 0

        params = params_count(self.encoder_z)
        logging.info('\tNum parameters encoder_z: {}'.format(params))
        total_params += params

        params = params_count(self.prior_cat_encoder)
        logging.info('\tNum parameters prior_cat_encoder: {}'.format(params))
        total_params += params

        params = params_count(self.post_cat_encoder)
        logging.info('\tNum parameters encoder_z: {}'.format(params))
        total_params += params

        params = params_count(self.cat_encoder_x)
        logging.info('\tNum parameters cat_encoder_x: {}'.format(params))
        total_params += params
        
        params_fn_rep = params_count(self.decoder.mlp)
        logging.info('\tNum parameters decoder, not counted but modeled: {}'.format(params_fn_rep))

        k = len(self.hyper_list)
        params = params_count(self.hyper_list[0].mlp) * k
        logging.info('\tNum parameters hyper: {}'.format(params))
        total_params += params

        logging.info('Num of total parameters: {}'.format(total_params))

        return total_params
    
def params_count(model):
        """Computes the number of parameters."""
        return sum([p.numel() for p in model.parameters()])