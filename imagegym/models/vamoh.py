#some snippets used from https://github.com/EmilienDupont/neural-function-distributions
import torch
import torch.nn as nn
import math
from imagegym.config import cfg
from imagegym.loss import ELBO_mixture_observed
from functorch import vmap

import torch
import torch.nn as nn
from functorch import vmap
from .vamoh_base import vamoh_base

from imagegym.utils.mask import *
from imagegym.utils.priors import *


class VAMOH(vamoh_base):
    def __init__(self, **kwargs) -> None:
        super(VAMOH,self).__init__(**kwargs)

        self.prior_z = self.set_z_prior_distr(kwargs['distr_z'])

    @staticmethod
    def kwargs(cfg, preparator):
        model = cfg.model
        layer = cfg.layer

        return {
            'dim_in': preparator.dim_coordinates(),
            'dim_out': preparator.dim_features(),
            'data_converter': preparator.data_converter(),
            'dim_latent': model.dim_latent,
            'num_layers': model.num_layers,
            'dim_hidden': model.dim_inner,
            'w0_initial': layer.w0_initial
        }


    def forward(self, batch, missingness: float = 0.0, **kwargs):

        x, label = batch[0], batch[1]

        x, observed_mask, observed_mask_point = create_mask_missingness(x,missingness=missingness)

        x_norm_im, x_norm_im_bn = self.preprocess_batch(x) #all of them (including input) [bs, 1, 28, 28] #if scaler is minmax1 they are same
        coors_point_all, x_norm_point_all = self.data_converter.batch_to_coordinates_and_features(data_batch=x_norm_im) #[bs, h*w, 2] #[bs,h*w,ch] (so this is also proper for shapenet + voxels)


        bs = coors_point_all.shape[0]
        
        #TODO think about sth more mearningful
        if self.create_encoded_coords:
            self.encoded_coors_point_all = self.create_encoded_coordinates_fwd(coordinates=coors_point_all) #if shapenet
        if bs != self.encoded_coors_point_all.shape[0]:
            self.encoded_coors_point_all = self.create_encoded_coordinates_fwd(coordinates=coors_point_all)

        # x_norm_im = x_norm_im * observed_mask
        x_norm_point = self.mask_to_input(input=x_norm_point_all,mask=observed_mask_point) #[bs,#points,ch]
        encoded_coors_point = self.mask_to_input(input=self.encoded_coors_point_all,mask=observed_mask_point) #[bs,#points,ch]
        coors_point =  self.mask_to_input(input=coors_point_all,mask=observed_mask_point) #[bs,#points,ch]
        
        outputs_z_dict = self._encode_z(coordinate=encoded_coors_point, features=x_norm_point)
        qz_x = outputs_z_dict["qz_x"]
        z = self._sample_z(qz_x)
        
        outputs_c_dict = self._encode_c(coordinate=encoded_coors_point, features=x_norm_point, z=z)

        logits_x = self._decode(z,coors_point_all,self.encoded_coors_point_all,resolution=None, return_image_format=False)

        logits_x = logits_x.reshape(self.K, bs, *logits_x.shape[1:]).permute((1,2,3,0))

        pi_x = nn.functional.softmax(logits_x,dim=-2) # [bs, pix_full, ch, K]
        pi_x = torch.clamp(pi_x, min=1e-6, max=None)

        if "logistic" in self.distr_x:
            mean_x, px_z = self.lik_x(logits_scales=[logits_x, self.lik_x_logscales], 
                                    return_mean=True, dim=-1) #mean_x: [bs,pixels,ch,K]

        else:
            mean_x, px_z = self.lik_x(logits=logits_x,
                                    return_mean=True, dim=-1) #this has the whole [bs,pixels,ch,K]

        loss_dict = ELBO_mixture_observed(qz_x=qz_x,
                        pz=self.prior_z,
                        px_z=px_z,
                        x=x_norm_point_all, #[bs,full_points,ch]
                        beta=self.beta_z,
                        beta_c = self.beta_c,
                        qc=outputs_c_dict["qc"],
                        pc=outputs_c_dict["pc"],
                        K=self.K,
                        mask = observed_mask_point)
        
        loss_dict['loss'] = - (loss_dict['elbo'])
        return loss_dict

    def elbo(self, batch, missingness: float = 0.0, **kwargs):

        x, label = batch[0], batch[1]

        x, observed_mask, observed_mask_point = create_mask_missingness(x,missingness=0.0)

        x_norm_im, x_norm_im_bn = self.preprocess_batch(batch[0]) #all of them (including input) [bs, 1, 28, 28] #if scaler is minmax1 they are same
        coors_point_all, x_norm_point_all = self.data_converter.batch_to_coordinates_and_features(data_batch=x_norm_im) #[bs, h*w, 2] #[bs,h*w,ch] (so this is also proper for shapenet + voxels)


        bs = coors_point_all.shape[0]
        
        if self.create_encoded_coords:
            self.encoded_coors_point_all = self.create_encoded_coordinates_fwd(coordinates=coors_point_all) #if shapenet
        if bs != self.encoded_coors_point_all.shape[0]:
            self.encoded_coors_point_all = self.create_encoded_coordinates_fwd(coordinates=coors_point_all)

        x_norm_im = x_norm_im * observed_mask
        x_norm_point = self.mask_to_input(input=x_norm_point_all,mask=observed_mask_point) #[bs,#points,ch]
        encoded_coors_point = self.mask_to_input(input=self.encoded_coors_point_all,mask=observed_mask_point) #[bs,#points,ch]
        coors_point =  self.mask_to_input(input=coors_point_all,mask=observed_mask_point) #[bs,#points,ch]

        outputs_z_dict = self._encode_z(coordinate=encoded_coors_point, features=x_norm_point)
        qz_x = outputs_z_dict["qz_x"]
        z = self._sample_z(qz_x)
        
        outputs_c_dict = self._encode_c(coordinate=encoded_coors_point, features=x_norm_point, z=z)

        logits_x = self._decode(z,coors_point_all,self.encoded_coors_point_all,resolution=None, return_image_format=False)

        pi_x = nn.functional.softmax(logits_x,dim=-2) # [bs, pix_full, ch, K]
        pi_x = torch.clamp(pi_x, min=1e-6, max=None)

        if "logistic" in self.distr_x:
            mean_x, px_z = self.lik_x(logits_scales=[logits_x, self.lik_x_logscales], 
                                    return_mean=True, dim=-1) #mean_x: [bs,pixels,ch,K]

        else:
            mean_x, px_z = self.lik_x(logits=logits_x,
                                    return_mean=True, dim=-1) #this has the whole [bs,pixels,ch,K]

        loss_dict = ELBO_mixture_observed(qz_x=qz_x,
                        pz=self.prior_z,
                        px_z=px_z,
                        x=x_norm_point_all, #[bs,full_points,ch]
                        beta=self.beta_z,
                        beta_c = self.beta_c,
                        qc=outputs_c_dict["qc"],
                        pc=outputs_c_dict["pc"],
                        K=self.K,
                        mask = observed_mask_point)

        return loss_dict['elbo']

    
    @torch.no_grad()
    def sample(self, sample_size, resolution, out_coordinates, z=None, mask=None, **kwargs):
        assert isinstance(sample_size, list)

        if z is None:
            z = self.prior_z.sample(sample_size)
        bs = z.shape[0]
        

        encoded_out_coords = [self.decoder.position_encoding(coord) for coord in out_coordinates] #this makes [bs,size*size,2] -> [bs,size*size,256]
        encoded_out_coords_recons_all = torch.stack(encoded_out_coords, 0).expand(bs,-1,-1)

        logits_x = self._decode(z,
                                out_coordinates,
                                encoded_out_coords_recons_all,
                                resolution=resolution, 
                                return_image_format=False) #[bs*K, *input, ch]
        
        logits_x = self._return_image_format(logits_x, resolution=resolution, coordinates=out_coordinates, bs=bs, K=self.K, mask=mask)
        
        if "logistic" in self.distr_x:
            mean_x, px_z = self.lik_x(logits_scales=[logits_x, self.lik_x_logscales], 
                                    return_mean=True, dim=-1) #mean_x: [bs,pixels,ch,K]

        else:
            mean_x, px_z = self.lik_x(logits=logits_x,
                                    return_mean=True, dim=-1) #this has the whole [bs,pixels,ch,K]
        

        logits_pi = self.prior_cat(coordinates=encoded_out_coords_recons_all,z=z)
        logits_pi = logits_pi.reshape(mean_x.shape[0], 1, *mean_x.shape[2:])   # (n, 1, size, size, K)
        pi = nn.functional.softmax(logits_pi,dim=-1) # (n, 1, size, size, K)
        #marginalizes over all pi
        mean_x_out = (mean_x*pi).sum(-1) #[bs,ch,h,w]

        #takes the most probable pi (MAP)
        mask = (torch.nn.functional.one_hot(torch.argmax(pi, -1), num_classes=self.K)) #[bs,1,h,w,K]
        mean_x_map = (mean_x*mask).sum(-1) #[bs,ch,h,w]

        entropy = (-1 * torch.log(pi) * pi).sum(-1) # (n, 1, size, size)

        segm = torch.argmax(pi, -1).float() / (cfg.params_k_mixture.K-1 + 1e-6) #[bs,ch,h,w]


        info_x = {
                # 'mean': mean_x,
                'mean': self.postprocess_batch(mean_x_out),
                'mean_map': self.postprocess_batch(mean_x_map),
                'segm': segm,
                'entropy': entropy
            }
        
        info_z = {
            'sample': z
        }
        return info_x, info_z


    @torch.no_grad()
    def reconstruct(self, coordinates_features, resolution=None, out_coordinates=None, mask=None, **kwargs):
        
        coors_point_all, x_norm_point_all = coordinates_features[0], coordinates_features[1]
        bs = coors_point_all.shape[0]

        # coordinates = coors_point_all
        encoded_coords = [self.decoder.position_encoding(coord) for coord in coors_point_all] #this makes [bs,size*size,2] -> [bs,size*size,256]
        encoded_coords_recons_all = torch.stack(encoded_coords, 0).expand(bs,-1,-1)
        
        
        outputs_z_dict = self._encode_z(coordinate=encoded_coords_recons_all, features=x_norm_point_all)
        z = outputs_z_dict["mean_z"]

        encoded_out_coords = [self.decoder.position_encoding(coord) for coord in out_coordinates] #this makes [bs,size*size,2] -> [bs,size*size,256]
        encoded_out_coords_recons_all = torch.stack(encoded_out_coords, 0).expand(bs,-1,-1)

        logits_x = self._decode(z,
                                out_coordinates,
                                encoded_out_coords_recons_all,
                                resolution=resolution, 
                                return_image_format=False)

        logits_x = self._return_image_format(logits_x, resolution=resolution, coordinates=out_coordinates, bs=bs, K=self.K)

        if "logistic" in self.distr_x:
            mean_x, px_z = self.lik_x(logits_scales=[logits_x, self.lik_x_logscales], 
                                    return_mean=True, dim=-1) #mean_x: [bs,pixels,ch,K]

        else:
            mean_x, px_z = self.lik_x(logits=logits_x,
                                    return_mean=True, dim=-1) #this has the whole [bs,pixels,ch,K]

        x_rec = px_z.sample() ##[bs,ch,h,w,K]

        logits_pi = self.prior_cat(coordinates=encoded_out_coords_recons_all,z=z)
        logits_post = self.post_cat(coordinates=encoded_coords_recons_all,features=x_norm_point_all,z=z)

        # merge_mask = torch.zeros((mask.shape),dtype=torch.bool)
        scale_pixels = math.ceil(resolution[0]/mask.shape[0])
        resolution_merge_mask = torch.zeros(resolution,dtype=torch.bool)

        if len(resolution)==2:
            resolution_merge_mask[::scale_pixels,::scale_pixels][mask]=True
        if len(resolution)==3:
            resolution_merge_mask[::scale_pixels,::scale_pixels,::scale_pixels][mask]=True

        logits_pi[:,resolution_merge_mask.flatten(),:] = logits_post

        pi = nn.functional.softmax(logits_pi,dim=-1) # [bs, #points, K]
        
        pi = pi.reshape(mean_x.shape[0], 1,  *x_rec.shape[2:])   # (n, 1, size, size, K)

        #marginalizes over all pi
        mean_x_out = (mean_x*pi).sum(-1) # [bs,ch,h,w,K] * [bs,1,h,w,K] -> [bs,ch,h,w]

        #takes the most probable pi (MAP)
        map_mask = (torch.nn.functional.one_hot(torch.argmax(pi, -1), num_classes=self.K)) #[bs,ch,h,w,K] 
        mean_x_map = (mean_x*map_mask).sum(-1) #[bs,ch,h,w]

        entropy = (-1 * torch.log(pi) * pi).sum(-1) # (n, 1, size, size)
        segm = torch.argmax(pi, -1).float() / (cfg.params_k_mixture.K-1 + 1e-6) #[bs,1,h,w]
        
        outputs_x_dict = {
            'mean': self.postprocess_batch(mean_x_out), #marginalized with pi's [bs,ch,h,w]
            'mean_map': self.postprocess_batch(mean_x_map),
            'segm': segm, #[bs,1,h,w]
            'pi': pi, #[bs,ch,h,w,k]
            'entropy': entropy 
        }

        return outputs_x_dict, outputs_z_dict
    
    def _encode_z(self, coordinate, features):
        # logits_z = self.encoder_z(coordinate, features)
        if self.encoder_type == "pointconv":
            logits_z = self.encoder_z(coordinate, features)
        else:
            raise ValueError("Unknown encoder type")
        
        mean_z, qz_x = self.lik_z(logits=logits_z,
                                    return_mean=True)

        outputs_dict = {
            "mean_z": mean_z,
            "qz_x": qz_x,
        }

        return outputs_dict
    
    def _encode_c(self, coordinate, features, z):
        #Categorical Logits
        logits_prior = self.prior_cat(coordinates=coordinate,z=z)
        logits_post = self.post_cat(coordinates=coordinate,features=features,z=z)
        if self.learn_residual_posterior==True:
            logits_post = logits_prior + logits_post
        
        #PIs (normalized)
        pi_prior = nn.functional.softmax(logits_prior,dim=-1)
        pi_post = nn.functional.softmax(logits_post,dim=-1) #bs,h*w,K
        
        #Categorical Distributions
        qc = torch.distributions.categorical.Categorical(probs=pi_post)
        pc = torch.distributions.categorical.Categorical(probs=pi_prior)

        outputs_dict = {
            'qc': qc,
            'pc': pc,
            'pi_prior': pi_prior,
            'pi_post': pi_post,
            'logits_prior': logits_prior,
            'logits_post': logits_post,
        }

        return outputs_dict

    def _return_image_format(self, logits, coordinates=None, resolution=None, bs=None, K=None, mask=None):
        if self.task == "image":
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,4,0))
        elif self.task == "era5_polar":
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,4,0))
            logits = logits[:,[2]] #even though we use polar for input it returns 2d due to dataconverter
        elif self.task == "voxels_chairs":
            if mask is not None:
                logits[:,~mask[0],:]=0
            logits = self.data_converter.batch_to_data(coordinates=coordinates, 
                                            features=logits,
                                            resolution=resolution)
            logits= logits.view(K, bs, *logits.shape[1:]).permute((1,2,3,4,5,0))

        return logits


    def _decode(self, z, coordinates, coordinates_encoded=None, resolution=None, return_image_format=True):
        
        stack_w_and_b =[]
        for i in range(self.K):
            pure_wb = self.hyper_list[i](latents=z)
            stack_w_and_b.append(pure_wb)
        
        all_w_and_b = torch.cat(stack_w_and_b) #[K*bs,dims_for_all_weight_and_biases]
        all_weights_and_biases_new=self.hyper_list[0].output_to_weights_2(all_w_and_b)
        bs, nop, dim = coordinates_encoded.shape

        cnew = coordinates_encoded.expand(self.K,-1,-1,-1)
        cnew = cnew.reshape(-1, cnew.shape[-2],cnew.shape[-1])
        fbuffers_new = tuple(p[:self.K*bs] for p in self.fbuffers)
        
        logits_new = vmap(self.fmodel)(all_weights_and_biases_new, fbuffers_new, cnew)

        return logits_new

    def _sample_z(self, dist):
        #samples a batch of z
        z = dist.rsample()
        return z
    def _sample_c(self, dist):
        #samples a batch of c
        return NotImplementedError

    @torch.no_grad()
    def _predict(self, latent, use_super_resolution=False):
        pred_img_i = self.wrapper(latent=latent, use_super_resolution=use_super_resolution)  # (1, 3, 256, 256)
        return pred_img_i
    @torch.no_grad()
    def _sample(self, latent, use_super_resolution=False):
        pred_img_i = self.wrapper(latent=latent, use_super_resolution=use_super_resolution)  # (1, 3, 256, 256)
        return pred_img_i
    
    def set_z_prior_distr(self,dist_name):
        
        self.prior_distr_z = PriorDistribution(dist_name,self.dim_z,self.device)
        
        return self.prior_distr_z
    
    # def _create_scenerios(self):
