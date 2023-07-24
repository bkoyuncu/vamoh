from locale import normalize
import logging
import time
from importlib_metadata import metadata
import torch
from imagegym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from imagegym.config import cfg
from imagegym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from tqdm import tqdm
from imagegym.utils.scenerios import * 
import matplotlib.pyplot as plt
from imagegym.utils.mask import compute_occlusion_mask, apply_occlusion_mask, create_fixed_mask_missingness

# Get the color map by name:
cm = plt.get_cmap('jet')

def add_custom_stats_from_loss_dict(loss_dict):
    custom_stats = {}

    for key, value in loss_dict.items():
        if key not in ["loss","z"]:
            try:
                custom_stats[key] = value.item()
            except:
                custom_stats[key] = value
    return custom_stats


def train_epoch(logger, loader, model, optimizer, scheduler, epoch, dict_model_tuners):
    

    _ = dict_model_tuners["scenerios"].check_scenerio(epoch, model)
    
    if dict_model_tuners["adjust_beta_c"].check_update(epoch)!=None:
        model.beta_c = dict_model_tuners["adjust_beta_c"].check_update(epoch)
    if dict_model_tuners["adjust_beta_z"].check_update(epoch)!=None:
        model.beta_z = dict_model_tuners["adjust_beta_z"].check_update(epoch)
    
    model.train()

    num_batches = len(loader)

    if cfg.dataset.use_number_batch>0:
        num_batches_max = cfg.dataset.use_number_batch
        print(f"Using {cfg.dataset.use_number_batch} batches of the dataset: {num_batches_max}/{num_batches}")
    
    else:
        num_batches_max = num_batches
        
    my_iter = loader
    
    
    i = 0
    for batch in tqdm(my_iter):
        time_start_dl = time.time()
        batch = [b.to(torch.device(cfg.device)) for b in batch]
        i += 1
        
        time_passed_dl = (time.time() - time_start_dl)
        time_start_gpu = time.time()

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model(batch,missingness=cfg.dataset.missing_perc)

        loss = loss_dict['loss']
        loss.backward()

        if cfg.train.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)

        optimizer.step()
        
        custom_stats = add_custom_stats_from_loss_dict(loss_dict)

        logger.update_stats(batch_size=batch[0].shape[0],
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start_gpu,
                            time_used_for_dl = time_passed_dl,
                            params=cfg.params,
                            **custom_stats)

        if i == (num_batches_max): 
            print(f"training exited at batch number:{i}/{num_batches}")
            break

    if cfg.optim.use_scheduler:
        scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, cur_epoch, split="none"):
    
    model.eval()

    num_batches = len(loader)

    if cfg.dataset.use_number_batch>0:
        num_batches_max = cfg.dataset.use_number_batch
        print(f"Using {cfg.dataset.use_number_batch} batches of the dataset: {num_batches_max}/{num_batches}")
    
    else:
        num_batches_max = num_batches

    
    i = 0
    
    for batch in loader:
        time_start_dl = time.time()
        batch = [b.to(torch.device(cfg.device)) for b in batch]
        i += 1

        time_passed_dl = (time.time() - time_start_dl)
        time_start_gpu = time.time()
 
        loss_dict = model(batch,missingness=0.0)

        custom_stats = add_custom_stats_from_loss_dict(loss_dict)
        
        logger.update_stats(batch_size=batch[0].shape[0],
                            loss=loss_dict['loss'].item(),
                            lr=0.0,
                            time_used=time.time() - time_start_gpu,
                            time_used_for_dl = time_passed_dl,
                            params=cfg.params,
                            **custom_stats)
        
        if i == (num_batches_max): 
            print(f"eval exited at batch number:{i}/{num_batches}")
            break
    

    original_size = batch[0].shape[2:]
    resolution_0 = tuple(original_size)
    resolution_1 = tuple(i*2 - 1 for i in resolution_0)
    normalize_bool = False
    bs = batch[0].shape[0]
    if model.task == 'image':
        if (cur_epoch+1) % cfg.plotting.res_epoch ==0 or cur_epoch==0:
            #Generation#
            number_of_samples = 8
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
            info_x, info_z = model.sample([number_of_samples], resolution = resolution_0, out_coordinates = out_coordinates, z =None)
            logger.save_image_grid(x=info_x['mean'],
                                name=f"gener_x_mean_{cur_epoch}",
                                    normalize=normalize_bool)

            #Generation Super Res#
            if (cur_epoch+1) % cfg.plotting.super_res_epoch ==0 or cur_epoch==0:
                out_coordinates = model.create_outcoordinates(resolution=resolution_1, number_of_samples=number_of_samples)
                info_x, info_z = model.sample([number_of_samples], resolution = resolution_1, out_coordinates = out_coordinates, z =info_z['sample'])
                logger.save_image_grid(x=info_x['mean'],
                                    name=f"gener_super_x_mean_{cur_epoch}",
                                    normalize=normalize_bool)

            #Reconstruction#
            coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)

            _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = None, occlusion_size=None)
            info_x, info_z = model.reconstruct(coordinates_features=[coors_point_all, x_norm_point_all], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
            if cur_epoch==0:
                logger.save_image_grid(x=batch[0][:number_of_samples],
                                name=f"recons_x_original_{cur_epoch}",
                                normalize=normalize_bool)
            logger.save_image_grid(x=info_x['mean'],
                                name=f"recons_x_mean_{cur_epoch}",
                                normalize=normalize_bool)

            #Reconstruction Super Res#
            if (cur_epoch+1) % cfg.plotting.super_res_epoch ==0 or cur_epoch==0:
            # Resolution_1
                coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
                out_coordinates = model.create_outcoordinates(resolution=resolution_1, number_of_samples=number_of_samples)
                _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = None, occlusion_size=None)
                info_x, info_z = model.reconstruct(coordinates_features=[coors_point_all, x_norm_point_all], 
                                                    resolution=resolution_1, 
                                                    out_coordinates=out_coordinates,
                                                    mask=_mask)
                logger.save_image_grid(x=info_x['mean'],
                                    name=f"recons_super_x_mean_{cur_epoch}",
                                    normalize=normalize_bool)

            #Completion Tasks#
            if (cur_epoch+1) % cfg.plotting.super_res_epoch ==0 or cur_epoch==0:
                model._encoder_to_inference(mode="inference")
                coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
                occlusion_size=(5,cfg.dataset.dims[1]//3)
                _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = "inpainting", occlusion_size=occlusion_size)
                coors_point_masked, x_norm_point_masked = apply_occlusion_mask(coordinates=coors_point_all, features=x_norm_point_all, mask=_mask)

                #output coors
                out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
                info_x, info_z = model.reconstruct(coordinates_features=[coors_point_masked, x_norm_point_masked], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
                
                #save ground truth with occlusion
                gt_features = batch[0][:number_of_samples].clone()
                gt_features[:,:,~_mask]=0
                
                #use gt features for observed parts
                info_x['mean'][:,:,_mask] = gt_features[:,:,_mask]

                if cur_epoch==0:
                    logger.save_image_grid(x=gt_features,
                                name=f"recons_miss_x_original_occ_{cur_epoch}",
                                normalize=normalize_bool)

                logger.save_image_grid(x=info_x['mean'],
                                    name=f"recons_miss_x_mean_{cur_epoch}",
                                    normalize=normalize_bool)

                #half
                coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
                _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = "half", occlusion_size=(None,None))
                coors_point_masked, x_norm_point_masked = apply_occlusion_mask(coordinates=coors_point_all, features=x_norm_point_all, mask=_mask)

                #output coors
                out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
                info_x, info_z = model.reconstruct(coordinates_features=[coors_point_masked, x_norm_point_masked], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
                
                #save ground truth with occlusion
                gt_features = batch[0][:number_of_samples].clone()
                gt_features[:,:,~_mask]=0
                
                #use gt features for observed parts
                info_x['mean'][:,:,_mask] = gt_features[:,:,_mask]

                if cur_epoch==0:
                    logger.save_image_grid(x=gt_features,
                                name=f"recons_half_x_original_occ_{cur_epoch}",
                                normalize=normalize_bool)

                logger.save_image_grid(x=info_x['mean'],
                                    name=f"recons_half_x_mean_{cur_epoch}",
                                    normalize=normalize_bool)

                model._encoder_to_inference(mode="train")
    
    if model.task == "era5_polar":
        if (cur_epoch+1) % cfg.plotting.res_epoch ==0 or cur_epoch==0:
            #Generation#
            number_of_samples = 8
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
            info_x, info_z = model.sample([number_of_samples], resolution = resolution_0, out_coordinates = out_coordinates, z =None)
            data= model.data_converter.batch_to_data(coordinates=None, features=info_x['mean'].permute(0,2,3,1))
            logger.save_era(x=data.cpu(),name=f"gener_x_mean_{cur_epoch}",globe=False) 

            #Generation Super Res#
            if (cur_epoch+1) % cfg.plotting.super_res_epoch ==0 or cur_epoch==0:
                out_coordinates = model.create_outcoordinates(resolution=resolution_1, number_of_samples=number_of_samples)
                info_x, info_z = model.sample([number_of_samples], resolution = resolution_1, out_coordinates = out_coordinates, z =info_z["sample"])
                data= model.data_converter.batch_to_data(coordinates=None, features=info_x['mean'].permute(0,2,3,1), resolution = resolution_1)
                logger.save_era(x=data.cpu(),name=f"gener_super_x_mean_{cur_epoch}",globe=False) 
        
            #Reconstruction#
            coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
            _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = None, occlusion_size=None)
            info_x, info_z = model.reconstruct(coordinates_features=[coors_point_all, x_norm_point_all], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
            if cur_epoch==0:
                gt_features = model.data_converter.batch_to_data(features=x_norm_point_all)
                logger.save_era(x=gt_features.cpu(),name=f"recons_x_original_{i}_{cur_epoch}")
            data= model.data_converter.batch_to_data(coordinates=None, features=info_x['mean'].permute(0,2,3,1))
            logger.save_era(x=data.cpu(),name=f"recons_x_mean_{i}_{cur_epoch}",globe=False)
            
            #Reconstruction Super Res#
            coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
            out_coordinates = model.create_outcoordinates(resolution=resolution_1, number_of_samples=number_of_samples)
            _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = None, occlusion_size=None)
            info_x, info_z = model.reconstruct(coordinates_features=[coors_point_all, x_norm_point_all], 
                                                resolution=resolution_1, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
            if cur_epoch==0:
                gt_features = model.data_converter.batch_to_data(features=x_norm_point_all)
                logger.save_era(x=gt_features.cpu(),name=f"recons_x_original_{i}_{cur_epoch}")
            data= model.data_converter.batch_to_data(coordinates=None, features=info_x['mean'].permute(0,2,3,1),resolution=resolution_1)
            logger.save_era(x=data.cpu(),name=f"recons_super_x_mean_{i}_{cur_epoch}",globe=False) 

            #Completion Tasks#
            if (cur_epoch+1) % cfg.plotting.super_res_epoch ==0 or cur_epoch==0:
                model._encoder_to_inference(mode="inference")
                #INPAINTING
                coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
                occlusion_size=(5,cfg.dataset.dims[1]//3)
                _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = "inpainting", occlusion_size=occlusion_size)
                coors_point_masked, x_norm_point_masked = apply_occlusion_mask(coordinates=coors_point_all, features=x_norm_point_all, mask=_mask)

                #output coors
                out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
                info_x, info_z = model.reconstruct(coordinates_features=[coors_point_masked, x_norm_point_masked], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
                gt_all = model.data_converter.batch_to_data(features=x_norm_point_all)
                gt_coordinates, gt_features = gt_all[:,:2].clone(), gt_all[:,2:].clone()

                #save ground truth with occlusion
                gt_features[:,:,~_mask]=0
                
                #use gt features for observed parts
                info_x['mean'][:,:,_mask] = gt_features[:,:,_mask]

                if cur_epoch==0:
                    gt_all = torch.concatenate([gt_coordinates,gt_features], dim=1)
                    logger.save_era(x=gt_all,
                                name=f"recons_miss_x_original_occ_{cur_epoch}",
                                globe=False)
                
                data= model.data_converter.batch_to_data(coordinates=None, features=info_x['mean'].permute(0,2,3,1))
                logger.save_era(x=data.cpu(),name=f"recons_miss_x_mean_{i}_{cur_epoch}",globe=False) 

                #HALF
                coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
                occlusion_size=(5,cfg.dataset.dims[1]//3)
                _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = "half", occlusion_size=occlusion_size)
                coors_point_masked, x_norm_point_masked = apply_occlusion_mask(coordinates=coors_point_all, features=x_norm_point_all, mask=_mask)

                #output coors
                out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
                info_x, info_z = model.reconstruct(coordinates_features=[coors_point_masked, x_norm_point_masked], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
                gt_all = model.data_converter.batch_to_data(features=x_norm_point_all)
                gt_coordinates, gt_features = gt_all[:,:2].clone(), gt_all[:,2:].clone()

                #save ground truth with occlusion
                gt_features[:,:,~_mask]=0
                
                #use gt features for observed parts
                info_x['mean'][:,:,_mask] = gt_features[:,:,_mask]

                if cur_epoch==0:
                    gt_all = torch.concatenate([gt_coordinates,gt_features], dim=1)
                    logger.save_era(x=gt_all,
                                name=f"recons_half_x_original_occ_{cur_epoch}",
                                globe=False)
                
                data= model.data_converter.batch_to_data(coordinates=None, features=info_x['mean'].permute(0,2,3,1))
                logger.save_era(x=data.cpu(),name=f"recons_half_x_mean_{i}_{cur_epoch}",globe=False) 
                
                model._encoder_to_inference(mode="train")
        
    if model.task == "voxels_chairs":
        if (cur_epoch+1) % cfg.plotting.res_epoch ==0 or cur_epoch==0: 
            #Generation#
            number_of_samples = 4
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
            info_x, info_z = model.sample([number_of_samples], resolution = resolution_0, out_coordinates = out_coordinates, z =None, mask=None)
            logger.save_voxel(x=info_x['mean'], name=f"gener_x_mean_{cur_epoch}")

            #Generation Super Res#
            if (cur_epoch+1) % cfg.plotting.super_res_epoch ==0 or cur_epoch==0:
                out_coordinates = model.create_outcoordinates(resolution=resolution_1, number_of_samples=number_of_samples)
                info_x, info_z = model.sample([number_of_samples], resolution = resolution_1, out_coordinates = out_coordinates, z = info_z["sample"], mask=None)
                logger.save_voxel(x=info_x['mean'], name=f"gener_super_x_mean_{cur_epoch}")
            
            #Reconstruction#
            coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
            voxel_mask, voxel_mask_point =create_fixed_mask_missingness((number_of_samples,1,*resolution_0))
            _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = None, occlusion_size=None)
            info_x, info_z = model.reconstruct(coordinates_features=[coors_point_all, x_norm_point_all], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
            if cur_epoch==0:
                gt_features = model.data_converter.batch_to_data(features=x_norm_point_all,coordinates = coors_point_all)
                logger.save_voxel(x=gt_features.cpu(),name=f"recons_x_original_{i}_{cur_epoch}")
            info_x["mean"][info_x["mean"]>0.5]=1
            info_x["mean"][info_x["mean"]<=0.5]=0
            logger.save_voxel(x=info_x["mean"].cpu(),name=f"recons_x_mean_{i}_{cur_epoch}")

            #Reconstruction Super Res#
            coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
            out_coordinates = model.create_outcoordinates(resolution=resolution_1, number_of_samples=number_of_samples)
            voxel_mask, voxel_mask_point =create_fixed_mask_missingness((number_of_samples,1,*resolution_1))
            _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = None, occlusion_size=None)
            info_x, info_z = model.reconstruct(coordinates_features=[coors_point_all, x_norm_point_all], 
                                                resolution=resolution_1, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
            info_x["mean"][info_x["mean"]>0.5]=1
            info_x["mean"][info_x["mean"]<=0.5]=0
            logger.save_voxel(x=info_x["mean"].cpu(),name=f"recons_super_x_mean_{i}_{cur_epoch}")

            #Completion#
            coors_point_all, x_norm_point_all = model._to_coordinates_and_features(batch[0][:number_of_samples])
            out_coordinates = model.create_outcoordinates(resolution=resolution_0, number_of_samples=number_of_samples)
            voxel_mask, voxel_mask_point =create_fixed_mask_missingness((number_of_samples,1,*resolution_0))
            occlusion_size=(5,cfg.dataset.dims[1]//3)
            _mask = compute_occlusion_mask(input_size=cfg.dataset.dims[1:], occlusion_type = "inpainting", occlusion_size=occlusion_size)
            coors_point_masked, x_norm_point_masked = apply_occlusion_mask(coordinates=coors_point_all, features=x_norm_point_all, mask=_mask)
            info_x, info_z = model.reconstruct(coordinates_features=[coors_point_masked, x_norm_point_masked], 
                                                resolution=resolution_0, 
                                                out_coordinates=out_coordinates,
                                                mask=_mask)
            
            gt_features = model.data_converter.batch_to_data(features=x_norm_point_all,coordinates = coors_point_all)
            #save ground truth with occlusion
            gt_features[:,:,~_mask]=0
            
            #use gt features for observed parts
            info_x['mean'][:,:,_mask] = gt_features[:,:,_mask]

            if cur_epoch==0:
                logger.save_voxel(x=gt_features,
                            name=f"recons_miss_x_original_occ_{cur_epoch}")
            
            logger.save_voxel(x=info_x['mean'].cpu(),name=f"recons_miss_x_mean_{i}_{cur_epoch}")

def train(loggers, loaders, model, optimizer, scheduler, scenerios):
    start_epoch = 0
    logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    if len(loaders)==3:
        split_names = ["train","val","test"]
    if len(loaders)==2:
        split_names = ["train","test"]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cur_epoch, scenerios)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                if i==2:
                    break
                eval_epoch(loggers[i], loaders[i], model,
                           cur_epoch=cur_epoch, split=split_names[i])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
    