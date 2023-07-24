import cartopy.crs as ccrs
import matplotlib.pyplot as plt
params = {'text.usetex': False}
plt.rcParams.update(params)
from scipy.interpolate import interp2d
import numpy as np
import warnings
from math import ceil
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# from tueplots import bundles
# plt.rcParams.update(bundles.icml2022())

def globe_plot_from_data_batch(batch, threshold=0.5, view=None, globe=False,smooth_factor=1.0, save_fig='',**kwargs):

        x = batch.numpy()
        batch_size = x.shape[0]

        ncol = int(np.ceil(np.sqrt(batch_size)))

        nrow = 1 if batch_size == 2 else ncol
        if globe:
            view = (100., 0.)
            projection = ccrs.Orthographic(*view)
        else:
            projection = ccrs.PlateCarree()

        N = batch_size

        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4 * nrow, 2 * ncol),
                                 subplot_kw={'projection': projection}
                                 )

        # fig, axes = plt.subplots(ncols=ncol, nrows=ceil(N/ncol), layout='constrained',
        #                         figsize=(3.5 * 4, 3.5 * ceil(N/ncol)),
        #                          subplot_kw={'projection': projection}
        #                          )

        idx = 0
        for i in range(nrow):
            for j in range(ncol):
                if nrow == 1 and ncol == 1:
                    ax_ij = axes
                elif nrow == 1:
                    ax_ij = axes[j]
                else:
                    ax_ij = axes[i, j]

                x_i = x[idx]
                latitude = x_i[0, :, 0]
                longitude = x_i[1, 0, :]
                temperature = x_i[2]


                mesh = ax_ij.pcolormesh(longitude, latitude, temperature,
                                        transform=ccrs.PlateCarree(),
                                        cmap='plasma')
                # Add coastlines
                ax_ij.coastlines()                                        
                # x_i = np.transpose( x[idx], (1, 2, 0))

                # ax_ij.set_xticks([])
                # ax_ij.set_yticks([])
                # ax_ij.set_axis_off()
                idx += 1
                if idx == batch_size: break
            if idx == batch_size: break
        
        plt.tight_layout()

        
        if len(save_fig):
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_fig)
            plt.clf()
            plt.close('all')
        else:
            plt.show()

def globe_plot_from_data_batch2(batch, threshold=0.5, view=None, globe=False,smooth_factor=1.0, save_fig='',**kwargs):
    
        x = batch
        batch_size = x.shape[0]
        num_images, nrow, ncol = 4,1,4


        if globe:
            view = (100., 0.)
            projection = ccrs.Orthographic(*view)
        else:
            projection = ccrs.PlateCarree()

        fig, axes = plt.subplots(nrow, ncol,
                                    figsize=(4 * nrow, 4 * ncol),
                                    subplot_kw={'projection': projection}
                                    )

        idx = 0
        for i in range(nrow):
            for j in range(ncol):
                if nrow == 1 and ncol == 1:
                    ax_ij = axes
                elif nrow == 1:
                    ax_ij = axes[j]
                else:
                    ax_ij = axes[i, j]

                x_i = x[idx]
                latitude = x_i[0, :, 0]
                longitude = x_i[1, 0, :]
                temperature = x_i[2]
                # temperature = np.random.rand(*x_i[2].shape)

                mesh = ax_ij.pcolormesh(longitude, latitude, temperature,
                                        transform=ccrs.PlateCarree(),
                                        cmap='plasma')
                # Add coastlines
                ax_ij.coastlines()
                # x_i = np.transpose( x[idx], (1, 2, 0))

                # ax_ij.set_xticks([])
                # ax_ij.set_yticks([])
                # ax_ij.set_axis_off()
                idx += 1
                if idx == batch_size: break
            if idx == batch_size: break
        # assert False

        plt.tight_layout()
        
        if len(save_fig):
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_fig)
            plt.clf()
            plt.close('all')
        else:
            plt.show()

def globe_plot_from_data_batch_inference(batch, threshold=0.5, view=None, globe=False,smooth_factor=1.0, save_fig='',**kwargs):

        x = batch.numpy()
        batch_size = x.shape[0]

        num_images, nrow, ncol = 4, 1, 4

        batch_size= num_images #overwriting only for inference

        if globe:
            view = (100., 0.)
            projection = ccrs.Orthographic(*view)
        else:
            projection = ccrs.PlateCarree()

        fig, axes = plt.subplots(nrow, ncol,
                                     figsize=(4 * nrow, 4 * ncol),
                                     subplot_kw={'projection': projection}
                                     )
    


        # fig, axes = plt.subplots(ncols=ncol, nrows=ceil(N/ncol), layout='constrained',
        #                         figsize=(3.5 * 4, 3.5 * ceil(N/ncol)),
        #                          subplot_kw={'projection': projection}
        #                          )

        idx = 0
        for i in range(nrow):
            for j in range(ncol):
                if nrow == 1 and ncol == 1:
                    ax_ij = axes
                elif nrow == 1:
                    ax_ij = axes[j]
                else:
                    ax_ij = axes[i, j]

                x_i = x[idx]
                latitude = x_i[0, :, 0]
                longitude = x_i[1, 0, :]
                temperature = x_i[2]

                mesh = ax_ij.pcolormesh(longitude, latitude, temperature,
                                            transform=ccrs.PlateCarree(),
                                            cmap='plasma')

                ax_ij.coastlines()

                # Add coastlines
                # ax_ij.coastlines()                                        
                # x_i = np.transpose( x[idx], (1, 2, 0))

                # ax_ij.set_xticks([])
                # ax_ij.set_yticks([])
                # ax_ij.set_axis_off()
                idx += 1
                if idx == batch_size: break
            if idx == batch_size: break

        plt.tight_layout()

        
        if len(save_fig):
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_fig)
            plt.clf()
            plt.close('all')
        else:
            plt.show()
