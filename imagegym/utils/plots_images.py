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

def image_plot(batch, save_fig='', **kwargs):

        images = batch.cpu().numpy()
        total_images = images.shape[0]

        num_images = 6

        ncol = 6
        nrow = 1
        
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * nrow, 4 * ncol))

        idx = 0

        for i in range(nrow):
            for j in range(ncol):
                if nrow == 1 and ncol == 1:
                    ax_ij = axes
                elif nrow == 1:
                    ax_ij = axes[j]
                else:
                    ax_ij = axes[i, j]

                image_ij = np.transpose(images[idx], (1, 2, 0))
                # image_ij = images[idx]
                # playbook_io.print_debug(f"image_ij: {image_ij.min()} {image_ij.max()}")

                ax_ij.imshow(image_ij,
                             interpolation=None)

                idx += 1
                if idx == num_images or idx == total_images: break
            if idx == num_images or idx == total_images: break

        # assert False
        for i in range(nrow):
            for j in range(ncol):
                if nrow == 1 and ncol == 1:
                    ax_ij = axes
                elif nrow == 1:
                    ax_ij = axes[j]
                else:
                    ax_ij = axes[i, j]
                ax_ij.set_xticks([])
                ax_ij.set_yticks([])
                ax_ij.set_frame_on(False)

        plt.tight_layout()

        print("if plot?",len(save_fig),"name",save_fig)
        if len(save_fig):
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_fig)
            plt.clf()
            plt.close('all')
        else:
            plt.show()