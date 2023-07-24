# import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = False


def plot_voxels_batch_new(voxels, ncols=6, threshold=0.5, save_fig=''):
    """Plots batches of point clouds
    
    Args:
        point_clouds (torch.Tensor): Shape (batch_size, num_points, 4).
        ncols (int): Number of columns in grid of images.
        threshold (float): Value above which to consider point cloud occupied.
    """
    batch_size = voxels.shape[0]
    voxel_res = voxels.shape[2]
    nrows = int((batch_size - 0.5) / ncols) + 1
    fig = plt.figure()

    print(f"THIS IS PLOTTING")

    # Permutation to get better angle of chair data
    voxels = voxels.permute(0, 1, 2, 4, 3)
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

        # Extract coordinates with feature values above threshold (corresponding
        # to occupied points)
        voxels[i, 0][voxels[i, 0]>0.5] = 1
        voxels[i, 0][voxels[i, 0]<=0.5] = 0
        # voxel[i,0] = voxel[i,0] 
        coords =  voxels[i, 0].nonzero(as_tuple=False)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=0.1, color="gray")           

        # coordinates = point_clouds[i, :, :3]
        # features = point_clouds[i, :, -1]
        # coordinates = coordinates[features > threshold]
        # ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=1)            
    
        # Set limits of plot
        ax.set_xlim(0, voxel_res)
        ax.set_ylim(0, voxel_res)
        ax.set_zlim(0, voxel_res)

    plt.tight_layout()
    
    # Optionally save figure
    if len(save_fig):
        # plt.savefig('plot.ps')
        plt.savefig(save_fig, format='png')
        plt.clf()
        plt.close()
    else:
        plt.show()