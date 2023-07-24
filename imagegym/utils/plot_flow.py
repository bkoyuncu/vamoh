import matplotlib
matplotlib.use('Agg')  # This is hacky (useful for running on VMs)
import matplotlib.pyplot as plt


def save_flow_grid(xx,yy,prob,save_fig='')->None:
        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.gca().set_aspect('equal', 'box')
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')