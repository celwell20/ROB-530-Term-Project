import scipy

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def visualize(states, title_string):
    # states estimated by the particle filters
    # true data without any noise from "measurements"
    positions = np.array([s[:3, 3] for s in states])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot the points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', s=1)

    # if title_string == "truth" or title_string == "filtered":
    # add lines to connect the points ; change alpha param to change color darkness
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c='r', alpha=0.2)    

    #ax.scatter(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2], c='k', s=1)
    # crdFrame_idx = 0
    # for p in states:

    #     if crdFrame_idx % 5 == 0:
    #         x, y, z = p[:3, 3]
    #         R = p[:3, :3]
    #         scale = 0.5
    #         ax.plot([x, x + scale * R[0, 0]], [y, y + scale * R[1, 0]], [z, z + scale * R[2, 0]], c='r')
    #         ax.plot([x, x + scale * R[0, 1]], [y, y + scale * R[1, 1]], [z, z + scale * R[2, 1]], c='g')
    #         ax.plot([x, x + scale * R[0, 2]], [y, y + scale * R[1, 2]], [z, z + scale * R[2, 2]], c='b')
        
    #     crdFrame_idx += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    min_lim = np.min(positions) - 1
    max_lim = np.max(positions) + 1
    ax.set_xlim3d(min_lim, max_lim)
    ax.set_ylim3d(min_lim, max_lim)

    if title_string == "truth":
        plt.title("Ground Truth")
    elif title_string == "filtered":
        plt.title("Particle Filter Estimate")
    elif title_string == "noise":
        plt.title("Noisy Data")
    elif title_string == "fused":
        plt.title("Fused CNN Data")
    elif title_string == "unfused":
        plt.title("Unfused CNN Data")

    ax.set_zlim3d(min_lim, max_lim)
    plt.show()    

def plot_mitl_errors(multi_errors,title):
     # Create a list of indices from 0 to the length of the errors list
    indices = range(len(multi_errors[0,:]))
    print(indices)

    # Create a plot with the indices on the x-axis and the errors on the y-axis
    plt.figure()
    ax = plt.axes()
    ax.plot(indices, multi_errors[0,:],label="PF error")
    ax.plot(indices, multi_errors[1,:],label="fused error")
    ax.plot(indices, multi_errors[2,:],label="unfused error")
    ax.legend()
    # Set the title and labels for the plot
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Error')

    # Show the plot
    plt.show()

#visualize results between PF and fuse
def plot_different_errors(viz_data,fused_CNN,unfused_CNN,gndTruth_CNN):
        all_R_error = np.zeros((3,len(viz_data)))
        all_t_error = np.zeros((3,len(viz_data)))
        for i in range(len(viz_data)):
            [PF_error, PF_R_err, PF_t_err] = error_calc(viz_data[i],gndTruth_CNN[i])
            all_R_error[0,i] = PF_R_err
            all_t_error[0,i] = PF_t_err
            [fused_error, fused_R_err, fused_t_err] = error_calc(fused_CNN[i],gndTruth_CNN[i])
            all_R_error[1,i] = fused_R_err
            all_t_error[1,i] = fused_t_err
            [unfuse_error, unfuse_R_err, unfuse_t_err] = error_calc(unfused_CNN[i],gndTruth_CNN[i])
            all_R_error[2,i] = unfuse_R_err
            all_t_error[2,i] = unfuse_t_err

        plot_mitl_errors(all_R_error,'rotation error')
        plot_mitl_errors(all_t_error,'translation error')
        mean_R = np.mean(all_R_error,axis=1)
        mean_t = np.mean(all_t_error,axis=1)
        var_R = np.mean(all_R_error ** 2,axis=1)
        var_t = np.mean(all_t_error **2,axis=1)
        return mean_R,mean_t,var_R,var_t
        

def plot_errors(errors):
    """
    Plots a list of errors versus its indices.

    Args:
    errors (list): a list of errors to be plotted

    Returns:
    None
    """
    # Create a list of indices from 0 to the length of the errors list
    indices = range(len(errors))

    
    # Create a plot with the indices on the x-axis and the errors on the y-axis
    plt.plot(indices, errors)

    # Set the title and labels for the plot
    plt.title('Errors vs Indices')
    plt.xlabel('Index')
    plt.ylabel('Error')

    # Show the plot
    plt.show()

def overlay_plots2(states1, states2, label1, label2):
    # states estimated by the particle filters
    # true data without any noise from "measurements"
    positions1 = np.array([s[:3, 3] for s in states1])
    positions2 = np.array([s[:3, 3] for s in states2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot the points
    ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], c='r', s=1, label=label1)

    ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], c='r', alpha=0.2)

    ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], c='b', s=1, label=label2)
    
    ax.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], c='b', alpha=0.2)       

    #ax.scatter(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2], c='k', s=1)
    # crdFrame_idx = 0
    # for p in states:

    #     if crdFrame_idx % 5 == 0:
    #         x, y, z = p[:3, 3]
    #         R = p[:3, :3]
    #         scale = 0.5
    #         ax.plot([x, x + scale * R[0, 0]], [y, y + scale * R[1, 0]], [z, z + scale * R[2, 0]], c='r')
    #         ax.plot([x, x + scale * R[0, 1]], [y, y + scale * R[1, 1]], [z, z + scale * R[2, 1]], c='g')
    #         ax.plot([x, x + scale * R[0, 2]], [y, y + scale * R[1, 2]], [z, z + scale * R[2, 2]], c='b')
        
    #     crdFrame_idx += 1
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()    

def overlay_plots3(states1, states2, states3, label1, label2, label3):
    # states estimated by the particle filters
    # true data without any noise from "measurements"
    positions1 = np.array([s[:3, 3] for s in states1])
    positions2 = np.array([s[:3, 3] for s in states2])
    positions3 = np.array([s[:3, 3] for s in states3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot the points
    ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], c='m', s=1, label=label1)

    ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], c='m', alpha=0.2)

    ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], c='b', s=1, label=label2)
    
    ax.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], c='b', alpha=0.2)       
    
    ax.scatter(positions3[:, 0], positions3[:, 1], positions3[:, 2], c='g', s=1, label=label3)
    
    ax.plot(positions3[:, 0], positions3[:, 1], positions3[:, 2], c='g', alpha=0.2) 

    #ax.scatter(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2], c='k', s=1)
    # crdFrame_idx = 0
    # for p in states:

    #     if crdFrame_idx % 5 == 0:
    #         x, y, z = p[:3, 3]
    #         R = p[:3, :3]
    #         scale = 0.5
    #         ax.plot([x, x + scale * R[0, 0]], [y, y + scale * R[1, 0]], [z, z + scale * R[2, 0]], c='r')
    #         ax.plot([x, x + scale * R[0, 1]], [y, y + scale * R[1, 1]], [z, z + scale * R[2, 1]], c='g')
    #         ax.plot([x, x + scale * R[0, 2]], [y, y + scale * R[1, 2]], [z, z + scale * R[2, 2]], c='b')
        
    #     crdFrame_idx += 1
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()    


def plot_particle_weights(particle_weights):
    """DEBUGGING TOOL: Plot the weights of the particles"""

    fig, ax = plt.subplots()
    ax.hist(particle_weights, bins=20, density=True, alpha=0.5, color='b')
    ax.set_xlabel('Particle Weights')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Particle Weights')
    plt.show()
