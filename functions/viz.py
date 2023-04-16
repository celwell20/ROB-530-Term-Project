import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from functions.utils import error_calc

def plot_errors(multi_errors, title):
     # Create a list of indices from 0 to the length of the errors list
    indices = range(len(multi_errors[0,:]))
    print(indices)

    # Create a plot with the indices on the x-axis and the errors on the y-axis
    plt.figure()
    ax = plt.axes()
    ax.plot(indices, multi_errors[0,:],label="Particle Filter error")
    ax.plot(indices, multi_errors[1,:],label="Fused Estimate Error")
    ax.plot(indices, multi_errors[2,:],label="Unfused Estimate Error")
    ax.legend()

    # Set the title and labels for the plot
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Error')

    # Show the plot
    plt.show()

def plot_compute_errors(viz_data, fused_CNN, unfused_CNN, gndTruth_CNN):
        all_R_error = np.zeros((3,len(viz_data)))
        all_t_error = np.zeros((3,len(viz_data)))

        for i in range(len(viz_data)):
            [_, PF_R_err, PF_t_err] = error_calc(viz_data[i],gndTruth_CNN[i])
            all_R_error[0,i] = PF_R_err
            all_t_error[0,i] = PF_t_err
            [_, fused_R_err, fused_t_err] = error_calc(fused_CNN[i],gndTruth_CNN[i])
            all_R_error[1,i] = fused_R_err
            all_t_error[1,i] = fused_t_err
            [_, unfuse_R_err, unfuse_t_err] = error_calc(unfused_CNN[i],gndTruth_CNN[i])
            all_R_error[2,i] = unfuse_R_err
            all_t_error[2,i] = unfuse_t_err

        plot_errors(all_R_error,'Rotation Error')
        plot_errors(all_t_error,'Translation Error')

        mean_R = np.mean(all_R_error,axis=1)
        mean_t = np.mean(all_t_error,axis=1)
        var_R = np.mean(all_R_error ** 2,axis=1)
        var_t = np.mean(all_t_error **2,axis=1)

        return mean_R,mean_t,var_R,var_t

def overlay_plots(states_list, label_list):
    # Create the plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Unpack the lists and plot
    for indx in range(len(states_list)):
        sts=states_list[indx]
        lbl=label_list[indx]
        position=np.array([s[:3, 3] for s in sts])
        ax.scatter(position[:, 0], position[:, 1], position[:, 2], s=1, label=lbl)
        ax.plot(position[:, 0], position[:, 1], position[:, 2], alpha=0.2)#,linewidth=1.5)

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

def plot_particles(particles, ground_truth, data, mean, weight):
    """Debugging Tool"""
    # Create the plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ground truth and the particles
    ax.scatter(ground_truth[0, 3], ground_truth[1, 3], ground_truth[2, 3], c='b', s=20, label="Ground Truth")
    ax.scatter(data[0, 3], data[1, 3], data[2, 3], c='m', s=20, label="Estimated Pose")
    #ax.scatter(mean[0,3], mean[1,3], mean[2, 3], c='y', s=20, label="Mean of Particles")
    ax.scatter(mean[0,], mean[1,], mean[2, ], c='k', s=20, label="Mean of Particles")
    for ind, particle in enumerate(particles):
        ax.scatter(particle[0, 3], particle[1, 3], particle[2, 3], c='r', s=weight[ind]*100, label="Particles")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
