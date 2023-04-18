import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

import functions.Particle_Filter as Particle_Filter
from functions.utils import twist_to_se3, se3toSE3, rot_vec
from functions.viz import plot_compute_errors, overlay_plots

def main(CNN_data, update_cov, init_pose, control=0, gndTruth_CNN=0):
    """Run the Particle Filter through a series of estimates, given the control inputs"""

    # Final lists to store posterior poses
    means = []
    covariances = []

    # Store the input data
    poses = CNN_data.copy()
    covariance = update_cov.copy()

    # Set the number of particles
    num_particles = 200
    
    # Create an initial SE3 pose from the provided init_pose, and initialise the particle filter with it
    initialiser = Particle_Filter.SE3(init_pose[:3,3], init_pose[:3,:3])
    pf = Particle_Filter.ParticleFilterSE3(num_particles, initialiser)
    
    # Loop through the data from the Convolutional Neural Network
    for i in range(len(poses)):
        print("Current Iteration: "+str(i))

        # DEBUGGING #
        # particles=pf.return_particles()
        # weigths=pf.return_weigths()
        # if i>0:
        #     plot_particles(particles, gndTruth_CNN[i], fused_CNN[i], est_mean, weigths)

        # Read the current control input and use it to perform the prediction step
        current_input =control[i] 
        pf.predict(current_input)

        # Propagate the prediction with the particle filter
        n_eff = pf.update(poses[i], covariance)
        
        # Check if re-sampling is needed
        if n_eff < num_particles/3:
            pf.resample()

        # Compute the mean and covariance of the 
        est_mean, est_cov = pf.mean_variance()
        means.append(est_mean)
        covariances.append(est_cov)

    return means, covariances

if __name__ == '__main__':
    # Load the data
    gndTruth_CNN = np.load(file='data/trajectory_gt_default.npy')
    unfused_CNN = np.load(file='data/trajectory_unfused_default.npy')
    fused_CNN = np.load(file='data/trajectory_fused_default.npy')
    
    # Initialise the pose to the start point of the trajectory
    init_pose = fused_CNN[0]

    # Extract the control inputs from the ground truth data
    control=[np.array([0.0 , 0.0 , 0.0 , 0.0, 0.0, 0.0])]
    for i in range(1, len(gndTruth_CNN)):
        prev_pose=gndTruth_CNN[i-1]
        current_pose=gndTruth_CNN[i]

        delta_pos=current_pose[:3,3]-prev_pose[:3,3]
        delta_rot=rot_vec(np.transpose(prev_pose[:3,:3])@current_pose[:3,:3])

        new_input=np.hstack((delta_pos.reshape((1,3)),delta_rot.reshape((1,3))))
        control.append(new_input[0])

    # Run the particle filter and produce the 6x1 variances associated with each variable
    pf_cov=np.diag([1, 1, 1, 1, 1, 1])
    states, covariances = main(fused_CNN, pf_cov, init_pose, control, gndTruth_CNN)

    # Extract the poses for plotting
    viz_data = []
    for pose in states:
        state_SE3 = se3toSE3(twist_to_se3(pose))
        #state_SE3=pose
        viz_data.append(state_SE3)
    np.save("PF_Data_X.npy", viz_data)

    # Compute and plot the different errors
    mean_R, mean_t, var_R, var_t = plot_compute_errors(viz_data=viz_data,fused_CNN=fused_CNN,unfused_CNN=unfused_CNN,gndTruth_CNN=gndTruth_CNN)
    
    # Report the errors in the terminal
    print("Mean of error(Rotation)")
    print("Particle filter: ", mean_R[0])
    print("Original Fused Estimate: ",mean_R[1])
    print("Original Unfused Unfused: ",mean_R[2])

    print("Variance of error(Roatation)")
    print("Particle filter: ", var_R[0])
    print("Original Fused Estimate: ",var_R[1])
    print("Original Unfused Unfused: ",var_R[2])

    print("Mean of error(tanslation)")
    print("Particle filter: ", mean_t[0])
    print("Original Fused Estimate: ",mean_t[1])
    print("Original Unfused Unfused: ",mean_t[2])

    print("Variance of error(tanslation)")
    print("Particle filter: ", var_t[0])
    print("Original Fused Estimate: ",var_t[1])
    print("Original Unfused Unfused: ",var_t[2])

    # Visualise the estimates
    states_list=[gndTruth_CNN, unfused_CNN, fused_CNN, viz_data]
    label_list=["Ground Truth", "Unfused Estimate", "Fused Estimate", "Particle Filter"]
    overlay_plots(states_list, label_list)
