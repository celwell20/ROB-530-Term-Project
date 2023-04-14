import scipy

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

import functions.Particle_Filter as Particle_Filter
import functions.read as read
from functions.utils import twist_to_se3, se3_to_twistcrd, se3toSE3, error_calc
from functions.viz import plot_mitl_errors, plot_different_errors, plot_errors, overlay_plots2, overlay_plots3, plot_particle_weights, visualize

def main(CNN_data, update_cov, init_pose, control=[np.zeros((1,6))]):
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

        # Read the current control input and use it to perform the prediction step
        current_input = np.array([0.0 , 0.0 , 0.0 , 0.0, 0.0, 0.0])#control[i]
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
    # Initialise the pose to the origin
    poses = [np.eye(4)]
   
    # Define the dt interval and the number of steps
    dt = 0.15
    num_steps = 150


    v = np.array([0.5, 0.5, 0.5])
    omega = np.array([0, 0, 0.5])

    # Generate the list of poses using the constant velocity motion model
    for i in range(num_steps):
        # Compute the displacement

        dT = np.eye(4)
        dT[:3, 3] = v * dt
        r = Rotation.from_rotvec(omega * dt)
        dT[:3, :3] = r.as_matrix()

        # Update the pose
        T = np.dot(poses[-1], dT)
        poses.append(T)

    # add some noise to the test data for use in the particle filter
    noisy_data = poses.copy()
    i = 0
    for pose in noisy_data:
          # Extract the rotational and translational components of the poses
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Generate white noise for the rotational and translational components
        # Adjst the scale term to increase the noise of the measurements
        t_noise = np.random.normal(scale=0.5, size=t.shape)
        
         # Add the noise to the original components
        R_hat = np.array([[0, -R[2, 1], R[1, 0]], [R[2, 1], 0, -R[0, 2]], [-R[1, 0], R[0, 2], 0]])
        R_noisy = R @ (np.eye(3) + np.sin(0.5) * R_hat + (1 - np.cos(0.5)) * R_hat @ R_hat)
        t_noisy = t + t_noise
        
        # Combine the noisy rotational and translational components into poses
        noisy_data[i] = np.row_stack((np.column_stack((R_noisy, t_noisy)),[0, 0, 0, 1]))
        i += 1
    
    parking_measurements = read.readThis('output.txt')
    noise_park = []
    for pose in parking_measurements:
          # Extract the rotational and translational components of the poses
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Generate white noise for the rotational and translational components
        # Adjst the scale term to increase the noise of the measurements
        # Change the translation noise here "scale = noise"
        t_noise = np.random.normal(scale=3, size=t.shape)
        
         # Add the noise to the original components
        R_hat = np.array([[0, -R[2, 1], R[1, 0]], [R[2, 1], 0, -R[0, 2]], [-R[1, 0], R[0, 2], 0]])

        # Change noise values here in np.sin(noise) and np.cos(noise)
        R_noisy = R @ (np.eye(3) + np.sin(3) * R_hat + (1 - np.cos(3)) * R_hat @ R_hat)
        t_noisy = t + t_noise
        
        # Combine the noisy rotational and translational components into poses
        noise_park.append(np.row_stack((np.column_stack((R_noisy, t_noisy)),[0, 0, 0, 1])))
        i += 1

    # 
    # gndTruth_CNN = np.load(file='data/trajectory_gt.npy')
    # unfused_CNN = np.load(file='data/trajectory_unfused.npy')
    # fused_CNN = np.load(file='data/trajectory_fused.npy')
    gndTruth_CNN = np.load(file='data/trajectory_gt_improved.npy')
    unfused_CNN = np.load(file='data/trajectory_unfused_improved.npy')
    fused_CNN = np.load(file='data/trajectory_fused_improved.npy')
    
    init_pose = unfused_CNN[0]
    states, covariances = main(fused_CNN, np.eye(6)*0.5, init_pose)
    # these are the posterior mean and variances produced by the particle filter. states are 6x1 twist coordinate vectors
    # and covariances are 6x1 variances associated with each variable. covariance one might need to change due to cross
    # covariance, but i'm not sure.
    viz_data = []
    #viz_data[0] 4x4
    #for pose_4_4 in viz_data:
    for pose in states:
        state_SE3 = se3toSE3(twist_to_se3(pose))
        viz_data.append(state_SE3)

    #error for PF, fused, unfused
    [mean_R,mean_t,var_R,var_t] = plot_different_errors(viz_data=viz_data,fused_CNN=fused_CNN,unfused_CNN=unfused_CNN,gndTruth_CNN=gndTruth_CNN)
    print("mean of error(Rotation)")
    print("PF: ", mean_R[0])
    print("fused: ",mean_R[1])
    print("unfused: ",mean_R[2])
    print("variance of error(Roatation)")
    print("PF: ", var_R[0])
    print("fused: ",var_R[1])
    print("unfused: ",var_R[2])

    print("mean of error(tanslation)")
    print("PF: ", mean_t[0])
    print("fused: ",mean_t[1])
    print("unfused: ",mean_t[2])
    print("variance of error(tanslation)")
    print("PF: ", var_t[0])
    print("fused: ",var_t[1])
    print("unfused: ",var_t[2])
    

    
    

    # print("gndtrue",gndTruth_CNN[0])
    # print("unfused",unfused_CNN[0])
    asdf = 5
    ## NOTE: string optiosn for visualize() include: "filtered", "fused", "unfused", "truth", "noise"
    ## specififying an arbitrary string will produce no title for the plot
    overlay_plots3(unfused_CNN,gndTruth_CNN,viz_data, "unfused", "gnd trth", "PF")
    overlay_plots3(unfused_CNN,gndTruth_CNN,fused_CNN, "unfused", "gnd trth", "fused_CNN")
    # visualize(viz_data, "filtered")
    # visualize(fused_CNN, "fused")
    # visualize(unfused_CNN, "unfused")
    # visualize(gndTruth_CNN, "truth")

    # i = 0
    # errors = []
    # t_errs = []
    # R_errs = []
    # for pose in poses:
    #     error,R_err, t_err = error_calc(pose.copy(), viz_data[i].copy())
    #     errors.append(np.linalg.norm(error))
    #     t_errs.append(np.linalg.norm(t_err))
    #     R_errs.append(np.linalg.norm(R_err))
    #     i += 1
    # plot_errors(errors)
    # print("PF Avg Error: " + sum(errors)/len(errors))
    # print(" ")
    
    # print("PF R Error: " + sum(R_errs)/len(R_errs))
    # print(" ")
    
    
    # print("PF t Error: " + sum(t_errs)/len(t_errs))
    # print(" ")

    # i = 0
    # errors = []
    # t_errs = []
    # R_errs = []
    # for pose in poses:
    #     error,R_err, t_err = error_calc(pose.copy(), noisy_data[i].copy())
    #     # error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( np.linalg.inv(pose.copy()) , noisy_data[i].copy() ) ) )
    #     errors.append(np.linalg.norm(error))
    #     t_errs.append(np.linalg.norm(t_err))
    #     R_errs.append(np.linalg.norm(R_err))
    #     i += 1
    # plot_errors(errors)
    # print("PF Avg Error: " + sum(errors)/len(errors))
    # print(" ")
    
    # print("PF R Error: " + sum(R_errs)/len(R_errs))
    # print(" ")
    
    
    # print("PF t Error: " + sum(t_errs)/len(t_errs))
    # print(" ")
