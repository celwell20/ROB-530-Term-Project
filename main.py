import Particle_Filter
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation
import read

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
    ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], c='r', s=1, label=label1)

    ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], c='r', alpha=0.2)

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

def twist_to_se3(twist):
    """
    Convert a 6x1 twist coordinate to a 4x4 twist in the Lie algebra se(3).
    
    Parameters:
    - twist: a numpy array of shape (6, 1) representing the twist coordinate
    
    Returns:
    - twist_se3: a numpy array of shape (4, 4) representing the twist in the Lie algebra se(3)
    """
    
    # Extract the translational and rotational components of the twist
    v = twist[:3]
    w = twist[3:]
    v = np.array( [ [v[0]] , [v[1]] , [v[2]] ] )
    # Construct the skew-symmetric matrix for the rotational component
    w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    
    # Construct the 4x4 twist matrix in se(3)
    twist_se3 = np.row_stack((np.column_stack((w_hat, v.reshape(3,))),[0, 0, 0, 0]))
    
    return twist_se3

def se3_to_twistcrd(twist):
        # Convert from a member of little se(3) to a
        # 6x1 twist coordinate [v;w]

        omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
        
        vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
        
        twistcrd = np.vstack((vel,omega))
        
        return twistcrd

def so3toSO3(twist):
    # converts an element of so(3) to an element of SO(3)
    # REVISE TITLE BECAUES THIS IS ACTUALLY se(3)  TO SE(3)
    # confirm if this is valid
    R = scipy.linalg.expm(twist[:3,:3])
    t = twist[:3,3]
    SE3 = np.row_stack((np.column_stack((R, t.reshape(3,).copy())),[0, 0, 0, 1]))
    return SE3

def plot_particle_weights(particle_weights):
    fig, ax = plt.subplots()
    ax.hist(particle_weights, bins=20, density=True, alpha=0.5, color='b')
    ax.set_xlabel('Particle Weights')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Particle Weights')
    plt.show()

def error_calc(state, truth):
    # calculate the error for a given state from the truth
    # chordal distance
    # R^T*R - I(3x3)
    pre_err = np.dot( np.transpose(truth[:3,:3]), state[:3,:3] ) - np.eye(3)
    R_err = np.linalg.norm(pre_err,  ord='fro')
    t_err = truth[:3,3] - state[:3,3]

    # COULD REVISE THIS TO HAVE THE NORM OF THE TRANSLATION ERROR OCCUR HERE

    error = np.vstack((R_err,t_err.reshape(3,1)))
    return error, R_err, t_err

def main(CNN_data, CNN_covariances, truth): #CNN_data should be input eventually
    # Final lists to store posterior poses
    states = []
    covariances = []
    # Need to get data containing the measurements from the CNN
    poses_CNN = CNN_data.copy()
    # covariance associated with the measurements
    covariance_CNN = CNN_covariances.copy()

    numParticles = 200
    
    initPose = Particle_Filter.SE3() # initialize at origin aligned with inertial ref. frame
    pf = Particle_Filter.ParticleFilterSE3(numParticles, initPose) # initialize particle filter object with initial pose
    # plt_bool = True
    
    for i in range(int(len(poses_CNN)/20)):
        # this is an array of the velocities we are using to predict the motion of the robot
        random_walk = np.array([0.0 , 0.0 , 0.0 , 0.0, 0.0, 0.0])
        pf.predict(random_walk)
        n_eff = pf.update(poses_CNN[i], covariance_CNN)
        # _ = pf.update(poses_CNN[i], covariance_CNN)
        
        if n_eff < numParticles/3:
            pf.resample()

        # if i == 64:
        #     brkpoint = 0 ## breakpoint debugging code 

        state, cov = pf.mean_variance()
        
        ## PARTICLE PLOTTING CODE AND COMPARISON OF STATE ESTIMATE AND TRUTH FOR VERIFICATION AND DEBUGGING
        # if i == 64:
            # visualize([pose.pose() for pose in pf.particles] , 'none')
            # visualize( [so3toSO3(twist_to_se3(state)), truth[i] ], 'none' )
            # error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( scipy.linalg.expm(twist_to_se3(state)) , truth[i].copy()) ) ) 
            # print(np.linalg.norm(error))
            # print(twist_to_se3(state))
        ###

        states.append(state)
        covariances.append(cov)

    return states, covariances

if __name__ == '__main__':
    #states, covariances = main()
    # Define the initial pose
    T0 = np.eye(4)
   
    v = np.array([0.5, 0.5, 0.5])
    omega = np.array([0, 0, 0.5])

    # Define the time interval and the number of steps
    dt = 0.15
    num_steps = 150

    # Initialize the list of poses
    poses = [T0]

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

    # these are the posterior mean and variances produced by the particle filter. states are 6x1 twist coordinate vectors
    # and covariances are 6x1 variances associated with each variable. covariance one might need to change due to cross
    # covariance, but i'm not sure.
    gndTruth_CNN = np.load(file='trajectory_gt.npy')
    unfused_CNN = np.load(file='trajectory_unfused.npy')
    fused_CNN = np.load(file='trajectory_fused.npy')

    # states, covariances = main(unfused_CNN, np.eye(6)*0.05, poses)

    # viz_data = []
    # for pose in states:
    #     state_SE3 = so3toSO3(twist_to_se3(pose))
    #     viz_data.append(state_SE3)


    ## NOTE: string optiosn for visualize() include: "filtered", "fused", "unfused", "truth", "noise"
    ## specififying an arbitrary string will produce no title for the plot
    overlay_plots3(unfused_CNN,gndTruth_CNN,fused_CNN, "unfused", "gnd trth", "fused")
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
