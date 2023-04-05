import Particle_Filter
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation

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
    #truth_pos = np.array([s[:3, 3] for s in truth])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', s=1)
    #ax.scatter(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2], c='k', s=1)
    crdFrame_idx = 0
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

    ax.set_zlim3d(min_lim, max_lim)
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
    # twist_se3 = np.row_stack((np.column_stack((w_hat, v)),[0, 0, 0, 0]))
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

def main(CNN_data, truth): #CNN_data should be input eventually
    # Final lists to store posterior poses
    states = []
    covariances = []
    # Need to get data containing the measurements from the CNN
    poses_CNN = CNN_data.copy()
    #poses_CNN = Particle_Filter.SE3()
    # covariance associated with the measurements
    covariance_CNN = np.eye(6)*0.5

    numParticles = 200
    
    initPose = Particle_Filter.SE3() # initialize at origin aligned with inertial ref. frame
    pf = Particle_Filter.ParticleFilterSE3(numParticles, initPose) # initialize particle filter object with initial pose
    # plt_bool = True
    j = 0
    for i in range(len(poses_CNN)):
        # this is an array of the constant velocities we are using to predict the motion of the robot
        #pf.predict(np.array([0.5,0.5,1,0.5,0,0]))

        # another set of motion model velocities for testing; should probaboly update code to have this be an input parameter in main
        # pf.predict(np.array([0.25, 0.2, 1, 0.01, 0.02, 0.03]))
        
        # random walk motion model
        # random_walk = np.array([0.5,0.5,1,0.5,0,0])
        random_walk = np.array([0.25, 0.2, .75, 0.01, 0.03, 0.03])
        
        
        n_eff = pf.update(poses_CNN[i], covariance_CNN)
        # _ = pf.update(poses_CNN[i], covariance_CNN)
        
        # update the measured pose as if it were moving with the constant velocity model (x-axis velocity only)
        #poses_CNN.position[1] += 1
        
        if n_eff < numParticles/3:
            pf.resample()

        # # pf.reinvigorate()
        # if i == 64:
        #     poo = 5

        state, cov = pf.mean_variance()
        pf.predict(random_walk)

        ## PARTICLE PLOTTING CODE AND COMPARISON OF STATE ESTIMATE AND TRUTH
        # if i == 64:
            # visualize([pose.pose() for pose in pf.particles] , 'none')
            # visualize( [so3toSO3(twist_to_se3(state)), truth[i] ], 'none' )
            # error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( scipy.linalg.expm(twist_to_se3(state)) , truth[i].copy()) ) ) 
            # print(np.linalg.norm(error))
            # print(twist_to_se3(state))

        states.append(state)
        covariances.append(cov)

        # error = np.linalg.norm(state - se3_to_twistcrd( scipy.linalg.logm ( truth[i] ) ) )
        # state_SE3 = scipy.linalg.expm(twist_to_se3(state))
        # error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( np.linalg.inv(state_SE3.copy()) , truth[i].copy() ) ) )
        
        ## HISTOGRAM PLOTTING CODE
        # if np.linalg.norm(error) > 2 and j < 5:
        #     print(i)
        #     plot_particle_weights(pf.weights)
        #     # plt_bool = False
        #     j += 1
        # elif i == 20:
        #     plot_particle_weights(pf.weights)


    return states, covariances

if __name__ == '__main__':
    #states, covariances = main()
    # Define the initial pose
    T0 = np.eye(4)

    # Define the velocity in the x, y, z directions and the angular velocity
    # v = np.array([0.15, 0.5, 1])
    # omega = np.array([0.25, 0.25, 0.25])
    # for i in range(3):
    #     v[i] += np.random.normal(loc=0., scale=0.1) 
    #     omega[i] += np.random.normal(loc=0., scale=0.1) 
    # another set of testing control velocities
    # v = np.array([0.25, 0.2, 0])
    # omega = np.array([0, 0, 0.025])
    v = np.array([0.5, 0.5, 0.5])
    omega = np.array([0.5, 0, 0])

    # another set of test velocities:

    # Define the time interval and the number of steps
    dt = 0.1
    num_steps = 300

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
        #R_noise = np.random.normal(scale=0.25, size=R.shape)
        t_noise = np.random.normal(scale=0.5, size=t.shape)
        
         # Add the noise to the original components
        R_hat = np.array([[0, -R[2, 1], R[1, 0]], [R[2, 1], 0, -R[0, 2]], [-R[1, 0], R[0, 2], 0]])
        R_noisy = R @ (np.eye(3) + np.sin(0.5) * R_hat + (1 - np.cos(0.5)) * R_hat @ R_hat)
        t_noisy = t + t_noise
        
        # Combine the noisy rotational and translational components into poses
        noisy_data[i] = np.row_stack((np.column_stack((R_noisy, t_noisy)),[0, 0, 0, 1]))
        i += 1
    
    # these are the posterior mean and variances produced by the particle filter. states are 6x1 twist coordinate vectors
    # and covariances are 6x1 variances associated with each variable. covariance one might need to change due to cross
    # covariance, but i'm not sure.
    states, covariances = main(noisy_data, poses)
    # # for state in states:
    #     print(state)

    viz_data = []
    for pose in states:
        state_SE3 = so3toSO3(twist_to_se3(pose))
        viz_data.append(state_SE3)
    visualize(noisy_data, "noise")
    visualize(viz_data, "filtered")
    visualize(poses, "truth")

    i = 0
    errors = []
    for pose in poses:
        error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( np.linalg.inv(pose.copy()) , viz_data[i].copy() ) ) )
        errors.append(np.linalg.norm(error))
        i += 1
    # plot_errors(errors)
    print("PF Avg Error: ")
    print(sum(errors)/len(errors))
    
    i = 0
    errors = []
    for pose in poses:
        error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( np.linalg.inv(pose.copy()) , noisy_data[i].copy() ) ) )
        errors.append(np.linalg.norm(error))
        i += 1
    # plot_errors(errors)
    print("noise Avg Error: ")
    print(sum(errors)/len(errors))  
    # theta = np.pi+0.1  # 180 degrees in radians

    # R1 = np.array(
    #     [[1,           0,            0, 0.1],
    #     [0, np.cos(theta), -np.sin(theta), 0.5],
    #     [0, np.sin(theta),  np.cos(theta), 0.1],
    #     [0,           0,            0, 1]
    # ])
    # theta = np.pi  # 180 degrees in radians

    # R2 = np.array(
    #     [[1,           0,            0, 0.1],
    #     [0, np.cos(theta), -np.sin(theta), -0.225],
    #     [0, np.sin(theta),  np.cos(theta), 0.5],
    #     [0,           0,            0, 1]
    # ])
    # # shit = 5
    # visualize([R1,R2], 'none')
    # error = se3_to_twistcrd( scipy.linalg.logm ( np.dot( np.linalg.inv(R1) , R2 ) ) ) 
    # print(np.linalg.norm(error))
    # # j = 0
    # for pose in poses:
    #     #printing ground truth data
    #     time.sleep(0.25)
    #     print(pose.astype(int))
        #printing particle filter data
        # print(viz_data[j])
        # j += 1
