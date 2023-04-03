import Particle_Filter
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    for p in states:
        x, y, z = p[:3, 3]
        R = p[:3, :3]
        scale = 0.1
        ax.plot([x, x + scale * R[0, 0]], [y, y + scale * R[1, 0]], [z, z + scale * R[2, 0]], c='r')
        ax.plot([x, x + scale * R[0, 1]], [y, y + scale * R[1, 1]], [z, z + scale * R[2, 1]], c='g')
        ax.plot([x, x + scale * R[0, 2]], [y, y + scale * R[1, 2]], [z, z + scale * R[2, 2]], c='b')
    
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
    
    # Construct the skew-symmetric matrix for the rotational component
    w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    
    # Construct the 4x4 twist matrix in se(3)
    # twist_se3 = np.row_stack((np.column_stack((w_hat, v)),[0, 0, 0, 0]))
    twist_se3 = np.block([[w_hat, v[:, np.newaxis]], [np.zeros((1, 3)), 0]])
    
    return twist_se3

def se3_to_twistcrd(twist):
        # Convert from a member of little se(3) to a
        # 6x1 twist coordinate [v;w]

        omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
        
        vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
        
        twistcrd = np.vstack((vel,omega))
        
        return twistcrd

def main(CNN_data): #CNN_data should be input eventually
    # Final lists to store posterior poses
    states = []
    covariances = []
    # Need to get data containing the measurements from the CNN
    poses_CNN = CNN_data
    #poses_CNN = Particle_Filter.SE3()
    # covariance associated with the measurements
    covariance_CNN = np.eye(6)*0.25

    numParticles = 250
    
    initPose = Particle_Filter.SE3() # initialize at origin aligned with inertial ref. frame
    pf = Particle_Filter.ParticleFilterSE3(numParticles, initPose) # initialize particle filter object with initial pose

    for i in range(len(poses_CNN)):
        # this is an array of the constant velocities we are using to predict the motion of the robot
        #pf.predict(np.array([0.5,0.5,1,0.5,0,0]))

        # another set of motion model velocities for testing; should probaboly update code to have this be an input parameter in main
        # pf.predict(np.array([0.25, 0.2, 1, 0.01, 0.02, 0.03]))
        
        # random walk motion model
        random_walk = np.array([0.5,0.5,1,0.5,0,0])
        # random_walk = np.array([0.25, 0.2, 1, 0.01, 0.02, 0.03])
        
        pf.predict(random_walk)
        
        n_eff = pf.update(poses_CNN[i], covariance_CNN)
        # _ = pf.update(poses_CNN[i], covariance_CNN)
        
        # update the measured pose as if it were moving with the constant velocity model (x-axis velocity only)
        #poses_CNN.position[1] += 1
        
        if n_eff < numParticles/3:
            pf.resample()

        # pf.reinvigorate()

        state, cov = pf.mean_variance()

        states.append(state)
        covariances.append(cov)
    return states, covariances

if __name__ == '__main__':
    #states, covariances = main()
    # Define the initial pose
    T0 = np.eye(4)

    # Define the velocity in the x, y, z directions and the angular velocity
    v = np.array([0.5, 0.5, 1])
    omega = np.array([0.5, 0, 0])
    # for i in range(3):
    #     v[i] += np.random.normal(loc=0., scale=0.1) 
    #     omega[i] += np.random.normal(loc=0., scale=0.1) 
    # another set of testing control velocities
    # v = np.array([0.25, 0.2, 1])
    # omega = np.array([0.01, 0.02, 0.03])

    # Define the time interval and the number of steps
    dt = 0.1
    num_steps = 300

    # Initialize the list of poses
    poses = [T0]

    # Generate the list of poses using the constant velocity motion model
    for i in range(num_steps):
        # Compute the displacement

        ## TEMP CODE TO RANDOMIZE INPUT TO CHECK
        # v = np.array([0.5, 0.5, 1])
        # omega = np.array([0.5, 0, 0])
        # for i in range(3):
        #     v[i] += np.random.normal(loc=0., scale=0.5) 
        #     omega[i] += np.random.normal(loc=0., scale=0.5)
        ### THIS WILL BE DELETED EVENTUALLY

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
        t_noise = np.random.normal(scale=0.25, size=t.shape)
        
         # Add the noise to the original components
        R_hat = np.array([[0, -R[2, 1], R[1, 0]], [R[2, 1], 0, -R[0, 2]], [-R[1, 0], R[0, 2], 0]])
        R_noisy = R @ (np.eye(3) + np.sin(0.25) * R_hat + (1 - np.cos(0.25)) * R_hat @ R_hat)
        t_noisy = t + t_noise
        
        # Combine the noisy rotational and translational components into poses
        noisy_data[i] = np.row_stack((np.column_stack((R_noisy, t_noisy)),[0, 0, 0, 1]))
        i += 1
    
    # these are the posterior mean and variances produced by the particle filter. states are 6x1 twist coordinate vectors
    # and covariances are 6x1 variances associated with each variable. covariance one might need to change due to cross
    # covariance, but i'm not sure.
    states, covariances = main(noisy_data)
    # # for state in states:
    # #     print(state)

    viz_data = []
    for pose in states:
        state_SE3 = scipy.linalg.expm(twist_to_se3(pose))
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
    plot_errors(errors)
    
    # j = 0
    # for pose in poses:
        #printing ground truth data
        #print(poses[j].astype(int))
        #printing particle filter data
        # print(pose)
        # j += 1
