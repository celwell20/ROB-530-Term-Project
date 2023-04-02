import Particle_Filter
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def visualize(states):
    positions = np.array([s[:3, 3] for s in states])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', s=1)
    for p in states:
        x, y, z = p[:3, 3]
        R = p[:3, :3]
        scale = 0.25
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
    ax.set_zlim3d(min_lim, max_lim)
    plt.show()    

def main(CNN_data): #CNN_data should be input eventually
    # Final lists to store posterior poses
    states = []
    covariances = []
    # Need to get data containing the measurements from the CNN
    poses_CNN = CNN_data
    #poses_CNN = Particle_Filter.SE3()
    # arbitrary covariance values
    covariance_CNN = np.eye(6)*0.1

    numParticles = 500

    initPose = Particle_Filter.SE3() # initialize at origin aligned with inertial ref. frame
    pf = Particle_Filter.ParticleFilterSE3(numParticles, initPose) # initialize particle filter object with initial pose

    for i in range(len(poses_CNN)):

        pf.predict(np.array([1,0,0,0,0,0]))
        n_eff = pf.update(poses_CNN[i], covariance_CNN)
        
        # update the measured pose as if it were moving with the constant velocity model (x-axis velocity only)
        #poses_CNN.position[1] += 1
        
        if n_eff < numParticles/3:
            pf.resample()

        state, cov = pf.mean_variance()

        states.append(state)
        covariances.append(cov)
    return states, covariances



def se3_to_twistcrd(twist):
        # Convert from a member of little se(3) to a
        # 6x1 twist coordinate [v;w]

        omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
        
        vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
        
        twistcrd = np.vstack((vel,omega))
        
        return twistcrd

if __name__ == '__main__':
    #states, covariances = main()
    # Define the initial pose
    T0 = np.eye(4)

    # Define the velocity in the x, y, z directions and the angular velocity
    v = np.array([0.25, 0.2, 1])
    omega = np.array([0.01, 0.02, 0.03])

    # Define the time interval and the number of steps
    dt = 0.2
    num_steps = 50

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
        R_noise = np.random.normal(scale=0.1, size=R.shape)
        t_noise = np.random.normal(scale=0.1, size=t.shape)
        
         # Add the noise to the original components
        R_hat = np.array([[0, -R[2, 1], R[1, 0]], [R[2, 1], 0, -R[0, 2]], [-R[1, 0], R[0, 2], 0]])
        R_noisy = R @ (np.eye(3) + np.sin(0.1) * R_hat + (1 - np.cos(0.1)) * R_hat @ R_hat)
        t_noisy = t + t_noise
        
        # Combine the noisy rotational and translational components into poses
        noisy_data[i] = np.row_stack((np.column_stack((R_noisy, t_noisy)),[0, 0, 0, 1]))
        i += 1
    #states, covariances = main(poses)
    visualize(noisy_data)
    # visualize(states)
