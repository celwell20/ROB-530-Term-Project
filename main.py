import Particle_Filter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(states):
    positions = np.array([s[:3, 3] for s in states])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r')
    for p in states:
        x, y, z = p[:3, 3]
        R = p[:3, :3]
        scale = 0.2
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

def main(): #CNN_data should be input eventually
    
    #
    
    states = []
    covariances = []
    # Need to get data containing the measurements from the CNN
    # poses_CNN = CNN_data
    poses_CNN = Particle_Filter.SE3()
    # arbitrary covariance values
    covariance_CNN = np.eye(6)*0.1

    numParticles = 1000

    initPose = Particle_Filter.SE3() # initialize at origin aligned with inertial ref. frame
    pf = Particle_Filter.ParticleFilterSE3(numParticles, initPose) # initialize particle filter object with initial pose

    for i in range(len(poses_CNN)):

        pf.predict(np.array([1,0,0,0,0,0]))
        n_eff = pf.update(poses_CNN, covariance_CNN)
        
        # update the measured pose as if it were moving with the constant velocity model (x-axis velocity only)
        poses_CNN.position[1] += 1
        
        if n_eff < numParticles/3:
            pf.resample()

        state, cov = pf.mean_variance()

        states.append(state)
        covariances.append(cov)
    return states, covariances

if __name__ == '__main__':
    #states, covariances = main()
    states = [np.array([[1, 0, 0, 2],
                       [0, 1, 0, 3],
                       [0, 0, 1, 4],
                       [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 5],
                       [0, 1, 0, 6],
                       [-1, 0, 0, 7],
                       [0, 0, 0, 1]])]
    covariances = [0.1*np.eye(6) , 0.1*np.eye(6)]
    visualize(states)
