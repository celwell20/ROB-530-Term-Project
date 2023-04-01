import Particle_Filter
import numpy as np
import matplotlib as plt

def main():
    
    #
    
    states = []
    covariances = []
    # Need to get data containing the measurements from the CNN
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


if __name__ == '__main__':
    main()