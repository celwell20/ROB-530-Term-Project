import numpy as np
import scipy
from scipy.stats import multivariate_normal

class SE3:
    def __init__(self, position=np.zeros(3), rotation=np.eye(3)):
        # Construct an SE(3) object based on a provided rotation matrix and position vector.
        self.position = position
        self.rotation = rotation

    def pose(self):
        # Return the pose
        pose = [np.row_stack(np.column_stack(self.rotation, self.position.T),[0, 0, 0, 1])]
        return pose
    
    def se3_to_twistcrd(self):
        # go from little se(3) to twist coordinates (6x1)
        # call the functionthat is elsewhere
        twist = self.pose()
        # Convert from a member of little se(3) to a
        # 6x1 twist coordinate [v;w]
        omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
        
        vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
        
        twistcrd = np.vstack((vel,omega))

        return twistcrd

    
class ParticleFilterSE3:
    def __init__(self, num_particles, initial_pose):

        self.num_particles = num_particles
        self.particles = [SE3(position=initial_pose.position, rotation=initial_pose.rotation) for _ in range(num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input):

        for i in range(self.num_particles):

            particle = self.particles[i]
            particle.position += control_input[:3]
            particle.rotation = particle.rotation @ self.rotation_matrix(control_input[3:])
    
    def update(self, measurement, covariance):
        # Loop through all the particles and compute their weigths based on their vicinity to the measurment
        for i in range(self.num_particles):
            particle = self.particles[i]
            # log map of dot product between particle's pose and the inverse of the measurement pose from CNN

            # write method to convert from 
            error = self.se3_to_twistcrd(scipy.linalg.logm(np.dot(np.linalg.inv(particle.pose()),measurement.pose())))
            self.weights[i] *= multivariate_normal.pdf(error, mean = np.zeros(6,1), cov = covariance)

        # Normalize the weights and compute the effective sample size
        self.weights /= np.sum(self.weights)
        n_eff = 1 / np.sum(self.weights**2)

        return n_eff
    
    def resample(self):
        # Initiate the resampled set
        new_particles = np.zeros_like(self.particles)
        new_weights = np.zeros_like(self.weights)
        
        ## ChatGPT Resampling Algorithm:
        ## It seems like the algorithm should resample based on the weights,
        ## so if we can't implement a resampling algorithm of our own, then 
        ## this one should work

        # Sample particles with replacement based on their weights
        # indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=self.weights)
        # for i in indices:
        #     new_particles.append(self.particles[i])
        # self.particles = new_particles
        # self.weights = np.ones(self.num_particles) / self.num_particles

        
        W = np.cumsum(self.weights)
        r = np.random.rand(1) / self.num_particles
        count = 0
        for j in range(self.num_particles):
            u = r + j/self.num_particles
            while u > W[count]:
                count += 1
            new_particles[j] = self.particles[count]
            new_weights[j] = 1 / self.num_particles
        self.particles = new_particles
        self.weights = new_weights

    def mean_variance(self):
        # X = np.mean(self.particles, axis=1)
        # sinSum3 = 0
        # cosSum3 = 0
        # sinSum4 = 0
        # cosSum4 = 0
        # sinSum5 = 0
        # cosSum5 = 0
        
        # for s in range(self.n):

        #     se3_particle = np.linalg.logm(self.particles[i])
        #     screw

        #     cosSum3 += np.cos(self.particles[3,s])
        #     sinSum3 += np.sin(self.particles[3,s])
        #     cosSum4 += np.cos(self.particles[4,s])
        #     sinSum4 += np.sin(self.particles[4,s])
        #     cosSum5 += np.cos(self.particles[5,s])
        #     sinSum5 += np.sin(self.particles[5,s])
        # X[3] = np.arctan2(sinSum3, cosSum3)
        # X[4] = np.arctan2(sinSum4, cosSum4)
        # X[5] = np.arctan2(sinSum5, cosSum5)
        
        # zero_mean = np.zeros_like(self.particles)
        # for s in range(self.n):
        #     zero_mean[:,s] = self.particles[:,s] - X
        #     zero_mean[3,s] = np.unwrap(zero_mean[3,s])
        #     zero_mean[4,s] = np.unwrap(zero_mean[4,s])
        #     zero_mean[5,s] = np.unwrap(zero_mean[5,s])
        # P = zero_mean @ zero_mean.T / self.num_particles

        # state_mean = np.mean(self.particles[:].position, axis=0)
        
        # # compute the variance of the particles
        # state_cov = np.zeros((6, 6))
        # for p in self.particles:
        #     diff = p.position - state_mean
        #     state_cov += np.square(diff).sum(axis=0)
        # state_cov /= len(self.particles)
        
        # geodesic mean --> consider 
        return state_mean, state_cov
        
    @staticmethod
    def rotation_matrix(rotation):
        # Compute the rotation 
        roll, pitch, yaw = rotation
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)
        rotation_matrix = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return rotation_matrix
    
    @staticmethod
    def se3_to_twistcrd(twist):
        # Convert from a member of little se(3) to a
        # 6x1 twist coordinate [v;w]

        omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
        
        vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
        
        twistcrd = np.vstack((vel,omega))

        return twistcrd