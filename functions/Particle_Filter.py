import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import mat2quat, qmult, qinverse
from functions.utils import se3_to_twistcrd, rotation_matrix, wedge
from scipy.linalg import logm, expm

class SE3:
    def __init__(self, position=np.zeros(3), rotation=np.eye(3)):
        # Construct an SE(3) object based on a provided rotation matrix and position vector.
        
        self.position = position
        self.rotation = rotation

    def pose(self):
        # Return the pose
        pose = np.row_stack((np.column_stack((self.rotation, self.position.T)),[0, 0, 0, 1]))

        return pose
    
class ParticleFilterSE3:
    def __init__(self, num_particles, initial_pose):

        self.num_particles = num_particles
        self.particles = self.init_particles(initial_pose)
        self.weights = np.ones(num_particles) / num_particles

    def init_particles(self, initPose):
        
        init_particles = []

        # Generate a uniform distribtion of particles by adding a uniformly sampled control input to a particle located at the origin
        for _ in range(self.num_particles):
            
            # Generate the control input
            cntr_input = np.hstack((np.random.uniform(-0.25, 0.25, size=(3,)),np.random.uniform(-np.pi/4, np.pi/4, size=(3,))))

            # Copute the new position and rotation
            new_pos=initPose.position+cntr_input[:3]
            new_rot=initPose.rotation@rotation_matrix(cntr_input[3:])
            
            # Apply the transformation
            init_particles.append(SE3(position=new_pos, rotation=new_rot))

        return init_particles

    def predict(self, control_input):
        """Use the known control input to propagate the particles forward"""

        # Initialise variables
        new_particles = []

        # Loop through the particles and apply the control input to each
        cntr_input = control_input.copy()
        for pose in self.particles:

            for i in range(6):
                # Assume the motion model is not perfect, and add some noise to the particles
                cntr_input[i] += np.random.normal(0, abs(cntr_input[i])*0.05) 

            # Copute the new position and rotation
            new_pos=pose.position+cntr_input[:3]
            new_rot=pose.rotation@rotation_matrix(cntr_input[3:])
            
            # Apply the transformation
            new_particles.append(SE3(position=new_pos, rotation=new_rot))
        
        self.particles = new_particles.copy()
    
    def update(self, measurement, covariance):
        # Loop through all the particles and compute their weigths based on their vicinity to the measurment
        for i in range(self.num_particles):
            particle = self.particles[i]

            particle_vect= se3_to_twistcrd(scipy.linalg.logm(particle.pose()))
            measurment_vect=se3_to_twistcrd(scipy.linalg.logm(measurement))

            self.weights[i] *= multivariate_normal.pdf( measurment_vect.ravel(), mean = particle_vect.ravel(), cov = covariance)

        # Normalize the weights and compute the effective sample size
        self.weights /= np.sum(self.weights)

        n_eff = 1 / np.sum(self.weights**2)

        return n_eff
        
    def resample(self):
        # Initiate the resampled set
        new_particles = []
        new_weights = np.zeros_like(self.weights)
        
        W = np.cumsum(self.weights)
        r = np.random.rand(1) / self.num_particles
        count = 0
        for j in range(self.num_particles):
            u = r + j/self.num_particles
            while u > W[count]:
                count += 1

            # check to make sure there is not a referencing issue here
            new_particles.append(self.particles[count])
            new_weights[j] = 1 / self.num_particles

        self.particles = new_particles.copy()
        self.weights = new_weights.copy()

    def mean_variance(self):

        #preallocate twist coord array
        twists = np.zeros((6,self.num_particles))

        # populate twist coord array
        for i in range(self.num_particles):
            # calculate the 6x1 twist coordinate value and add it to the twist coordinate array
            omega = scipy.linalg.logm(self.particles[i].rotation)
            v = self.particles[i].position
            
            v = np.array([  [v[0]] , [v[1]] , [v[2]]  ])
            omega = np.array( [ [omega[2,1]] , [omega[0,2]] , [omega[1,0]] ] )

            twists[:,i] = np.vstack(( v, omega )).reshape(6,)

        # calculate the mean of all the 6x1 twist coordiantes
        X = np.mean(twists, axis=1)

        #initiating values
        sinSum3 = 0
        cosSum3 = 0
        sinSum4 = 0
        cosSum4 = 0
        sinSum5 = 0
        cosSum5 = 0
        
        for s in range(self.num_particles):
            # could alternatively index from the twists array
            twist = twists[:,s].copy()

            # idk what this stuff does but maani does it
            cosSum3 += np.cos(twist[3])
            sinSum3 += np.sin(twist[3])
            cosSum4 += np.cos(twist[4])
            sinSum4 += np.sin(twist[4])
            cosSum5 += np.cos(twist[5])
            sinSum5 += np.sin(twist[5])

        # same here it's just stuff from maani's code but expanded to SE(3)
        X[3] = np.arctan2(sinSum3, cosSum3)
        X[4] = np.arctan2(sinSum4, cosSum4)
        X[5] = np.arctan2(sinSum5, cosSum5)
        
        # preallocating array for covariance calculation
        zero_mean = np.zeros((6,self.num_particles))

        # morer random stuff taken from maani's code
        for s in range(self.num_particles):

            zero_mean[:,s] = twists[:,s] - X
            zero_mean[3,s] = self.wrapToPi(zero_mean[3,s])
            zero_mean[4,s] = self.wrapToPi(zero_mean[4,s])
            zero_mean[5,s] = self.wrapToPi(zero_mean[5,s])
        
        state_cov = zero_mean @ zero_mean.T / self.num_particles
        
        state_mean = X.copy()

        return state_mean, state_cov

    def Lie_sample_statistics(self):
        # compute sample mean and covariance on matrix Lie group
        mu0 = self.particles[0].pose()  # pick a sample as initial guess
        v = self.return_particles()
        max_iter = 1000
        iter = 1
        while iter < max_iter:
            mu = np.zeros_like(mu0)
            Sigma = np.zeros((6, 6))
            for i in range(self.num_particles):
                # left-invariant error: eta^L = X^-1 * X^hat
                v[i] = logm(np.linalg.inv(mu0) @ self.particles[i].pose())
                mu += v[i]
                vec_v = wedge(v[i])
                Sigma += vec_v @ vec_v.T
            mu = mu0 @ expm(mu / self.num_particles)
            Sigma = (1 / (self.num_particles - 1)) * Sigma   # unbiased sample covariance
            # check if we're done here!
            if np.linalg.norm(logm(np.linalg.inv(mu0) @ mu), 'fro') < 1e-8:
                return mu, Sigma
            else:
                mu0 = mu.copy()
            iter += 1
        return mu, Sigma

    def return_particles(self):
        """Debugging Tool"""
        particle_list=[]
        for particle in self.particles:
            particle_list.append(particle.pose())
        return particle_list
    
    def return_weigths(self):
        """Debugging Tool"""
        weight_list=[]
        for weight in self.weights:
            weight_list.append(weight)
        return weight_list
    
    @staticmethod
    def wrapToPi(angle):
        # Correct an angle for overflow beyond pi radians

        wrapped_angle = np.mod(angle + np.pi, 2*np.pi) - np.pi

        return wrapped_angle