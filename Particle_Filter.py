import numpy as np
from scipy.stats import norm

class SE3:
    def __init__(self, position=np.zeros(3), rotation=np.eye(3)):
        # Construct an SE(3) object based on a provided rotation matrix and position vector.
        self.position = position
        self.rotation = rotation

    def pose(self):
        # Return the pose
        pose = [[self.rotation, self.position.T],[0, 0, 0, 1]]
        return pose
    
class ParticleFilterSE3:
    def __init__(self, num_particles, initial_pose):
        # Initialise the filter with a specified number of particles and their weights
        self.num_particles = num_particles
        self.particles = [SE3(position=initial_pose.position, rotation=initial_pose.rotation) for _ in range(num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input):
        # Loop through all the particles and predict their location based on the control inputs
        for i in range(self.num_particles):
            particle = self.particles[i]
            particle.position += control_input[:3]
            particle.rotation = particle.rotation @ self.rotation_matrix(control_input[3:])
    
    def update(self, measurement, covariance):
        # Loop through all the particles and compute their weigths based on their vicinity to the measurment
        for i in range(self.num_particles):
            particle = self.particles[i]
            error = np.linalg.logm(np.dot(particle.pose(),np.linalg.inv(measurement.pose())))
            weight = norm.pdf(error, mean = 0, cov = covariance)
            self.weights[i] = weight

        # Normalize the weights and compute the effective sample size
        self.weights /= np.sum(self.weights)
        n_eff = 1 / np.sum(self.weights**2)

        return n_eff
    
    def resample(self):
        # Initiate the resampled set
        new_particles = []
        new_weights = []
        
        """
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

        ## Written by a human:
        ## Low Variance Resampling Algorithm
        """

        # Generate a random number r between 0 and Minv
        Minv = 1 / self.num_particles
        r = np.random.normal(0, Minv)
        c = self.weights[0]

        i = 0
        U = 0
        
        for m in range(self.num_particles):
            U = r + m * Minv
            
            while (c < U):
                i += 1
                c += self.weights[i]

            new_particles.append(self.particles[i])
            new_weights.append(self.weights[i])

        # Generate more of the resampled particles to match self.num_particles
        new_particles = np.random.choice(new_particles, self.num_particles, replace=True, p=self.weights)
        self.particles = new_particles

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