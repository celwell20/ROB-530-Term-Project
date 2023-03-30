import numpy as np
from scipy.stats import norm

class SE3:
    def __init__(self, position=np.zeros(3), rotation=np.eye(3)):
        
        self.position = position
        self.rotation = rotation
        self.pose = [rotation, position.T
                     [0, 0, 0, 1]        ]

    
class ParticleFilterSE3:
    def __init__(self, num_particles, initial_pose):

        self.num_particles = num_particles
        self.particles = [SE3(position=initial_pose.position, rotation=initial_pose.rotation) for _ in range(num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input):

        for i in range(self.num_particles):

            particle = self.particles[i]
            
            # Update the particle pose using the control input
            particle.position += control_input[:3]
            particle.rotation = particle.rotation @ self.rotation_matrix(control_input[3:])
    
    def update(self, measurement):

        for i in range(self.num_particles):

            particle = self.particles[i]

            # Calculate the error in se(3)
            error = np.linalg.logm(np.dot(particle.pose,np.linalg.inv(measurement.pose)))
            weight = norm.pdf(error, mean=0, cov=np.eye(6)*0.1) # NEED TO TUNE COVARIANCVE

            self.weights[i] = weight

        # Normalize the weights
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        new_particles = []
        # Sample particles with replacement based on their weights
        indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=self.weights)
        for i in indices:
            new_particles.append(self.particles[i])
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    @staticmethod
    def rotation_matrix(rotation):
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