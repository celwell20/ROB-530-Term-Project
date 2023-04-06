import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import mat2quat, qmult, qinverse

class SE3:
    def __init__(self, position=np.zeros(3), rotation=np.eye(3)):
        # Construct an SE(3) object based on a provided rotation matrix and position vector.
        # if other:  
        #     self.position = other.position
        #     self.rotation = other.rotation
        # else:
        #     self.position = position
        #     self.rotation = rotation
        self.position = position
        self.rotation = rotation

    def pose(self):
        # Return the pose
        pose = np.row_stack((np.column_stack((self.rotation, self.position.T)),[0, 0, 0, 1]))
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
        self.particles = [SE3(position=initial_pose.position.copy(), rotation=initial_pose.rotation.copy()) for _ in range(num_particles)]
        self.weights = np.ones(num_particles) / num_particles

    def invSE3(self, pose):
        # Closed form inverse solution for a pose in SE(3)
        R = pose[:3,:3]
        p = pose[:3,3]
        inv = np.row_stack((np.column_stack( ( np.transpose(R), np.dot(-np.transpose(R),p) ) ),[0, 0, 0, 1]))
        return inv
    
    def orientation_error(self, T1, T2):
        q1 = mat2quat(T1[0:3, 0:3])
        q2 = mat2quat(T2[0:3, 0:3])
        q_rel = qmult(qinverse(q1), q2)
        angle = 2 * np.arccos(abs(q_rel[0]))

        if angle > np.pi:
            angle = 2 * np.pi - angle

        if angle < 1e-12:
            orientation_error = 0
        else:
            log_q_rel = np.zeros(4)
            log_q_rel[0] = np.log(abs(q_rel[0])) * np.sign(q_rel[0])
            log_q_rel[1:] = q_rel[1:] / np.sqrt(1 - q_rel[0] ** 2) * angle
            orientation_error = np.linalg.norm(log_q_rel[1:])
        return orientation_error

    # def orientation_error(R1, R2):
    #     log_R = scipy.linalg.logm(np.dot(np.transpose(R1), R2))

    #     # Calculate the geodesic distance as the Frobenius norm of the matrix logarithm
    #     distance = np.linalg.norm(log_R)

    #     return distance

    def predict(self, control_input):
        # control_input is the constant control velocities
        dt = 1
        new_particles = []
        # To apply the motion model we want to loop thru the particles and apply the
        # same constant control input to each particle
        init_ctrl = control_input.copy()
        for pose in self.particles:

            # make the motion model "semi-random"
            control_input = init_ctrl.copy()
            for i in range(6):
                control_input[i] += np.random.normal(loc=0., scale=3.5) 

            # first we want to calculate the "delta" transformation matrix induced by 
            # the constant control input
            dT = np.eye(4)
            dT[:3, 3] = control_input[:3] * dt
            r = self.rotation_matrix(control_input[3:]*dt)
            dT[:3, :3] = r.copy()
            # then we want to apply this dT to each particle, 
            # pose.pose() calls a function to return the 4x4 homoegenous transformation
            # that represents the SE(3) pose
            new_pose = np.dot(pose.pose(), dT)
            # and append these updated SE(3) object particles to the list "new_particles"
            # Use copy to ensure there is no memory address confusion on Python's end
            new_particles.append(SE3(position=new_pose[:3,3].copy(), rotation=new_pose[:3,:3].copy()))
        
        # we then copy new_particles to self.particles to ensure there are no silly referencing
        # issues that can be oh-so frustrating to debug
        self.particles = new_particles.copy()
    
    def update(self, measurement, covariance):
        # Loop through all the particles and compute their weigths based on their vicinity to the measurment
        for i in range(self.num_particles):
            particle = self.particles[i]
            # log map of dot product between particle's pose and the inverse of the measurement pose from CNN
            # log map gives us twist in se(3) which is subsequently converted to 6x1 twist coordinates
            # error = self.se3_to_twistcrd(scipy.linalg.logm(np.dot(self.invSE3(particle.pose()),measurement)))
            pre_err = np.dot( np.transpose(particle.pose()[:3,:3]), measurement[:3,:3] ) - np.eye(3)
            # pre_err = particle.pose()[:3,:3] - measurement[:3,:3]
            R_err = np.linalg.norm(pre_err,  ord='fro')
            t_err = particle.pose()[:3,3] - measurement[:3,3]
            # # this returns a scalar rotation error which is calculated using quaternions to try and avoid singularity issues 
            # R_err = np.array([self.orientation_error( (particle.pose()[:3,:3]) , (measurement[:3,:3]) )])
            error = np.vstack((R_err,t_err.reshape(3,1)))

            R_cov = covariance[3,3]**2 + covariance[4,4]**2 + covariance[5,5]**2
            t_cov = np.diag(covariance[3:,3:])
            quat_cov_vec = np.vstack((R_cov,t_cov.reshape(3,1)))
            quat_cov = np.diag( quat_cov_vec.reshape(4,) )
            # print(quat_cov)
            self.weights[i] *= multivariate_normal.pdf(error.ravel(), mean = np.zeros(4), cov = quat_cov)

        # Normalize the weights and compute the effective sample size
        self.weights /= np.sum(self.weights)

        n_eff = 1 / np.sum(self.weights**2)

        return n_eff
    
    def reinvigorate(self):
    # """
    # Reinvigorates the particles by randomly selecting them according to their weights.

    # Args:
    # particles (list): a list of particles
    # weights (list): a list of weights corresponding to each particle

    # Returns:
    # list: a list of particles that have been reinvigorated
    # """

        # Normalize the weights to sum to 1
        weights_sum = np.sum(self.weights)
        normalized_weights = [weight/weights_sum for weight in self.weights]

        # Sample new particles according to the normalized weights
        new_particles = np.random.choice(self.particles, size=self.num_particles, replace=True, p=normalized_weights)
        new_particle_weights = np.array([self.weights[self.particles.index(particle)] for particle in new_particles])

        self.weights = new_particle_weights.copy()
        self.particles = new_particles.copy().tolist()

        
    def resample(self):
        # Initiate the resampled set
        new_particles = []
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
            # check to make sure there is not a referencing issue here
            # new_particles.append(SE3(other=self.particles[count]))
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
            # omega = self.rot_vec(self.particles[i].rotation)
            omega = scipy.linalg.logm(self.particles[i].rotation)
            v = self.particles[i].position
            
            v = np.array([  [v[0]] , [v[1]] , [v[2]]  ])
            omega = np.array( [ [omega[2,1]] , [omega[0,2]] , [omega[1,0]] ] )

            twists[:,i] = np.vstack(( v, omega )).reshape(6,)
            # twists[:,i] = self.se3_to_twistcrd(scipy.linalg.logm(self.particles[i].pose())).reshape(6,)

        

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
            #twist = self.se3_to_twistcrd(scipy.linalg.logm(self.particles[s].pose()))

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
        
        # geodesic mean --> consider ?
        state_mean = X.copy()
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
    def rot_vec(R):
        # get the 3x1 rotation vector from the 3x3 rotation matrix
        r = Rotation.from_matrix(R)
        return r.as_rotvec()
    
    @staticmethod
    def se3_to_twistcrd(twist):
        # Convert from a member of little se(3) to a
        # 6x1 twist coordinate [v;w]

        omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
        
        vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
        
        twistcrd = np.vstack((vel,omega))

        return twistcrd
    
    @staticmethod
    def wrapToPi(angle):
        # Correct an angle for overflow beyond pi radians
        wrapped_angle = np.mod(angle + np.pi, 2*np.pi) - np.pi

        return wrapped_angle
    
    @staticmethod
    def sort_objects(objects, values):
    # Sort a list of objects by the corresponding values in another list.

    # :param objects: the list of objects to be sorted
    # :param values: the list of values to sort by
    # :return: a new list of objects sorted by the corresponding values
    # """
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    
        sorted_objects = [objects[i] for i in sorted_indices]
        sorted_weights = np.array([values[i] for i in sorted_indices])
        return sorted_objects, sorted_weights