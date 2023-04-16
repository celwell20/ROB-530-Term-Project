import scipy

import numpy as np
from scipy.spatial.transform import Rotation

def rot_vec(R):
        # get the 3x1 rotation vector from the 3x3 rotation matrix

        r = Rotation.from_matrix(R)

        return r.as_rotvec()

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
    twist_se3 = np.row_stack((np.column_stack((w_hat, v.reshape(3,))),[0, 0, 0, 0]))
    
    return twist_se3

def se3_to_twistcrd(twist):
    """Converts an element from se(3) to 6x1 twist coordinate [v;w]"""

    omega = np.array([[twist[2,1]],[twist[0,2]],[twist[1,0]]])
    vel = np.array([[twist[0,3]],[twist[1,3]],[twist[2,3]]])
    twistcrd = np.vstack((vel,omega))

    return twistcrd

def se3toSE3(twist):
    """Converts an element from se(3) to SE(3)"""

    R = scipy.linalg.expm(twist[:3,:3])
    t = twist[:3,3]
    SE3 = np.row_stack((np.column_stack((R, t.reshape(3,).copy())),[0, 0, 0, 1]))

    return SE3

def error_calc(state, truth):
    """Calculate the error for a given value from the ground truth"""

    # For the rotational error, compute the chordal distance R^T*R - I(3x3) and find the Frobenius norm
    pre_err = np.dot( np.transpose(truth[:3,:3]), state[:3,:3] ) - np.eye(3)
    R_err = np.linalg.norm(pre_err,  ord='fro')

    # For the translational error, compute the delta and find the 2-norm
    t_err = truth[:3,3] - state[:3,3]
    t_err = np.linalg.norm(t_err)

    # Stack the errors
    error = np.vstack((t_err,R_err))
    
    return error, R_err, t_err