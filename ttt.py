import Particle_Filter
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation
import read

gndTruth_CNN = np.load(file='trajectory_gt_default.npy')
unfused_CNN = np.load(file='trajectory_unfused_default.npy')
fused_CNN = np.load(file='trajectory_fused_default.npy')
print(len(gndTruth_CNN))
print(len(unfused_CNN))
print(len(fused_CNN))
