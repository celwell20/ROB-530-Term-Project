import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from functions.utils import error_calc
from functions.viz import plot_errors, overlay_plots, plot_particles, plot_compute_errors

# Load the Convolutional Neural Network data
gndTruth_CNN = np.load(file='data/trajectory_gt_default.npy')
unfused_CNN = np.load(file='data/trajectory_unfused_default.npy')
fused_CNN = np.load(file='data/trajectory_fused_default.npy')

# Initialise array
all_R_error = np.zeros((4,len(gndTruth_CNN)))
all_t_error = np.zeros((4,len(gndTruth_CNN)))
all_PF_R_error = np.zeros((10,len(gndTruth_CNN)))
all_PF_t_error = np.zeros((10,len(gndTruth_CNN)))

# Load the Particle Filter results
for i in range(1, 11):
    curr_data=np.load(file='numpy_results/PF_Data_'+str(i)+'.npy')

    for j in range(len(gndTruth_CNN)):
    
        [_, PF_R_err, PF_t_err] = error_calc(curr_data[j],gndTruth_CNN[j])
        all_PF_R_error[i-1,j]=PF_R_err
        all_PF_t_error[i-1,j]=PF_t_err

all_R_error[0,:]=np.min(all_PF_R_error, axis=0)
all_R_error[1,:]=np.max(all_PF_R_error, axis=0)

all_t_error[0,:]=np.min(all_PF_t_error, axis=0)
all_t_error[1,:]=np.max(all_PF_t_error, axis=0)

for i in range(len(gndTruth_CNN)):

    [_, fused_R_err, fused_t_err] = error_calc(fused_CNN[i],gndTruth_CNN[i])
    all_R_error[2,i] = fused_R_err
    all_t_error[2,i] = fused_t_err
    [_, unfuse_R_err, unfuse_t_err] = error_calc(unfused_CNN[i],gndTruth_CNN[i])
    all_R_error[3,i] = unfuse_R_err
    all_t_error[3,i] = unfuse_t_err

indices = range(len(all_R_error[0,:]))

# Create a plot with the indices on the x-axis and the errors on the y-axis
plt.figure()
ax = plt.axes()
ax.plot(indices, all_R_error[0,:],label="Particle Filter Lower-Bound",color='red',linestyle='--')
ax.plot(indices, all_R_error[1,:],label="Particle Filter Upper-Bound",color='red')
ax.plot(indices, all_R_error[2,:],label="Fused Estimate Error",color='green')
ax.plot(indices, all_R_error[3,:],label="Unfused Estimate Error",color='yellow')
plt.fill_between(indices, all_R_error[0,:], all_R_error[1,:], color='red', alpha=0.5)
ax.legend()

# Set the title and labels for the plot
plt.title('Rotation Error')
plt.xlabel('Trajectory Time-Step')
plt.ylabel('Chordal  Distance')

# Show the plot
plt.show()

# Create a plot with the indices on the x-axis and the errors on the y-axis
plt.figure()
ax = plt.axes()
ax.plot(indices, all_t_error[0,:],label="Particle Filter Lower-Bound",color='red',linestyle='--')
ax.plot(indices, all_t_error[1,:],label="Particle Filter Upper-Bound",color='red')
ax.plot(indices, all_t_error[2,:],label="Fused Estimate Error",color='green')
ax.plot(indices, all_t_error[3,:],label="Unfused Estimate Error",color='yellow')
plt.fill_between(indices, all_t_error[0,:], all_t_error[1,:], color='red', alpha=0.5)
ax.legend()

# Set the title and labels for the plot
plt.title('Translation Error')
plt.xlabel('Trajectory Time-Step')
plt.ylabel('Euclidian  Distance')

# Show the plot
plt.show()

mean_R = np.mean(all_R_error,axis=1)
mean_t = np.mean(all_t_error,axis=1)
var_R = np.mean(all_R_error ** 2,axis=1)
var_t = np.mean(all_t_error **2,axis=1)

# Report the errors in the terminal
print("Mean of error(Rotation)")
print("Particle filter Lower Bound: ", mean_R[0])
print("Particle filter Upper Bound: ", mean_R[1])
print("Original Fused Estimate: ",mean_R[2])
print("Original Unfused Unfused: ",mean_R[3])

print("Variance of error(Roatation)")
print("Particle filter Lower Bound: ", var_R[0])
print("Particle filter Upper Bound: ", var_R[1])
print("Original Fused Estimate: ",var_R[2])
print("Original Unfused Unfused: ",var_R[3])

print("Mean of error(tanslation)")
print("Particle filter Lower Bound: ", mean_t[0])
print("Particle filter Upper Bound: ", mean_t[1])
print("Original Fused Estimate: ",mean_t[2])
print("Original Unfused Unfused: ",mean_t[3])

print("Variance of error(tanslation)")
print("Particle filter Lower Bound: ", var_t[0])
print("Particle filter Upper Bound: ", var_t[1])
print("Original Fused Estimate: ",var_t[2])
print("Original Unfused Unfused: ",var_t[3])

###########################################################

viz_data=np.load(file='numpy_results/PF_Data_2.npy')

states_list=[gndTruth_CNN, unfused_CNN, fused_CNN, viz_data]
label_list=["Ground Truth", "Unfused Estimate", "Fused Estimate", "Particle Filter"]
overlay_plots(states_list, label_list)