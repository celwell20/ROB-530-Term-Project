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
    curr_data=np.load(file='PF_Data_'+str(i)+'.npy')
    #mean_R, mean_t, var_R, var_t = plot_compute_errors(viz_data=curr_data,fused_CNN=fused_CNN,unfused_CNN=unfused_CNN,gndTruth_CNN=gndTruth_CNN)

    for j in range(len(gndTruth_CNN)):
    
        [_, PF_R_err, PF_t_err] = error_calc(curr_data[j],gndTruth_CNN[j])
        all_PF_R_error[i-1,j]=PF_R_err
        all_PF_t_error[i-1,j]=PF_t_err

all_R_error[0,:]=np.min(all_PF_R_error, axis=0)
all_R_error[1,:]=np.max(all_PF_R_error, axis=0)
print(all_PF_R_error)
all_t_error[0,:]=np.min(all_PF_t_error, axis=0)
all_t_error[1,:]=np.max(all_PF_t_error, axis=0)
#print(all_PF_t_error)
for i in range(len(gndTruth_CNN)):

    [_, fused_R_err, fused_t_err] = error_calc(fused_CNN[i],gndTruth_CNN[i])
    all_R_error[2,i] = fused_R_err
    all_t_error[2,i] = fused_t_err
    [_, unfuse_R_err, unfuse_t_err] = error_calc(unfused_CNN[i],gndTruth_CNN[i])
    all_R_error[3,i] = unfuse_R_err
    all_t_error[3,i] = unfuse_t_err

#####

indices = range(len(all_R_error[0,:]))
print(indices)

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
plt.ylabel('Chordal  Error')

# Show the plot
plt.show()

######

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
plt.ylabel('Chordal  Error')

# Show the plot
plt.show()


viz_data=np.load(file='PF_Data_1.npy')

states_list=[gndTruth_CNN, unfused_CNN, fused_CNN, viz_data]
label_list=["Ground Truth", "Unfused Estimate", "Fused Estimate", "Particle Filter"]
overlay_plots(states_list, label_list)