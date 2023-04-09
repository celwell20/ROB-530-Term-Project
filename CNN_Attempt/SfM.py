import numpy as np
import matplotlib.pyplot as plt
from utils.cameraParams import generateIntrinsics
from utils.dataloader_new import load_TUM_data
from scripts.forward_process import process_7scene_SIFT
from scripts.optimization_GL import optimizationLS

from mpl_toolkits import mplot3d

def SfM(dataset='TUM', subdataset='1_desk2', plot_on=False):
    # print("hello world from SfM!")
    ### Load dataset
    if dataset == 'TUM':
        data_dict, posenet_x_predicted = load_TUM_data(subdataset)
    gap = 2
    num_images = 7 # CHANGE THIS FOR BETTER PERFORMANCE(?)

    #### ADDED CODE ####
    # Generate unfused trajectory
    xyz_cnn_pos  = np.array([data_dict['train_position'][int(i)] for i in (posenet_x_predicted-1)])
    xyz_cnn_orient  = np.array([data_dict['train_orientation'][int(i)] for i in (posenet_x_predicted-1)])

    trajectory_unfused=[]
    for indx in range(xyz_cnn_pos.shape[0]):
        relevant_array=np.concatenate((xyz_cnn_orient[indx,:,:],xyz_cnn_pos[indx,:].reshape((-1,1))),axis=1)
        final_array=np.concatenate((relevant_array,np.array([[0,0,0,1]])),axis=0)
        trajectory_unfused.append(final_array)

    np.save('trajectory_unfused.npy', trajectory_unfused)
    ####################

    ### parameters dictionary
    params = {}
    params['bm']        = 0.1
    params['sigma']     = 0.2
    params['alpha_m']   = 3
    params['max_range'] = 100

    ### results dictionary
    results = dict()
    results['orient']          = list()
    results['pose']            = list()
    results['orient_error']    = list()
    results['pose_error']      = list()
    results['reproject_error'] = list()

    index_list = list()
    # for i in range(0, min(posenet_x_predicted.shape[0], len(data_dict['train_images'])), 100):
    for i in range(0, 100, 5):
        idx = int(posenet_x_predicted[i] - 1)
        camParams = generateIntrinsics()

        ### Forward process for retrieves 3D points
        orientation, robotpose, pts2D, pts3D, K = process_7scene_SIFT(data_dict, i, idx,
                                                                      camParams, params,
                                                                      num_images=num_images, gap=gap)
        
        ### Need enough of 3D points for backward intersection
        if pts3D.shape[0] < 3:
            continue

        ### Backward intersection and optimization
        ### the output estimation is a tuple contains (orientation, robotpose, reproerror2, angle_var, position_var)
        estimation = optimizationLS(orientation, robotpose, pts2D, pts3D, pts3D.shape[0], K)

        if estimation[0] is None:
            continue
        else:
            index_list.append(i)
            for i, key in enumerate(results):
                results[key].append(estimation[i])

    ### plot the final results
    if plot_on:
        xyz_gt  = np.array([data_dict['test_position'][i] for i in index_list])
        xyz_est = np.array(results['pose'])
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], label='Ground Truth')
        ax.scatter3D(xyz_est[:, 0], xyz_est[:, 1], xyz_est[:, 2], label='Estimation')
        ax.set_zlim3d(0, 2)
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        plt.legend()
        plt.show()

    #### ADDED CODE ####
    # Save ground-truth trajectory
    xyz_gt_pos  = np.array([data_dict['test_position'][i] for i in index_list])
    xyz_gt_orient  = np.array([data_dict['test_orientation'][i] for i in index_list])

    trajectory_gt=[]
    for indx in range(xyz_gt_pos.shape[0]):
        relevant_array=np.concatenate((xyz_gt_orient[indx,:,:],xyz_gt_pos[indx,:].reshape((-1,1))),axis=1)
        final_array=np.concatenate((relevant_array,np.array([[0,0,0,1]])),axis=0)
        trajectory_gt.append(final_array)

    np.save('trajectory_gt.npy', trajectory_gt)

    # Save fused trajectory
    xyz_est_pos  = np.array(results['pose'])
    xyz_est_orient  = np.array(results['orient'])
    
    trajectory_fused=[]
    for indx in range(xyz_est_pos.shape[0]):

        relevant_array=np.concatenate((xyz_est_orient[indx,:,:],xyz_est_pos[indx,:].reshape((-1,1))),axis=1)
        final_array=np.concatenate((relevant_array,np.array([[0,0,0,1]])),axis=0)
        trajectory_fused.append(final_array)

    np.save('trajectory_fused.npy', trajectory_fused)
    ####################

if __name__ == '__main__':
    SfM(dataset='TUM', subdataset='1_desk2', plot_on=True)
