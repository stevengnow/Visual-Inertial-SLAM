# %%
import numpy as np
from utils import *
from functions import *
from itertools import compress
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
if __name__ == '__main__':
   
	# data 1
    filename = "./data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)
    features = features[:,::10,:]

    # Initializations
    gen_velocity = np.vstack((linear_velocity,angular_velocity))
    T = np.eye(4)
    pose = np.zeros((4,4,3026))
    pose[:,:,0] = T

    begin_time = datetime.datetime.now()

    # (a) IMU Localization via EKF Prediction
    for time in tqdm(range(0,t.shape[1]-1)):
        tao = (t[:,time+1] - t[:,time])
        T = np.dot(T,exponential_map(tao*hat_map(gen_velocity[:,time])))
        pose[:,:,time+1] = T


    master_idx = []
    sigma = 0.01*np.eye(3*features.shape[1])
    I = np.eye(3*features.shape[1])
    V = np.eye(4)
    M = calibration_matrix(K,b)
    m_all = np.zeros((features.shape[0],features.shape[1]))

	# (c) Landmark Mapping via EKF Update
    for time in tqdm(range(0,t.shape[1])):								
        valid = features[0,:,time] != -1
        idx = np.array(list(compress(range(len(valid)), valid))) 
        idx_update = np.intersect1d(idx, master_idx)    # these are the obervations to update
        idx_init = np.setdiff1d(idx, idx_update)		# these are the obersvations to update
        master_idx = np.union1d(master_idx,idx).astype(int)		# update the master index

		#UPDATE
        if idx_update.shape[0] > 0:
            coordinates = features[:,idx_update,time]
            o = pixel_to_optical(coordinates, M, K, b)
            # m = np.dot(pose[:,:,time], np.dot(imu_T_cam, o))     # for dead reckoning, comment in lines 51-52, and comment out lines 54-58.
            # m_all[:,idx_update] = m 
            
            m = m_all[:,idx_update]
            H = H_jacobian(idx_update, features, M, m, imu_T_cam, pose[:,:,time])
            K_gain = Kalman_gain(H, sigma, V)
            m_all = mu_update(coordinates, K_gain, M, imu_T_cam, pose[:,:,time], m_all, idx_update)
            sigma = sigma_update(K_gain, H, sigma, I)

        #INITIALIZE
        if idx_init.shape[0] > 0:
            coordinates_i = features[:,idx_init,time]
            o_i = pixel_to_optical(coordinates_i, M, K, b)
            m_i = np.dot(pose[:,:,time], np.dot(imu_T_cam, o_i))
            m_all[:,idx_init] = m_i
        
    print('Time | ', datetime.datetime.now() - begin_time)
    visualize_trajectory_map_2d(pose, m_all, show_ori = True)
    visualize_trajectory_2d(pose, show_ori = True)
    plt.scatter(m_all[0,:], m_all[1,:], 1)
    plt.show()