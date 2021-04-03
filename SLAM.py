# %%
import numpy as np
from utils import *
from functions import *
from SLAM_functions import *
from itertools import compress
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm

if __name__ == '__main__':

        
    # data 1.
    filename = "./data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)
    features = features[:,::10,:]

    # Initializations
    begin_time = datetime.datetime.now()
    gen_velocity = np.vstack((linear_velocity,angular_velocity))
    master_idx = []
    
    pred_pose = np.zeros((4,4,3026))
    sigma_pose = np.zeros((6,6,3026))
    mu_pose = np.zeros((4,4,3026))
    mu_map = np.zeros((features.shape[0],features.shape[1]))

    # ----- NOISE & INITIALIZATION HYPER PARAMETER -----
    pred_pose[:,:,0] = 1*np.eye(4)                  # pose initialize
    sigma_predict = 1*np.eye(6)                     # pose covariance initialize
    sigma_m = 0.01*np.eye(3*features.shape[1])      # map covariance initialize
    W = (0.0005)*np.eye(6)
    ob_noise = 30
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- 

    V = np.eye(4)
    I = np.eye(3*features.shape[1]+6)
    mu_pose[:,:,0] = np.eye(4)
    M = calibration_matrix(K,b)

    #SLAM INITIALIZATIONS
    sigma = np.zeros((3*features.shape[1]+6,3*features.shape[1]+6))
    sigma[:6,:6] = sigma_predict
    sigma[6:,6:] = sigma_m
    
    #SLAM
    for time in tqdm(range(0,t.shape[1]-1)):
        tao = (t[:,time+1] - t[:,time])
        valid = features[0,:,time] != -1
        idx = np.array(list(compress(range(len(valid)), valid))).astype(int) 
        idx_update = np.intersect1d(idx, master_idx)            # these are the obervations to update
        idx_init = np.setdiff1d(idx, idx_update)		        # these are the obersvations to update
        master_idx = np.union1d(master_idx,idx).astype(int)		# update the master index
        
        # PREDICT NEXT POSE : MU(T+1|T)
        pred_pose[:,:,time+1] = np.dot(pred_pose[:,:,time],expm(tao*hat_map(gen_velocity[:,time])))

        #PREDICT NEXT SIGMA : SIGMA(T+1|T)
        eq1 = expm(-tao*adjoint(gen_velocity[:,time]))
        sigma[:6,:6] = (eq1 @ sigma[:6,:6] @ eq1.T) + W  
        sigma[:6,6:] = eq1 @ sigma[:6,6:]
        sigma[6:,:6] = sigma[6:,:6] @ eq1

        # INITIALIZE
        if idx_init.shape[0] > 0:
            mu_pose[:,:,time] = pred_pose[:,:,time+1]
            coordinates_i = features[:,idx_init,time]
            o_i = pixel_to_optical(coordinates_i, M, K, b)
            m_i = np.dot(pred_pose[:,:,time], np.dot(imu_T_cam, o_i))
            mu_map[:,idx_init] = m_i
            
        # UPDATE
        if idx_update.shape[0] > 0:
            # CALCULATES THE Z AND Z_approx
            coordinates = features[:,idx_update,time]   
            m = mu_map[:,idx_update]

            # CALCULATES THE H-POSE AND H-MAP THEN CONCATANATE
            H_p = H_pose(M, imu_T_cam, pred_pose[:,:,time], m, idx_update)
            H_m = H_jacobian(idx_update, features, M, m, imu_T_cam, pred_pose[:,:,time])
            H = np.hstack((H_p,H_m))
            
            # CALCULATES UPDATE KALMAN GAIN
            K_gain = Kalman_gain(H, sigma, ob_noise*V)

            # CALCULATES UPDATE SIGMA
            sigma = sigma_update(K_gain, H, sigma, I)
            
            # CALCULATES UPDATE MU
            delta_mu = delta_mu_calc(K_gain, coordinates, pred_pose[:,:,time], m, M, imu_T_cam, idx_update)
            mu_pose[:,:,time+1] = pred_pose[:,:,time] @ expm(hat_map(delta_mu[:6]))
            mu_map = mu_map_update(mu_map, delta_mu[6:],idx_update)

        else:
            mu_pose[:,:,time+1] = pred_pose[:,:,time+1]

        # FOR SHOWING IMAGES
        if time % 300 == 0:
            visualize_trajectory_map_2d(mu_pose[:,:,:time-1], mu_map, show_ori = False)
            
    visualize_trajectory_map_2d(mu_pose, mu_map, show_ori = True)

