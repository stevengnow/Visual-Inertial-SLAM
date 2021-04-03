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
	    
	# data 1
	filename = "./data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)
	features = features[:,::10,:]

	# Initializations
	gen_velocity = np.vstack((linear_velocity,angular_velocity))
	T = np.eye(4)
	sigma_predict = np.eye(6)
	pose = np.zeros((4,4,3026))
	sigma_pose = np.zeros((6,6,3026))
	mu_pose = np.zeros((4,4,3026))
	pose[:,:,0] = T
	mu_pose[:,:,0] = np.eye(4)
	sigma_pose[:,:,0] = sigma_predict
	valid = features[0,:,0] != -1
	M = calibration_matrix(K,b)
	m_all = np.zeros((features.shape[0],features.shape[1]))
	begin_time = datetime.datetime.now()
	W = (0.00005)*np.eye(6)
	for time in tqdm(range(0,t.shape[1]-1)):
		tao = (t[:,time+1] - t[:,time])
		T = np.dot(T,exponential_map(tao*hat_map(gen_velocity[:,time])))
		pose[:,:,time+1] = T  									
		eq1 = expm(-tao*adjoint(gen_velocity[:,time]))
		sigma_predict = eq1 @ sigma_pose[:,:,time] @ eq1.T	+ W	
		valid = features[0,:,time] != -1
		idx = np.array(list(compress(range(len(valid)), valid))).astype(int) 
		coordinates = features[:,idx,time]
		o = pixel_to_optical(coordinates, M, K, b)
		m = np.dot(pose[:,:,time], np.dot(imu_T_cam, o))
		H = H_pose(M, imu_T_cam, pose[:,:,time], m, idx)
		if H.shape[0] > 0:  											# if observations seen, do EKF update
			Kp_gain = K_pose_gain(sigma_predict, H)
			sigma_pose[:,:,time+1] = sigma_pose_update(Kp_gain, H, sigma_predict)
			mu_pose[:,:,time+1] = mu_pose_update(Kp_gain, coordinates, pose[:,:,time+1], m, M, imu_T_cam)

		else:															# if no observations are seen, inherit the prediction
			sigma_pose[:,:,time+1] = sigma_predict
			mu_pose[:,:,time+1] = pose[:,:,time+1]
	visualize_trajectory_2d(mu_pose, show_ori = True)
