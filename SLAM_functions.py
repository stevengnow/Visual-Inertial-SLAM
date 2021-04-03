import numpy as np
from numpy import linalg as LA
from functions import *
from scipy.linalg import expm

def adjoint(u):
    '''
    FUNCTION:
        Return the adjoint of the generalized velocity vector
    
    INPUT: 
        u: (6x1) generalized velocity vector

    OUTPUT:
        adj: (6x6) adjoint of vector u
    '''
    adj = np.zeros((6,6))
    adj[0:3,3:] = hat_map_2(u[:3])
    adj[0:3,0:3], adj[3:,3:] = hat_map_2(u[3:]), hat_map_2(u[3:])

    return adj

def hat_map_2(vect):
    '''
    FUNCTION:
        Return the 3x3 hat map matrix of an input vector

    INPUT:
        vect: Vector (3x1)

    OUTPUT:
        hat map (3x3) of the vector
    '''
    h = np.zeros((3,3))
    h[0,1] = -vect[2]
    h[1,0] = vect[2]
    h[0,2] = vect[1]
    h[2,0] = -vect[1]
    h[1,2] = -vect[0]
    h[2,1] = vect[0]

    return h

def H_pose_block(M, imu_T_cam, mu, m):
    '''
    FUNCTION: 
        Compute the (4x6) H jacobian block

    INPUT: 
        M: (4x4) calibration matrix 
        imu_T_cam: (4x4) camera to imu
        mu: (4x4) current imu pose
        m: (4x1) obervation world coordinates
        
    OUTPUT:
        H_block: (4x6) Jacobian of obervation model for a single landmark
    '''
    H = np.zeros((4,6))
    o_T_imu = np.linalg.inv(imu_T_cam)
    H_block = - M @ projection_derivative(o_T_imu @ np.linalg.inv(mu) @ m) @ o_T_imu @ circle_dot(np.linalg.inv(mu) @ m)
    return H_block

def H_pose(M, imu_T_cam, mu, m, idx):
    '''
    FUNCTION:
        Compute entire H (4Nx6) Jacobian
    
    INPUT:
        M: (4x4) calibration matrix 
        imu_T_cam: (4x4) camera to imu
        mu: (4x4) current imu pose
        m: (4xN) obervation world coordinates
        idx: (Nx1) list of landmark indices
    
    OUTPUT:
        H: (4Nx6) full Jacobian matrix for EKF update
    '''
    H = np.zeros((4*idx.shape[0],6))
    
    for i in range(idx.shape[0]):
        H[4*i:4*i+4,:] = H_pose_block(M, imu_T_cam, mu, m[:,i])
    return H

def circle_dot(s):
    '''
    FUNCTION:
        Compute the circle dot exponential

    INPUT:
        s: (4x1) Vector

    OUTPUT:
        arr: (4x6) circle dot exponent
    '''
    arr = np.zeros((4,6)) 
    arr[:3,:3] = np.eye(3)
    arr[:3,3:] = -hat_map_2(s[:3])
    return arr

def K_pose_gain(sigma_predict, H):
    '''
    FUNCTION:
        EKF Kalman gain calculation
    
    INPUT:
        sigma_predict: (6x6) predicted covariance
        H: (4Nx6) Jacobian of the observation model

    OUTPUT:
        K gain : (6x4N) Kalman gain 
    '''
    noise = np.kron(np.eye(int(H.shape[0]/4)), 7*np.eye(4))
    return sigma_predict @ H.T @ np.linalg.pinv(H @ sigma_predict @ H.T + noise)

def sigma_pose_update(K,H, sigma):
    '''
    FUNCTION:
        EKF covariance of the pose update calculation
    
    INPUT: 
        K: (6x4N) Kalman gain
        H: (4Nx6) Jacobian of the observation model
        sigma: (6x6) predicted pose covariance
    
    OUTPUT:
        covariance_update: (6x6) updated covariance
    '''
    I = np.eye(6)
    return (I- K @ H) @ sigma

def mu_pose_update(K, coordinates, pose, m, M, imu_T_cam):
    '''
    FUNCTION:
        EKF mu update calculation
    
    INPUT:
        K: (6x4N) Kalman gain
        coordinates: (4xN) pixel coordinates of observations
        pose: (4x4) current predicted robot pose
        m: (4xN) world coordinates of observations
        M: (4x4) Calibration matrix
        imu_T_cam: (4x4) camera to imu transformation
    
    OUTPUT:
        mu_updated: (4x4) updated pose of the robot
    '''
    innovation = coordinates - xyz_to_pixel(M, imu_T_cam, pose, m)
    innovation = innovation.ravel(order = 'F')
    eq1 = expm(hat_map(K @ innovation))

    return pose @ eq1

def delta_mu_calc(K, coordinates, pose, m, M, imu_T_cam,idx_update):
    '''
    FUNCTION: 
        Calculates the delta mu function. First six entries are for pose EKF update, next 3M is for map EKF update
    INPUT:
        K: (6x4N) Kalman gain
        coordinates: (4xN) pixel coordinates of observations
        pose: (4x4) current predicted robot pose
        m: (4xN) world coordinates of observations
        M: (4x4) Calibration matrix
        imu_T_cam: (4x4) camera to imu transformation
        idx_update: (Nx1) landmark indices to update
    OUTPUT:
        Delta mu: ((3M+6)x1) 
    '''
    innovation = coordinates - xyz_to_pixel(M, imu_T_cam, pose, m)
    innovation = innovation.ravel(order = 'F')
    return K @ innovation

def mu_map_update(mu, delta,idx_update):
    '''
    FUNCTION:
        Updates the map landmark positions for the SLAM algorithm
    
    INPUT:
        mu: (4xM) vector of all landmark points
        delta: (3Mx1) vector of delta mu
        idx_udpate: (Nx1) landmark indices to update
    OUTPUT:
        updated map: (4xM) updated landmark points
    '''
    d = np.zeros((delta.shape[0]))
    d[idx_update] = delta[idx_update]
    eq1 = mu[:3,:].ravel(order = 'F') + d
    eq1 = eq1.reshape((3, mu.shape[1]), order = 'F')
    
    return np.vstack((eq1, np.ones(mu.shape[1])))


