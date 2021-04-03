import numpy as np
from numpy import linalg as LA

def exponential_map(hat):
    '''
    FUNCTION: 
        Define the transformation from the se(3) Lie Algebra to the SE(3) manifold.

    INPUT: 
        hat: Generalized Velocity hat map (4x4) = hat_map([linear velocity (3x1), angular velocity(3x1)])

    OUTPUT: 
        T in SE(3), the transformation 
    '''
    theta_norm = LA.norm(np.array((hat[2,1],hat[0,2],hat[1,0])))
    T = np.identity(4) + hat + ((1-np.cos(theta_norm))/theta_norm**2)*hat@hat + ((theta_norm-np.sin(theta_norm))/theta_norm**3)*hat@hat@hat
    
    return T

def hat_map(vect):
    '''
    FUNCTION: 
        Convert vector 6x1 into its hat map 4x4 pose matrix in se(3)

    INPUT: 
        vect: Vector 6x1

    OUTPUT: 
        4x4 pose matrix in se(3)
    '''
    eta_hat = np.zeros((4,4))
    eta_hat[:3,3] = vect[:3]
    eta_hat[0,1] = -vect[-1]
    eta_hat[1,0] = vect[-1]
    eta_hat[0,2] = vect[4]
    eta_hat[2,0] = -vect[4]
    eta_hat[1,2] = -vect[3]
    eta_hat[2,1] = vect[3]
    
    return eta_hat

def projection_derivative(q):
    '''
    FUNCTION: 
        Compute the projection derivative

    INPUT: 
        q: Vector (4x1)

    OUTPUT: 
        Projection Derivative (4x4)
    '''
    d = np.zeros((4,4))
    d[0,0], d[1,1], d[3,3] = 1/q[2], 1/q[2], 1/q[2]
    d[0,2] = -q[0]/q[2]**2
    d[1,2] = -q[1]/q[2]**2
    d[3,2] = -q[3]/q[2]**2
    
    return d

def pixel_to_optical(coordinates, M, K, b):
    '''
    FUNCTION: 
        Convert pixel coordinates in left and right camera to optical frame coordinates

    INPUT: 
        coordinates: (4xN) coordinates of obervations in pixel frame
        M: (4x4) calibration matrix
        K: (3x3) intrinsic matrix 
        b: (1x1) baseline  

    OUTPUT: 
        Outouts the stacked optical coordinate vector [x,y,z,1] (4xN)
    '''
    eq1 = np.dot(np.linalg.pinv(M),coordinates)
    z = K[0,0]*b/np.abs(coordinates[0]-coordinates[2])
    z[z>50] = 50
    x = z*(coordinates[0,:] - K[0,2])/K[0,0]
    y = z*(coordinates[1,:] - K[1,2])/K[1,1]
    o = np.vstack((x,y,z,np.ones(coordinates.shape[1])))

    return o

def calibration_matrix(K,b):

    '''
    FUNCTION: 
        Create the calibration matrix from camera intrinsic matrix and baseline 

    INPUT: 
        K: Camera intrinsic matrix (3x3)
        b: Baseline (1x1)

    OUTPUT: 
        Calibration matrix M (4x4)
    '''
    M = np.zeros((4,4))
    fsu = K[0,0] 
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]
    M[0,0], M[2,0] = fsu, fsu
    M[1,1], M[3,1] = fsv, fsv
    M[0,2], M[2,2] = cu, cu
    M[1,2], M[3,2] = cv, cv
    M[2,3] = -fsu*b
    
    return M

def H_jacobian_block(M, mu, imu_T_cam, T):
    '''
    FUNCTION: 
        This function will return the 4x3 jacobian block evaluated at mu_t

    INPUT: 
        M: (4x4) calibration matrix 
        mu: (4x1) obervation world coordinates
        imu_T_cam: (4x4) camera to imu
        T: (4x4) current imu pose

    OUTPUT: 
        Return Jacobian H (4x3)
    '''
    P = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))
    H = np.zeros((4,3))
    o_T_imu = np.linalg.inv(imu_T_cam)
    eq1 = M @ projection_derivative(o_T_imu @ np.linalg.inv(T) @ mu)
    
    return eq1 @ o_T_imu @ np.linalg.inv(T) @ P.T

def H_jacobian(idx_update, features, M, mu, imu_T_cam, T):
    '''
    FUNCTION: 
        Computes the (4Nx3M) Observation Model Jacobian matrix H

    INPUT: 
        idx_update: (Nx1) indices of observations needed to update 
        features: (4x13289x3026) obervations per time step
        M: (4x4) calibration matrix 
        mu: (4xN) obervations in need of updates in world coordinates
        imu_T_cam: (4x4) camera to imu
        T: (4x4) current imu pose

    OUTPUT:
        Jacobian matrix H (4Nx3M) used for EKF Update
    '''
    H = np.zeros((4*idx_update.shape[0],3*features.shape[1]))
    for i in range(idx_update.shape[0]):
        H[4*i:4*i+4, 3*int(idx_update[i]):3*int(idx_update[i])+3] = H_jacobian_block(M, mu[:,i], imu_T_cam, T)
    
    return H

def Kalman_gain(H, sigma, V):
    '''
    FUNCTION: 
        Calculate the Kalman Gain

    INPUT:
        H: (4Nx3M) Observation Model Jacobian from H_jacobian function
        sigma: (3Mx3M) EKF update covariance matrix
        V: (4x4) noise scalar

    OUTPUT:
        The Kalman Gain used for EKF update step
    '''
    noise = np.kron(np.eye(int(H.shape[0]/4)), V)
    K = sigma @ H.T @ np.linalg.pinv(H @ sigma @ H.T + noise)
    return K

def sigma_update(K, H, sigma, I):
    '''
    FUNCTION:
        EKF update for the covariance

    INPUT:
        K: (3Mx4N) Kalman Gain
        H: (4Nx3M) Observation Model Jacobian
        sigma: (3Mx3M) previous iteration Covariance
        I: (3Mx3M) identity matrix 

    OUTPUT:
        Covariance for landmark prior of the next update step
    '''
    return (I - K @ H) @ sigma

def mu_update(observe, K, M, imu_T_cam, T, mu, idx_update):
    '''
    FUNCTION: 
        Perform the EKF mu update
    INPUT:
        observe: (4xN) the observed coordinates to update
        K: (3Mx4N) Kalman gain
        M: (4x4) Calibration matrix
        imu_T_cam: (4x4) camera to imu transformation
        T: (4x4) current imu pose
        mu: (4xM) all current mu
        idx_update: (1xN) the indices to update
    OUTPUT:
        The EKF mu update: update the mean of all landmark 
        locations that needed updating
    '''
    innovation = observe - xyz_to_pixel(M, imu_T_cam, T, mu[:,idx_update])
    innovation = innovation.ravel(order = 'F')
    eq1 = mu[:3,:].ravel(order = 'F') + K @ innovation
    eq1 = eq1.reshape((3,mu.shape[1]), order = 'F')
    
    return np.vstack((eq1, np.ones(mu.shape[1])))

def xyz_to_pixel(M, imu_T_cam, T, mu):
    '''
    FUNCTION: 
        Convert from xyz world coordinates to pixel (ul ur vl vr) coordinates

    INPUT:
        M: (4x4) Calibration matrix
        imu_T_cam: (4x4) camera to imu transformation
        T: (4x4) current imu pose
        mu: (4xN) all current guesses for landmark positions

    OUTPUT:
        z: estimate of pixel coordinates
    '''
    o_T_imu = np.linalg.inv(imu_T_cam)
    eq1 = o_T_imu @ np.linalg.inv(T) @ mu
    pi = eq1/eq1[2]

    return M @ pi

def Kalman_gain_patch(update_idx, H, sigma):
    '''
    FOR FURTHER COMPUTATIONAL EFFICIENCY, NOT IMPLEMENTED.
    REMOVES SPARSITY AND COMPUTES ONLY NON-ZERO ENTRIES OF INTEREST FOR THE KALMAN GAIN
    '''
    A = np.array((1,2,3,4,5,6,7,8,9,10))
    patch_idx = get_patch_idx(update_idx)
    noise = 7*np.eye(4)
    
    K_gain = np.zeros((H.shape[1], H.shape[0]))
    sig_update = np.zeros((H.shape[1],H.shape[1]))
    for patch in range(len(update_idx)-1):
        
        H_patch = H[4*patch:4*patch+4 , patch_idx[3*patch]:patch_idx[3*patch+3]]
        sigma_patch = sigma[patch_idx[3*patch]:patch_idx[3*patch+3] , patch_idx[3*patch]:patch_idx[3*patch+3]]
        I = np.eye(sigma_patch.shape[0])
        K_patch = sigma_patch @ H_patch.T @ np.linalg.inv(H_patch @ sigma_patch @ H_patch.T + noise)
        K_gain[patch_idx[3*patch]:patch_idx[3*patch+3], 4*patch:4*patch+4] = K_patch
        sig_update[patch_idx[3*patch]:patch_idx[3*patch+3] , patch_idx[3*patch]:patch_idx[3*patch+3]] = (I - K_patch @ H_patch) @ sigma_patch

    return K_gain, sig_update

def get_patch_idx(update_idx):
    '''
    FOR FURTHER COMPUTATIONAL EFFICIENCY, NOT IMPLEMENTED.
    '''

    return [i for n in update_idx for i in (3*n,3*n+1,3*n+2)]



