# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:13:44 2021

@author: ZHU Haoren
"""
import numpy as np

def global_min_var(omega,r):
    """
    This function generates the markov mean-variance portfolio.
    
    params:
        omega: covariance matrix
        r:     expected return vector
        
    returns:
        a dictionary including the information of the portfolio
    """
    omega,r = [np.array(x) for x in [omega,r]]
    omega = np.array(omega)
    omega_inv = np.linalg.inv(omega)
    ones = np.ones(omega.shape[1])
    ones_t = ones.transpose()
    r_t = r.transpose()
    a = ones_t.dot(omega_inv.dot(ones))
    b = ones_t.dot(omega_inv.dot(r))
    c = r_t.dot(omega_inv.dot(r))
    w_g = 1/a*omega_inv.dot(ones)
    #print(cal_var(w_g,omega))
    return dict({'inv': omega_inv,
                 'a': a,
                 'b': b,
                 'c': c,
                 'r': b/a, 
                 'var': 1/a, 
                 'w_g': w_g})
    
def recover_cov_image(cov_images, num_asset):
    """
    This function recover the cov/cor/inv vector to the original matrix.
    
    params:
        cov_images: an array of cov/cor/inv vectors
        num_asset:  number of assets
        
    returns:
        an array of the recovered cov/cor/inv matrixes
    """
    recover_images = []
    for i in cov_images:
        blank = np.eye(num_asset)
        idx = np.triu_indices(num_asset,0)
        blank[idx] = i
        
        recover_image = np.triu(blank,1).T + blank
        recover_images.append(recover_image)
    return np.array(recover_images)

def mat_2_vec(mat, length=136):
    """
    This function transform the predicted image to cov/cor/inv vector by removing the dummy zero and reshape the matrix. 
    
    params:
        mat:    the input matrix representing the predicted image
        length: the orignal length of the cov/cor/inv vector
        
    returns:
        the cov/cor/inv vector
    """
    vec = mat.reshape((1,-1)).squeeze()[:length]
    return vec

def recover(t, num_asset = 16):
    """
    This function transform the predicted image to the original cov/cor/inv matrix without using cluster.
    
    params:
        t:         an array of predicted images
        num_asset: number of asset
        
    returns:
        the original cov/cor/inv matrix
    """
    return recover_cov_image(np.array([mat_2_vec(i, int(num_asset*(num_asset+1)/2)) for i in t]),num_asset)    

def recover_cluster(t, cor_map, num_asset = 16):
    """
    This function transform the predicted image to the original cov/cor/inv matrix with cluster information.
    
    params:
        t:         an array of predicted images
        cor_map:   an array indicating the mapping between the vector after clustering and vector before clustering (this part should be modified if we use another way to arrange the cluster in the image)
        num_asset: number of asset
        
    returns:
        the original cov/cor/inv matrix
    """
    vecs = np.array([mat_2_vec(i, int(num_asset*(num_asset+1)/2)) for i in t])
    new_vecs = np.zeros(vecs.shape)
    for i in range(vecs.shape[1]):
        new_vecs[:, cor_map[i]] = vecs[:,i]
    return recover_cov_image(new_vecs,num_asset)

def get_pred_risk(pred, result, num_asset, adm, method, maps = None):
    """
    Calculate the portfolio risk of the predicted correlation images.
    
    params:
        pred:        a list of correlation predicted images (the one before recovered)
        prev_cov:    previous covariance matrix
        truth_cov:   groundtruth covariance matrix
        use_cluster: whether the predicted correlation images have used cluster
        cor_map:     mapping between the original correlation vector and the one after cluster.
        
    returns:
        risks of the groundtruth, predicted one, and previous one, in shape of [r4, r5, r6]
    """
    covs = result[0][2]
    truth_cov = [] # Store the true covariance matrix
    prev_cov = [] # Store the previous covariance matrix
    for i in covs:
        truth_cov.append(i[2][0])
        prev_cov.append(i[1][-1])
    truth_cov = np.array(truth_cov).squeeze()
    prev_cov = np.array(prev_cov).squeeze()
    if method == "cluster":
        truth_cov = recover_cluster(truth_cov, maps[0],num_asset)
        prev_cov = recover_cluster(prev_cov, maps[0],num_asset)
    else:   
        truth_cov = recover(truth_cov,num_asset)
        prev_cov = recover(prev_cov,num_asset)
    
    if adm == 1:
        pred = pred * 2 - 1 # Correlation matrix is normalized
        if method == "cluster":
            pred = recover_cluster(pred, maps[adm], num_asset)
        else:
            pred = recover(pred, num_asset)
        self_var = np.array([pow(np.diag(i),0.5) for i in prev_cov])
        pair_var = np.array([i*i.reshape((len(i),1)) for i in self_var])
        pred_cov = pred * pair_var    
    elif adm == 0:
        if method == "cluster":
            pred = recover_cluster(pred, maps[adm], num_asset)
        else:
            pred_cov = recover(pred, num_asset)
        
    exp_returns = np.zeros(num_asset)

    r = []
    r_truth = []
    r_prev = []
    for i in range(truth_cov.shape[0]):
        pred_t = global_min_var(pred_cov[i], exp_returns)
        w_pred = pred_t['w_g']
        r.append(np.matmul(w_pred.T,np.matmul(truth_cov[i], w_pred)))

        t = global_min_var(truth_cov[i], exp_returns)
        r_truth.append(t['var'])
        
        prev_t = global_min_var(prev_cov[i], exp_returns)
        w_prev = prev_t['w_g']
        r_prev.append(np.matmul(w_prev.T,np.matmul(truth_cov[i], w_prev)))

    r_p = sum(r)/len(r)
    r_m = sum(r_prev)/len(r_prev)
    return r_p, r_p/r_m