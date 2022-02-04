# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 20:09:08 2021

@author: ZHU Haoren
"""
import numpy as np
import math
import os
from sklearn.cluster import KMeans


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Stationary transformation by log i.e. linear stationary
def log_price(closes):
    return np.log(closes)

def transform_cov_image(closes, lag_t, tril = True):
    matrixs = []
    length = closes.shape[0] 
    for i in range(length):
        if i < lag_t:
            continue
        else:
            sub_closes = closes[i - lag_t: i,:]
            matrix = np.cov(np.transpose(sub_closes))
            if tril == True: # Cut the matrix by the diagonal axis and transform it to a vector
                vec = matrix[np.triu_indices(matrix.shape[0],0)]
                matrixs.append(vec)
            else:
                matrixs.append(matrix)
    return np.array(matrixs)

def normalize_cor_image(image, low = 0, upper = 2):
    return (image + 1) * (upper - low)/2 + low

def transform_cor_image(closes, lag_t, normal=False, low = 0, upper = 1, tril = True):
    # Maintain diagonal
    corr_matrixs = []
    length = closes.shape[0] 
    for i in range(length):
        if i < lag_t:
            continue
        else:
            sub_closes = closes[i - lag_t: i,:]

            corr_matrix = np.corrcoef(np.transpose(sub_closes))
            if True in np.isnan(corr_matrix):
                corr_matrixs.append(corr_matrixs[-1])
            else:
                if normal == True:
                    corr_matrix = normalize_cor_image(corr_matrix, low, upper)
                if tril == True: # Cut the correlation matrix by the diagonal axis and transform it to a vector
                    corr_vec = corr_matrix[np.triu_indices(corr_matrix.shape[0],0)]
                    corr_matrixs.append(corr_vec)
                else:
                    corr_matrix[np.tril_indices(corr_matrix.shape[0],-1)] = 0
                    corr_matrixs.append(corr_matrix)
    return np.array(corr_matrixs)

def simulate_cor_preprocess(train_returns,validate_returns,test_returns, num_asset, lag_t, input_length=10, input_gap = 1,output_length=1, \
                     output_gap=1,rebalance=21, normal = True, low = 0, upper = 1):
    if normal == False:
        train_cors = transform_cor_image(train_returns,lag_t)
        validate_cors = transform_cor_image(validate_returns,lag_t)
        test_cors = transform_cor_image(test_returns,lag_t)
    else:
        train_cors = transform_cor_image(train_returns, lag_t, normal,low, upper)
        validate_cors = transform_cor_image(validate_returns, lag_t, normal,low, upper)
        test_cors = transform_cor_image(test_returns, lag_t, normal,low, upper)

    def get_frame(sequences, input_length=10, input_gap=1, output_length=1, \
                  output_gap=1, rebalance=21):
        data = []
        shape = sequences.shape
        input_span = (input_length-1)*input_gap+1
        num_data = shape[0] - input_span - rebalance + 1
        height = int(np.ceil(pow(shape[1],0.5)))
        
        for i in range(num_data):
            # Get w/o padding input frames and input frames idx
            input_frames = []
            frame_idx = []
            for j in range(input_length):
                frame_idx.append(i+j*input_gap)
                input_frames.append(sequences[i+j*input_gap])
            
            # Get padding input frames
            input_frames_extend = []
            for image_vector in input_frames:
                image_vector_extend = np.zeros(height*height)
                image_vector_extend[:len(image_vector)] = image_vector
                input_frames_extend.append([image_vector_extend.reshape(height,height)]) # add [] represent channel
            
            # Get w/o padding output frames
            output_frames = []
            output_frames_start_idx = i+input_span+rebalance-1
            for j in range(output_length):
                output_frames.append(sequences[output_frames_start_idx+j*output_gap])
            
            # Get padding output frames
            output_frames_extend = []
            for image_vector in output_frames:
                image_vector_extend = np.zeros(height*height)
                image_vector_extend[:len(image_vector)] = image_vector
                output_frames_extend.append([image_vector_extend.reshape(height,height)])

            data.append([frame_idx, input_frames_extend, output_frames_extend])

        data = np.array(data,dtype=object)
        return data

    train_cors_result = get_frame(train_cors,input_length, input_gap, output_length, output_gap,rebalance)

    validate_cors_result = get_frame(validate_cors, input_length, input_gap, output_length, output_gap, rebalance)
    test_cors_result = get_frame(test_cors,input_length, input_gap, output_length, output_gap,rebalance)

    return train_cors_result, validate_cors_result, test_cors_result


def cor_preprocess(returns, num_asset, lag_t, input_length=10, input_gap = 1,output_length=1, \
                     output_gap=1,rebalance=21, split = [0.85, 0.05, 0.1], normal = True, \
                     low = 0, upper = 1):
    """
    Get the origin correlation matrix
    """
    if normal == False:
        cors = transform_cor_image(returns,lag_t, tril=False)
    else:
        cors = transform_cor_image(returns, lag_t, normal,low, upper, tril = False)
        
    def get_frame(sequences, input_length=10, input_gap=1, output_length=1, \
                  output_gap=1, rebalance=21,split=[0.85, 0.05, 0.1]):
        data = []
        shape = sequences.shape
        input_span = (input_length-1)*input_gap+1
        num_data = shape[0] - input_span - rebalance + 1
        
        for i in range(num_data):
            # Get w/o padding input frames and input frames idx
            input_frames = []
            frame_idx = []
            for j in range(input_length):
                frame_idx.append(i+j*input_gap)
                input_frames.append(sequences[i+j*input_gap])
            input_frames = [[j] for j in input_frames]
            
            # Get w/o padding output frames
            output_frames = []
            output_frames_start_idx = i+input_span+rebalance-1
            for j in range(output_length):
                output_frames.append(sequences[output_frames_start_idx+j*output_gap])
            output_frames = [[j] for j in output_frames]
            
            data.append([frame_idx, input_frames, output_frames])

        data = np.array(data)
        train_split_index = int(len(data)*split[0])
        validate_split_index = train_split_index+int(len(data)*split[1])
        train_data = data[:train_split_index]
        validate_data = data[train_split_index:validate_split_index]
        test_data = data[validate_split_index:]
        return train_data,validate_data, test_data
    
    d_cors = get_frame(cors,input_length, input_gap, output_length, output_gap,rebalance,split)
    return [], d_cors

def final_preprocess(returns, num_asset, lag_t, input_length=10, input_gap = 1,output_length=1, \
                     output_gap=1,rebalance=21, split = [0.85, 0.05, 0.1], normal = True, \
                     low = 0, upper = 1, cov_scale = "log"):
    """
    This is the final version of handling preprocessing.
        
    params:
        returns:      the log returns of assets
        num_asset:    the number of assets
        lag_t:        the time lag to calculate the matrix
        input_length: the length of input sequence
        input_gap:    the distance between input matrix
        output_length:the length of output sequence
        output_gap:   the distance between output matrix
        rebalance:    the distance to predict the future matrix \
        (i.e. the distance between last input matrix and first output matrix)
        split:        the train, valid, test split ratio
        normal:       whether to normalize the correlation
        low:          lower boundary of normalization, used if normal set true
        upper:        upper boundary of normalization, used if normal set true
        
    returns:
        d_covs: covariance data in shape of [train, valid, test]
        d_cors: correlation data in shape of [train, valid, test]
    """
    # Get cov and cor matrices whole sequence
    # Generate cov data
    covs = transform_cov_image(returns,lag_t)
    if cov_scale == "log":
        covs = np.log(covs) # logarithm scaler, should be made as hyperparameters
    
    # Generate correlation data
    if normal == False:
        cors = transform_cor_image(returns,lag_t)
    else:
        cors = transform_cor_image(returns, lag_t, normal,low, upper)

    
    def get_frame(sequences, input_length=10, input_gap=1, output_length=1, \
                  output_gap=1, rebalance=21,split=[0.85, 0.05, 0.1]):
        data = []
        shape = sequences.shape
        input_span = (input_length-1)*input_gap+1
        num_data = shape[0] - input_span - rebalance + 1
        height = int(np.ceil(pow(shape[1],0.5)))
        
        for i in range(num_data):
            # Get w/o padding input frames and input frames idx
            input_frames = []
            frame_idx = []
            for j in range(input_length):
                frame_idx.append(i+j*input_gap)
                input_frames.append(sequences[i+j*input_gap])
            
            # Get padding input frames
            input_frames_extend = []
            for image_vector in input_frames:
                image_vector_extend = np.zeros(height*height)
                image_vector_extend[:len(image_vector)] = image_vector
                input_frames_extend.append([image_vector_extend.reshape(height,height)]) # add [] represent channel
            
            # Get w/o padding output frames
            output_frames = []
            output_frames_start_idx = i+input_span+rebalance-1
            for j in range(output_length):
                output_frames.append(sequences[output_frames_start_idx+j*output_gap])
            
            # Get padding output frames
            output_frames_extend = []
            for image_vector in output_frames:
                image_vector_extend = np.zeros(height*height)
                image_vector_extend[:len(image_vector)] = image_vector
                output_frames_extend.append([image_vector_extend.reshape(height,height)])


            data.append([frame_idx, input_frames_extend, output_frames_extend])

        data = np.array(data)
        train_split_index = int(len(data)*split[0])
        validate_split_index = train_split_index+int(len(data)*split[1])
        train_data = data[:train_split_index]
        validate_data = data[train_split_index:validate_split_index]
        test_data = data[validate_split_index:]
        return train_data,validate_data, test_data
    
    d_covs = get_frame(covs,input_length, input_gap, output_length, output_gap, rebalance,split)
    d_cors = get_frame(cors,input_length, input_gap, output_length, output_gap,rebalance,split)

    return d_covs, d_cors

def return_largest_element(array):
    largest = 0
    for x in range(0, len(array)):
        if(array[x] > largest):
            largest = array[x]
    return largest

def get_cluster(sequence, n_clusters = 12, split_index=-1):
    train_sequence = sequence[:split_index]

    labels = KMeans(n_clusters=n_clusters, random_state=0).fit(train_sequence.T).labels_

    new_cors = np.zeros(sequence.shape) 
    pos_idx = 0
    pos_map = []
    for i in range(return_largest_element(labels)+1):
        for j in range(labels.shape[0]):
            if labels[j] == i:
                new_cors[:,pos_idx] = sequence.T[j]
                pos_idx += 1
                pos_map.append(j)
    #new_cors = np.moveaxis(new_cors, -1, 0)
    return new_cors, pos_map

def cluster_preprocess(returns, num_asset, lag_t, input_length=10, input_gap = 1,output_length=1, \
                     output_gap=1,rebalance=21, split = [0.85, 0.05, 0.1], normal = True, \
                     low = 0, upper = 1, n_clusters = 12):
    """
    This is the final version of handling preprocessing.
        
    params:
        returns:      the log returns of assets
        num_asset:    the number of assets
        lag_t:        the time lag to calculate the matrix
        input_length: the length of input sequence
        input_gap:    the distance between input matrix
        output_length:the length of output sequence
        output_gap:   the distance between output matrix
        rebalance:    the distance to predict the future matrix \
        (i.e. the distance between last input matrix and first output matrix)
        split:        the train, valid, test split ratio
        normal:       whether to normalize the correlation
        low:          lower boundary of normalization, used if normal set true
        upper:        upper boundary of normalization, used if normal set true
        
    returns:
        d_covs: covariance data in shape of [train, valid, test]
        d_cors: correlation data in shape of [train, valid, test]
    """
    # Get cov and cor matrices whole sequence
    covs = transform_cov_image(returns,lag_t)
    if normal == False:
        cors = transform_cor_image(returns,lag_t)
    else:
        cors = transform_cor_image(returns, lag_t, normal,low, upper)

    
    def get_frame(sequences, input_length=10, input_gap=1, output_length=1, \
                  output_gap=1, rebalance=21,split=[0.85, 0.05, 0.1]):
        data = []
        shape = sequences.shape
        input_span = (input_length-1)*input_gap+1
        num_data = shape[0] - input_span - rebalance + 1
        height = int(np.ceil(pow(shape[1],0.5)))
        
        for i in range(num_data):
            # Get w/o padding input frames and input frames idx
            input_frames = []
            frame_idx = []
            for j in range(input_length):
                frame_idx.append(i+j*input_gap)
                input_frames.append(sequences[i+j*input_gap])
            
            # Get padding input frames
            input_frames_extend = []
            for image_vector in input_frames:
                image_vector_extend = np.zeros(height*height)
                image_vector_extend[:len(image_vector)] = image_vector
                input_frames_extend.append([image_vector_extend.reshape(height,height)]) # add [] represent channel
            
            # Get w/o padding output frames
            output_frames = []
            output_frames_start_idx = i+input_span+rebalance-1
            for j in range(output_length):
                output_frames.append(sequences[output_frames_start_idx+j*output_gap])
            
            # Get padding output frames
            output_frames_extend = []
            for image_vector in output_frames:
                image_vector_extend = np.zeros(height*height)
                image_vector_extend[:len(image_vector)] = image_vector
                output_frames_extend.append([image_vector_extend.reshape(height,height)])


            data.append([frame_idx, input_frames_extend, output_frames_extend])

        data = np.array(data)
        train_split_index = int(len(data)*split[0])
        validate_split_index = train_split_index+int(len(data)*split[1])
        train_data = data[:train_split_index]
        validate_data = data[train_split_index:validate_split_index]
        test_data = data[validate_split_index:]
        return train_data,validate_data, test_data
    
    c_input_span = (input_length-1)*input_gap+1
    c_num_data = len(covs) - c_input_span - rebalance + 1
    cluster_split = int(split[0]*c_num_data)
    covs, map_covs = get_cluster(covs, n_clusters, cluster_split)
    cors, map_cors = get_cluster(cors, n_clusters, cluster_split)

    d_covs = get_frame(covs,input_length, input_gap, output_length, output_gap, rebalance,split)
    d_cors = get_frame(cors,input_length, input_gap, output_length, output_gap,rebalance,split)

    return [d_covs, d_cors], [map_covs, map_cors]


def lstm_preprocess(returns, num_asset, lag_t, input_length=10, input_gap = 1,output_length=1, \
                     output_gap=1,rebalance=21, split = [0.85, 0.05, 0.1], normal = True, \
                     low = 0, upper = 1):

    covs = transform_cov_image(returns,lag_t)
    if normal == False:
        cors = transform_cor_image(returns,lag_t)
    else:
        cors = transform_cor_image(returns, lag_t, normal,low, upper)

    
    def get_frame(sequences, input_length=10, input_gap=1, output_length=1, \
                  output_gap=1, rebalance=21,split=[0.85, 0.05, 0.1]):
        data = []
        shape = sequences.shape
        input_span = (input_length-1)*input_gap+1
        num_data = shape[0] - input_span - rebalance + 1
        height = int(np.ceil(pow(shape[1],0.5)))
        
        for i in range(num_data):
            # Get w/o padding input frames and input frames idx
            input_frames = []
            frame_idx = []
            for j in range(input_length):
                frame_idx.append(i+j*input_gap)
                input_frames.append(sequences[i+j*input_gap])
            
            # # Get padding input frames
            # input_frames_extend = []
            # for image_vector in input_frames:
            #     image_vector_extend = np.zeros(height*height)
            #     image_vector_extend[:len(image_vector)] = image_vector
            #     input_frames_extend.append([image_vector_extend.reshape(height,height)]) # add [] represent channel
            
            # Get w/o padding output frames
            output_frames = []
            output_frames_start_idx = i+input_span+rebalance-1
            for j in range(output_length):
                output_frames.append(sequences[output_frames_start_idx+j*output_gap])
            
            data.append([frame_idx, input_frames, output_frames])

        data = np.array(data)
        train_split_index = int(len(data)*split[0])
        validate_split_index = train_split_index+int(len(data)*split[1])
        train_data = data[:train_split_index]
        validate_data = data[train_split_index:validate_split_index]
        test_data = data[validate_split_index:]
        return train_data,validate_data, test_data
    
    d_covs = get_frame(covs,input_length, input_gap, output_length, output_gap, rebalance,split)
    d_cors = get_frame(cors,input_length, input_gap, output_length, output_gap,rebalance,split)

    return d_covs, d_cors       

if __name__ == "__main__":
    closes = np.load("../close_price.npy")
    closes = log_price(closes)[1:] - log_price(closes)[:-1]
    num_asset = 32
    lag_t = 42
    input_gap = 10
    rebalance = 21
    input_length = 10
    output_length = 1
    normal = True
    returns = closes[:,:32]
    pick = np.arange(32)
    save_dir = '../data/' + "num_%i_lag_%i/"%(num_asset, lag_t)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+"pick.npy",pick)
    result = final_preprocess(returns, num_asset, lag_t, input_length=input_length,input_gap=input_gap, \
                          rebalance=rebalance, output_length = output_length, normal = normal, low = 0, upper = 1)
    np.save(save_dir+"result_gap_%i_horizon_%i.npy"%(input_gap, rebalance), result)