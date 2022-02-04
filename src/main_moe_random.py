# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:29:52 2021

@author: ZHU Haoren
"""
### Server code ###

###################
from tqdm import tqdm
import os
import random
from encoder import Encoder
from decoder import Decoder
from model import *
from net_params import *
from utils import *
import torch
from torch import nn 
from torch.optim import lr_scheduler 
import torch.optim as optim 
from earlystopping import EarlyStopping 
import numpy as np 
from dataloader import * 
from lr_scheduler import GradualWarmupScheduler
from build_portfolio import *
# from Balanced_DP import BalancedDataParallel
import wandb
from preprocess import *
import argparse

from sweep import *
import lstm_encoder_decoder
#from e3d_train import E3D_train

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--num_asset',default=32,type=int)
parser.add_argument('--n_clusters',default=12,type=int)
parser.add_argument('--batch_size',default=512,type=int,help='mini-batch size')
parser.add_argument('--input_length',default=10,type=int)
parser.add_argument('--input_gap',default=21,type=int)
parser.add_argument('--horizon',default=21,type=int)
parser.add_argument('--random_select',default=False,type=bool)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--dir_folder', type=str, default="")
parser.add_argument('--save_model', type=boolean_string, default=True)
parser.add_argument('--method', type=str, default="raw")
parser.add_argument('--dilated', type=int, default=0)
parser.add_argument('--n_e', type=int, default=8)
parser.add_argument('--top_k', type=int, default=4)
parser.add_argument('--adm', type=int, default=1)
parser.add_argument("--lr_sche_type", type=str, default = "grad_cos")
parser.add_argument("--project_name", type=str, default = "hyper")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--early_stop_patience", type=int, default=20)
parser.add_argument("--lag_t", type = int, default=42)
parser.add_argument("--filter_size", type=int, default=5)
parser.add_argument("--dataset", type=str, default='us')
parser.add_argument("--data_dir", type=str, default='/tmp2/syliu/adnn/nk_picks')
parser.add_argument("--gpu_id",type=str, default="0")
parser.add_argument("--group_id", type=int, default= 0)
args = parser.parse_args()

print("args.save_model", args.save_model)

default_config = dict(
    lr=args.lr,
    lr_patience = 6, 
    input_length=args.input_length,
    output_length = 1,
    input_gap = args.input_gap,
    filter_size = args.filter_size,
    early_stop_patience = args.early_stop_patience,
    normal = True,
    cov_scale = "log",
    optimizer = "adam",
    noisy_gating = False,
    num_experts = args.n_e,
    top_k = args.top_k,
    lag_t = args.lag_t,
    epochs = args.epochs,
    num_asset = args.num_asset,
    horizon = args.horizon,
    dilated = args.dilated, # 0: no dilation, 1: (1,2,5), 2: (1,2,3)
    method=args.method,
    random_select = args.random_select, 
    save_model = args.save_model,
    dir_folder = args.dir_folder,
    batch_size = args.batch_size,
    adm = args.adm,
    lr_sche_type = args.lr_sche_type,
    n_clusters = args.n_clusters,
    dataset = args.dataset,
    g_id = args.group_id,
    wandb_name = args.dir_folder
    )

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

# Device configuration
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def get_cluster_result(config):
    """
    Preprocessing the dataset based on the cluster configuration
    """
    print("clustering data")
    # closes = np.load("../close_price.npy")
    if config.dataset == 'us':
        print("us")
        closes = np.load("./Data/close_price.npy")
        closes = log_price(closes)[1:] - log_price(closes)[:-1]
        
    if config.random_select == False:
        print("first 32")
        returns = closes[:,:config.num_asset]
        pick = np.arange(config.num_asset)
    else:
        g_id = config.g_id
        print("running group:"+str(g_id))
        picks = np.load("./Data/"+config.dataset+"_picks.npy")
        pick = picks[g_id]
        print(pick)
        returns = closes[:,pick]
    
    num_asset = config.num_asset
    lag_t = config.lag_t
    input_gap = config.input_gap
    rebalance = config.horizon
    input_length = config.input_length
    output_length = config.output_length
    normal = config.normal
    result, maps = cluster_preprocess(returns, num_asset, lag_t, input_length=input_length,\
                                      input_gap=input_gap, rebalance=rebalance, output_length = output_length, \
                                      normal = normal, low = 0, upper = 1, n_clusters = config.n_clusters)
    return result, maps

def get_lstm_result(config):
    """
    Preprocessing the dataset based on the lstm input format
    """
    # closes = np.load("../close_price.npy")
    if config.dataset == 'us':
        print("us")
        closes = np.load("./Data/close_price.npy")
        closes = log_price(closes)[1:] - log_price(closes)[:-1]

    if config.random_select == False:
        print("first 32")
        returns = closes[:,:config.num_asset]
        pick = np.arange(config.num_asset)
    else:
        g_id = config.g_id
        print("running group:"+str(g_id))
        picks = np.load("./Data/"+config.dataset+"_picks.npy")
        pick = picks[g_id]
        print(pick)
        returns = closes[:,pick]
    
    num_asset = config.num_asset
    lag_t = config.lag_t
    input_gap = config.input_gap
    rebalance = config.horizon
    input_length = config.input_length
    output_length = config.output_length
    normal = config.normal
    result = lstm_preprocess(returns, num_asset, lag_t, input_length=input_length,input_gap=input_gap, \
                          rebalance=rebalance, output_length = output_length, normal = normal, low = 0, upper = 1)
    return result

def get_preprocess_result(config):
    """
    Generate standard sequences based on the configuration. 
    
    If random_select is true, will re-sample a group of assets and generate 
    corresponding ADMs. Otherwise, the first n assets will be selected to 
    generate the ADMs.
    """
    if config.dataset == 'us':
        print("us")
        closes = np.load("./Data/close_price.npy")
        closes = log_price(closes)[1:] - log_price(closes)[:-1]
  
    # Change the way to sample assets
    # Default way is to pick the fisrt k assets
    if config.random_select == False:
        print("first 32")
        returns = closes[:,:config.num_asset]
        pick = np.arange(config.num_asset)
    else:
        g_id = config.g_id
        print("running group:"+str(g_id))
        picks = np.load("./Data/"+config.dataset+"_picks.npy")
        pick = picks[g_id]
        print(pick)
        returns = closes[:,pick]

    
    # print(returns.shape)
    num_asset = config.num_asset
    lag_t = config.lag_t
    input_gap = config.input_gap
    rebalance = config.horizon
    input_length = config.input_length
    output_length = config.output_length
    normal = config.normal
    
    if config.method[:3] == "org":
        result = cor_preprocess(returns, num_asset, lag_t, input_length=input_length,input_gap=input_gap, \
                          rebalance=rebalance, output_length = output_length, normal = normal, low = 0, upper = 1)
    
    elif config.method == "cluster":
        print("preprocessing clustering data")
        result, maps = cluster_preprocess(returns, num_asset, lag_t, input_length=input_length,input_gap=input_gap, \
                          rebalance=rebalance, output_length = output_length, normal = normal, low = 0, upper = 1)
    else:
        result = final_preprocess(returns, num_asset, lag_t, input_length=input_length,input_gap=input_gap, \
                          rebalance=rebalance, output_length = output_length, normal = normal, low = 0, upper = 1)
    
    return result,pick

def train(config):
    print(config)
    '''
    main function to run the training
    '''
    # Get input data pack
    # Notice that different methods have different formats of input data
    if config.method == "cluster":
        result, maps = get_cluster_result(config)
    elif config.method == "lstm":
        print('preprocess_lstm_data')
        result = get_lstm_result(config)
    else:
        result, pick = get_preprocess_result(config)
        
    # Decide type of ADM to predict
    # 2: Inverse covariance (not implemented)
    if config.adm == 1: # Correlation
        matrics = result[1]
    else: # Covariance
        matrics = result[0]
    
    # Construct dataset and dataloader
    train_dataset = DatasetPrice(matrics[0])
    valid_dataset = DatasetPrice(matrics[1])
    test_dataset = DatasetPrice(matrics[2])
    
    train_data_loader = DataLoaderPrice(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    valid_data_loader = DataLoaderPrice(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_data_loader = DataLoaderPrice(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Load encoder and decoder parameters
    net_params = NetParams(config)
    encoder_params = getattr(net_params, "encoder_params")
    decoder_params = getattr(net_params, "decoder_params")
    
    # Initialize encoder and decoder
    encoder = Encoder(encoder_params[0], encoder_params[1], config)
    decoder = Decoder(decoder_params[0], decoder_params[1], config)

    # Initialize model
    if config.method == "moe_0": # Linear MoE
        net = ED_simple_MOE(encoder, decoder, config.input_length, config.width_n_height, device, \
                            config.num_experts, config.top_k, config.noisy_gating)
    elif config.method == "moe_1": # Quadratic MoE
        print("ED_QUAD_MOE")
        print(config.num_experts)
        print(config.top_k)
        net = ED_QUAD_MOE(encoder, decoder, config.input_length, config.width_n_height, device, \
                            config.num_experts, config.top_k, config.noisy_gating)
    elif config.method == "moe_2": # Cubic MoE
        net = ED_CUBIC_MOE(encoder, decoder, config.input_length, config.width_n_height, device, \
                            config.num_experts, config.top_k, config.noisy_gating)
    elif config.method == "tran": # T-ConvLSTM
        net = ED_Tran(encoder, decoder, config.input_length, config.width_n_height)
    elif config.method == "pos_0": # P-ConvLSTM
        net = POS_MOE(encoder, decoder, config.input_length, config.width_n_height, device, \
                            config.num_experts, config.top_k, config.noisy_gating)
    elif config.method == "mlp_0": # Linear MLP
        net = ED_simple_MLP(encoder, decoder, config.input_length, config.width_n_height)
    elif config.method == "mlp_1": # Quadratic MLP
        print("ED_QUAD_MLP")
        net = ED_QUAD_MLP(encoder, decoder, config.input_length, config.width_n_height)
    elif config.method == "conv3d": # Conv-3d that output a sequence of ADMs
        net = Conv3d(config.input_length)
    elif config.method == "conv3d_1": # Conv-3d that output one ADM
        print("conv3d_1")
        net = Conv3d_1(config.input_length)
    elif config.method == "lstm": # LSTM
        print("lstm")
        net = lstm_encoder_decoder.lstm_seq2seq(input_size = 528, hidden_size = 1056, device = device)
    else: # Raw-ConvLSTM
        print("ConvLSTM")
        net = ED_RAW(encoder, decoder)

    
    net.to(device)

    # initialize the early_stopping object, and load existing model (disabled in multiple runs)

    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)
    print("config.save_model",config.save_model)
    early_stopping = EarlyStopping(patience=config.early_stop_patience, verbose=True, safe_model = config.save_model)

    # Initialize loss function and optimizer
    lossfunction = nn.MSELoss().to(device)
    if config.optimizer=='sgd': # SGD
        optimizer = optim.SGD(net.parameters(),lr=config.lr, momentum=0.9)
    else: # Adam
        optimizer = optim.Adam(net.parameters(),lr=config.lr)

    # Initialize learning rate scheduler
    if config.lr_sche_type == "reduce": # Reduce learning rate
        pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,\
                                                          patience=config.lr_patience,verbose=True)
    elif config.lr_sche_type == "grad_reduce": # Warmup + reduce
        after_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config.lr_patience,verbose=True)
        pla_lr_scheduler = GradualWarmupScheduler(optimizer, 5, 20,after_scheduler)
    elif config.lr_sche_type == "grad_cos": # Warmup + cosine annealing
        after_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
        pla_lr_scheduler = GradualWarmupScheduler(optimizer, 5, 20, after_scheduler)
    else:
        pla_lr_scheduler = GradualWarmupScheduler(optimizer, 5, 20,None)
        
    # Track the training, validation, and uniform loss 
    train_losses = []
    valid_losses = []
    uni_losses = 0
    # Track the average training and validation loss 
    avg_train_losses = []
    avg_valid_losses = []
    print("Start Training")
    
    # Watch for the project
    torch.cuda.empty_cache()
    wandb.watch(net, lossfunction, log="all", log_freq=10)
    example_ct = 0
    batch_ct = 0
    cur_epoch = 0
    for epoch in tqdm(range(cur_epoch, config.epochs)):
        ###################
        # train the model #
        ###################
        for i, (idx, inputVar, targetVar) in tqdm(enumerate(train_data_loader)):
            
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            # If the method is moe, extra aux_loss will be returned
            if config.method[:3] == "moe":
                pred, aux_loss = net(inputs)  # B,S,C,H,W
                loss = lossfunction(pred, label) + torch.mean(aux_loss)
                #print(aux_loss)
                loss_aver = loss.item()
            else: # Other kinds of model
                pred  = net(inputs)  # B,S,C,H,W
                loss = lossfunction(pred, label) 
                loss_aver = loss.item()
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10)
            optimizer.step()
           
            example_ct +=  len(targetVar)
            batch_ct += 1
            # Log information
            if ((batch_ct + 1) % 4) == 0:
                wandb.log({"epoch": epoch, "loss": loss_aver}, step=example_ct)
        
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            for i, (idx, inputVar, targetVar) in enumerate(valid_data_loader):
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                if config.method[:3] == "moe":
                    pred, aux_loss = net(inputs)  # B,S,C,H,W
                    loss = lossfunction(pred, label) + torch.mean(aux_loss)
                    loss_aver = loss.item()
                else:
                    pred  = net(inputs)  # B,S,C,H,W
                    loss = lossfunction(pred, label) 
                    loss_aver = loss.item()
                # record validation loss
                valid_losses.append(loss_aver)
                
                # Uniform evaluation will neglect the padding 0 at the end of input matrix
                uni_loss = uniform_evaluation(pred, label,config) 
                uni_losses += uni_loss
                


        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        uni_loss = uni_losses/len(valid_dataset)
        wandb.log({"val_loss": valid_loss},step=example_ct)
        wandb.log({"uniform_loss": uni_loss},step=example_ct)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        uni_losses = 0
        # Step the learning rate scheduler
        if config.lr_sche_type != "grad_cos":
            pla_lr_scheduler.step(metrics = valid_loss)  # lr_scheduler
        else:
            pla_lr_scheduler.step()
        
        # Early stop starts if the training warms up

        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss, model_dict, epoch, config.save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    ######################
    #   Test the model   #
    ######################
    test_losses = []
    test_uni_losses = 0
    prev_uni_losses = 0
    model = "model_params.pth.tar"

    # Load best model parameters
    model_info = torch.load(os.path.join(config.save_dir, model))
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    
    print("start testing")
    # Keep tracks of the prediction and labels
    if config.save_model == True:
        labels = []
        preds = []
    with torch.no_grad():
        net.eval()
        #t = tqdm(data_loader, leave=False, total=len(data_loader))
        print("start test data loader")
        for i, (idx, inputVar, targetVar) in enumerate(test_data_loader):
            ######################
            #   Test the model   #
            ######################
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            if config.method[:3] == "moe":
                pred, aux_loss = net(inputs)  # B,S,C,H,W
                loss = lossfunction(pred, label) + torch.mean(aux_loss)
                loss_aver = loss.item()
            else:
                pred  = net(inputs)  # B,S,C,H,W
                loss = lossfunction(pred, label) 
                loss_aver = loss.item()
            # record validation loss
            test_losses.append(loss_aver)
            
            prev_loss = uniform_evaluation(inputs[:,-1:], label, config)
            uni_loss = uniform_evaluation(pred, label,config) 
            test_uni_losses += uni_loss
            prev_uni_losses += prev_loss
            # Store the prediction and labels
            if config.save_model == True:
                pred = pred.cpu().detach().numpy() #B, 10, 12, 12
                #print(pred.shape)
                if len(preds) == 0:
                    preds = pred
                else:
                    preds = np.concatenate((preds, pred), axis=0)
                
                label = label.cpu().detach().numpy() #B, 10, 12, 12
                #print(pred.shape)
                if len(labels) == 0:
                    labels = label
                else:
                    labels = np.concatenate((labels, label), axis=0)
            
    torch.cuda.empty_cache()
    
    # Save metrics and prediction
    test_loss = np.average(test_losses)
    test_uni_loss = test_uni_losses/len(test_dataset)
    prev_uni_loss = prev_uni_losses/len(test_dataset)
    wandb.log({"test_average_loss": test_loss})
    wandb.log({"test uniform_loss": test_uni_loss})
    wandb.log({"previous uniform_loss": prev_uni_loss})
    gain = 1 - (test_uni_loss/prev_uni_loss)
    wandb.log({"gain": gain})
        
    # if config.method == "lstm":   
    #     return
    # elif config.method == "cluster":   
    #     r_pred, r_gain = get_pred_risk(preds, result, config.num_asset, config.adm, config.method, maps)
    # else:
    #     r_pred, r_gain = get_pred_risk(preds, result, config.num_asset, config.adm, config.method)

    # wandb.log({"risk": r_pred})
    # wandb.log({"risk gain":1-r_gain})

def model_pipeline(data_dir, project_name = "hyper"):
    with wandb.init(project=project_name, config=default_config):
        config = wandb.config 
        wandb.run.name = config.wandb_name
        wandb.run.save()
        config.vec_len = int(config.num_asset*(config.num_asset+1)/2)
        config.width_n_height = int(np.ceil(pow(config.vec_len,0.5)))
        save_dir = './save_model/' + config.dir_folder + '/'
        config.data_dir = args.data_dir
        config.save_dir = save_dir
        
        train(config)
        


        
if __name__ == "__main__":

    data_dir = args.data_dir
    model_pipeline(data_dir, args.project_name)

        