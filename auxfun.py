#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import re

from scipy.stats import spearmanr
import torch

import networkmetrics

def group_shuffle(X, Y, indexes):
    '''Shuffle the rows of X and Y by keeping row with the same index grouped
    
    Input
    -----
    X: ndarray of shape (M,N) 
    
    Y: ndarray of shape (M,K)
    
    indexes: ndarray of shape (M,) of int that group rows. E.g., rows with 
        1 in the indexes array are the same contigouus group of observations
        to be kept together durign shuffling.
    
    Output
    ------
    X: The shuffled X
    
    Y: The shuffled Y
    
    indexes: The shuffled indexes    
    '''
    unique_indexes = np.unique(indexes)
    unique_indexes = unique_indexes[np.random.permutation(len(unique_indexes))]
    rearrange_idx = None
    for i in range(0, len(unique_indexes)):
        idx = np.where(unique_indexes[i] == indexes)[0]
        if rearrange_idx is None:
            rearrange_idx = idx
        else:
            rearrange_idx = np.hstack((rearrange_idx, idx))
        
    X = X[rearrange_idx, :]
    Y = Y[rearrange_idx, :] 
    indexes = indexes[rearrange_idx]
    
    return X, Y, indexes
        
def combo_params(params):
    '''Create a list with tuples denoting all possible combos of values to run
    the model with. Parameters are specified in the dictionary params
    
    Input
    -----
    params: dict, specifying the names and values of the parameters. 
        Values of the dict are lists of the params (a list of str, int, float)
        
    Output
    ------
    all_combos: list of tuples, each tuple corresponding to one unique
        combination of the values of the dict params.
        
    all_keys: list of str, denoting the names of the parameters and the 
        position that they occupy in each tuple in all_combos.
        For instance, parameter values in all_combos[0][0] correspond to param
        with name all_keys[0]
    '''
    all_values = []
    all_keys = []
    # assemble the values of the dictionaries in a list 
    for value in params.values():
        all_values.append(value)
        
    # assemble the keys of the dictionaries in a list 
    for keys in params.keys():                                                 
        all_keys.append(keys)
                 
    all_combos = list(itertools.product(*all_values))
    
    return all_combos, all_keys
           
def map_weights_to_template(w_template = None, w_to_map = None):
    '''Reorder the values of w_to_map in such a way that the valeus obey the 
    rank ordering of the values in w_template.
    The result is X so that: 
        rank order of X[i,j] == rank order of w_template[i,j] where 
        i,j in X[i,j] belongs to all non_zeros values in w_to_map 
     
    Input
    -----
    w_template: torch.Tensor tensor of size (N,M) specifying the rank order 
        of values to be used as reference for reordering the valeus of the 
        w_to_map tensor.
    
    w_to_map: torch.Tensor tensor of size (N,M) containing the values to be 
        reordered so that their rank ordering matches the rank ordering of the
        corresponding values of w_template.
        
    Output
    ------
    X: torch.Tensor tensor of size (N,M) with the reordered values of w_to_map
        such that:
            rank order of X[i,j] == rank order of w_template[i,j] where 
            i,j in X[i,j] belongs to all non_zeros values in w_to_map 
    '''
    X = torch.zeros(w_template.shape)
    idx = torch.where(w_template!=0)
    w_template_values = w_template[idx]
    w_to_map_values = w_to_map[idx]
    
    (sorted_w_template, 
    sorted_index_w_template) = torch.sort(w_template_values, 
                                          dim = 0, 
                                          descending=True)
    
    (sorted_w_to_map, 
     sorted_index_w_to_map) = torch.sort(w_to_map_values, 
                                         dim = 0, 
                                         descending=True)
        
    X[idx[0][sorted_index_w_template], 
      idx[1][sorted_index_w_template]] = sorted_w_to_map                                               
                                                   
    return X   

# Auxiliary function to get the desired parameters from the model
# model: the model from which we should fetch parameters 
# params_to_get: a list of str specifying the names of the params to be fetched     
def get_model_params(model, params_to_get = None):
    '''Extracts the parameters, names, and 'requires gradient' status from a 
    model.
    
    Input
    -----
    model: class instance based on the base class torch.nn.Module
    
    params_to_get: list of str, default=None, specifying the names of the 
        parameters to be extracted.
        If None, then all parameters and names of parameters from the model 
        will be extracted
    
    Output
    ------     
    params_name:, list, contaning one str for each extracted parameter
    
    params_values: list, containg one tensor corresponding to each 
        parameter. NOTE: The tensor is detached from the computation graph 
    req_grad_orig: list, containing one Boolean variable for each parameter
        denoting the requires_grad status of the original tensor/parameter 
        of the model     
    '''    
    params_names = []
    params_values = [] 
    req_grad_orig = []
    for name, param in zip(model.named_parameters(), model.parameters()):             
        if params_to_get is not None:
            if name[0] in params_to_get: 
                params_names.append(name[0])
                params_values.append(param.detach().clone())
                req_grad_orig.append(param.requires_grad)
        else:
            params_names.append(name[0])
            params_values.append(param.detach().clone())
            req_grad_orig.append(param.requires_grad)
                       
    return params_values, params_names, req_grad_orig

# Freeze (update=False) or unfreeze (update=True) model params
def freeze_params(model, 
                  params_to_freeze = None,
                  update = True):  
    '''Freeze or unfreeze the parametrs of a model
    
    Input
    -----
    model:  class instance based on the base class torch.nn.Module
    
    params_to_freeze: list of str specifying the names of the params to be 
        frozen or unfrozen
        
    update: bool, default True, specifying the freeze (update=False) or 
        unfreeze (update=True) of model params 
        
    Output
    ------
    model: class instance based on the base class torch.nn.Module with changed
        requires_grad param for the anmes params in params_to_freeze
        (freeze = requires_grad is False unfreeze = requires_grad is True)   
    '''
    for name, param in zip(model.named_parameters(), model.parameters()):             
        if params_to_freeze is not None:
            if name[0] in params_to_freeze: 
                param.requires_grad = update 
        else:
            param.requires_grad = update 
    
    return model

def calc_accuracy(output = None, labels = None):
    '''Classification accuracy calculation as acc = (TP + TN) / nr total pred
    
    Input
    -----
    output: torch.Tensor tensor of size (N,M) where N are the observations and 
        M the classes. Values must be such that highest values denote the 
        most probable class prediction.
    
    labels: torch.Tensor tensor of size (N,) of int denoting for each of the N
        observations the class that it belongs to, thus int must be in the 
        range 0 to M-1
    
    Output
    ------
    acc: float, accuracy of the predictions    
    '''
    _ , predicted = torch.max(output.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc = 100*(correct/total)

    return acc

def save_model_state(model, 
                     epoch = None, 
                     iteration = None,
                     folder_name = None):
       '''Save the model's state dict
       
       Input
       -----
       model: class instance based on the base class torch.nn.Module
       
       epoch: positive int, denoting the training epoch in which the model to 
           be saved is in
       
       iteration: positive int, denoting the iteration in which the model to be 
           saved is in. Iteration is 1 complete training cycle of the model
           across N epochs.
           NOTE: This should not be confused with the number of times a batch 
           of data passed through the algorithm (also called iteration).
       
       folder_name: object of class pathlib.PosixPath
           The folder in which the model state dict will be stored 
       '''
       file_name = 'model_state_dict' + '_epoch_' + str(epoch) + '_iter_' + str(iteration) + '.pth'
       file_to_save = folder_name / file_name
       torch.save(model.state_dict(), file_to_save)   
       
# Load pretrained model
def load_pretrained(model,
                    pretrained_folder = None, 
                    epoch = 0,
                    it = 0,
                    combo_nr = 0):
    '''Load the model's state dict
       
    Input
    -----
    model: class instance based on the base class torch.nn.Module
    
    pretrained_folder: object of class pathlib.PosixPath
        The folder in which the model state dict will be stored 
           
    epoch: positive int, denoting the training epoch in which the model to 
        be saved is in
       
    it: positive int, denoting the iteration in which the model to be 
        saved is in. Iteration is 1 complete training cycle of the model
        across N epochs.
        NOTE: This should not be confused with the number of times a batch 
        of data passed through the algorithm (also called iteration).
    
    combo_nr: positive int, denoting the nr of the parameter combination
        that corresponds to the trained stored model. 
        The int value can be arbitrary with the only constrained that it is 
        unique, that is, each int denotes one stored model trained 
        with a unique combination of parameters.
        
    Output
    ------
    model: class instance based on the base class torch.nn.Module with the 
        stored state dict      
    '''    
    file_name = 'model_state_dict_epoch_' + str(epoch) + '_iter_' + str(it) + '.pth'
    file_to_open = pretrained_folder / str(combo_nr) / file_name
    
    model.load_state_dict(torch.load(file_to_open))    
    
    return model           
       
# Scale tensor to [0 1] by takin into account the global min and max
def scale_tensor(X, global_scaling = True, epsilon = 1e-12):
    '''Scale tensor to [0 1] by takin into account the global min and max 
    (global_scaling=True) or the row-wise min max (global_scaling=False)
    
    Input
    -----
    X : torch.Tensor tensor os size (N,M), the tensor to be rescaled to [0 1]
        
    global_scaling: bool, default True
        Boolean variable specifying if the scaling should be performed by 
        taking into account the global min and max values (default)
        
    epsilon: float (default: 1e-12) 
        A small number to avoid potential divisions with 0  
        
    Output
    ------
    X_norm: torch.Tensor tensor os size (N,M) which is the rescaled tensor X   
    '''
    if global_scaling is True:
        min_val = torch.min(X)
        max_val = torch.max(X)
        denom = torch.clamp(max_val-min_val, min = 2*epsilon)
        nom = X-min_val
        X_norm = torch.div(nom+epsilon, denom)
    else:
        min_val = torch.min(X, dim=1)[0]
        max_val = torch.max(X, dim=1)[0]
        denom = torch.clamp(max_val-min_val, min = 2*epsilon)
        nom = X.T-min_val
        nom = nom.T + epsilon
        X_norm = torch.div(nom.T, denom)
        X_norm = X_norm.T
        
    return X_norm   

def concatenate_arrays(master_container = None, 
                       leech = None, 
                       mode = 'h'):
    '''Concatenate ndarrays vertically or horizontally
    
    Input
    -----
    master_container: ndarray, of shape (M,N), default None, that each time 
        changes from a concatanation with leech. When None, then the leech will
        become the master_container
        
    leech: ndarray, of shape (K,L) that will be concatenated to 
        master_container. The shape (K,L) must match to the shape (M,N) 
        dependong on whether the mode of concatenation is horizontal or 
        vertical. See np.hstack and np.vstack documentation. 
            
    mode: str, default 'h', specifying if the concatenation is horizontal ('h') 
        or vertical ('v').
    
    Output
    ------
    master_container: ndarray of shape (M,N+L) if mode='h' or (M+K,N) if 
        mode='v'.
        
    '''
    if master_container is not None:
        if mode == 'h':
            master_container = np.hstack((master_container, leech))
        if mode == 'v':
            master_container = np.vstack((master_container, leech))
    else:
        master_container = leech  
        
    return master_container  

def calculate_metrics(model,
                      file_to_model = None, 
                      metrics = [],
                      params_to_get = None):
    '''Compute network metrics on a specified layer of a given PyTorch model that 
    is stored.
    
    Input
    -----
    model: a class instance based on the base class torch.nn.Module. The 
        model must contain at least a recurrent layer (to be named explicitly
        in params_to_get)
        
    file_to_model: object of class pathlib.PosixPath specifying the full path
        to the stored model
    metrics: list of str, default [], with the metrics to be computed on each 
        reccurent layer.
        Currently to options:
            'hi': homophily index
            'sil': silhouette specifyign the clusterness of the recurrent
                mlayer as specified by kmeans
    params_to_get: str specifying the name of the reccurent layer to use
        for computing the metrics.
        
    Output
    ------
    all_metrics: a list containing the metrics    
    '''
    all_metrics = []# store all the computed metrics in a list
    
    # Load pretrained model
    model.load_state_dict(torch.load(file_to_model))
    values, names = get_model_params(model,
                                     params_to_get=params_to_get)#select what matrix of the model needs analysis
    w = values[0]#this is the matrix that we have to work with
    
    if 'hi' in metrics:
        print('Computing homophily...')
        hi = networkmetrics.calc_homophily(w.data.numpy())
        all_metrics.append(hi)
    
    if 'sil' in metrics:
        print('Computing silhouette...')
        scores, labels = networkmetrics.get_clusters(
                                                      w.data.numpy(), 
                                                      nr_cluster=[2, 3, 4, 5], 
                                                      metric='euclidean'
                                                      )
        all_metrics.append(scores)        
    
    return all_metrics

# Get the min loss
def min_loss(losses):
    '''Find min value in each row of ndarray losses
    
    Input
    -----
    losses: ndarray of shape (M,N) 
    
    Output
    ------
    all_losses: list containing the min value of each row of losses
    '''
    all_losses = []
    for i in range(losses.shape[0]):
        all_losses.append(np.min(losses[i, :]))
    
    return all_losses 
   
def min_loss_epoch(losses, perc = None):
    '''Get the index of min value for each row in losses.
    
    If a perc is specified, then the index of the min value in each row
    satisfies the following:
    ((losses-np.min(losses))/(np.max(losses)-np.min(losses)))*100 <= 100-perc
    
    Input
    -----
    losses: ndarray of shape (M,N)
    
    perc:  int 0 < perc < 100   

    Output
    ------
    all_min_loss_ep: list with an idx for each row denoting where the 
        min value was observed (taking into account the perc params or not)        
    
    '''
    all_min_loss_ep = [] 
    if perc is not None: perc = 100-perc      
    for i in range(losses.shape[0]):
            if perc is None:
                value = np.min(losses[i, :])
                all_min_loss_ep.append(
                                       np.where(value == losses[i, :])[0][0]#get the integer value of the index/epoch
                                       )                
            else:
                val_perc = ((losses[i, :]-np.min(losses[i, :]))/(np.max(losses[i, :])-np.min(losses[i, :])))*100
                idx = np.where(val_perc <= perc)[0]
                all_min_loss_ep.append(np.min(idx))
                
    return all_min_loss_ep    

def reshape_to_vector(x):
    '''Reshape an ndarray x
    
    Input
    -----
    x: ndarray of shape (M,N)
    
    Output
    ------
    x_reshaped: the reshaped x ndarray
    
    see np.reshape documentation  
    '''
    x_reshaped =np.reshape(x, 
                          (x.size), 
                           'C')

    return x_reshaped 

# Read the results and extract desired quantiities
def read_results(results_folder = None, 
                 results_id = None,
                 start = None,
                 stop = None):
    #Dict to store all raw results
    raw_results = {}
    
    #Dict to store all quantities calculated on raw results
    quantities_on_results = {}
    
    # Get metrics/loss
    file_name = 'train_loss_all.npy'
    file_to_open = results_folder / str(results_id) / file_name
    train_loss_all= np.load(file_to_open)
    
    # The shape of the results indicate the epochs and iterations
    iterations = train_loss_all.shape[0]
    total_epochs = train_loss_all.shape[1]
    
    if stop is None:
        stop = total_epochs
    
    file_name = 'validate_loss_all.npy'
    file_to_open = results_folder / str(results_id) / file_name
    validate_loss_all = np.load(file_to_open)

    current_min_loss = min_loss(validate_loss_all[:, start:stop])
    
    # get min epoch for loss only for validation
    current_min_loss_ep = min_loss_epoch(validate_loss_all[:, start:stop], 
                                         perc=None)
    
    # get min epoch for 99% loss only for validation
    current_min_loss_ep_perc = min_loss_epoch(validate_loss_all[:, start:stop], 
                                              perc=99)
    
    # Check if a file corresponding to metrics exists and is not empty
    # (maybe stored empty) and if so, set the boolean value load_metrics to True
    load_metrics = False
    file_name = 'train_metrics_all.npy'
    file_to_open = results_folder / str(results_id) / file_name
    metrics_exists = os.path.exists(file_to_open)
    
    if metrics_exists:
        train_metrics_all = np.load(file_to_open)
        if train_metrics_all.size > 0:
            load_metrics = True
    
    # Get metrics if load_metrics is True
    if load_metrics:
        file_name = 'train_metrics_all.npy'
        file_to_open = results_folder / str(results_id) / file_name
        train_metrics_all = np.load(file_to_open)
        
        file_name = 'validate_metrics_all.npy'
        file_to_open = results_folder / str(results_id) / file_name
        validate_metrics_all = np.load(file_to_open)
        
    #Store results - raw results    
    raw_results = {
                  'train_loss': train_loss_all[:, start:stop],
                  'validate_loss': validate_loss_all[:, start:stop]
                  }
    
    if load_metrics:
        raw_results['train_metrics'] = train_metrics_all[:, start:stop]
        raw_results['validate_metrics'] = validate_metrics_all[:, start:stop]
            
    #Store results - quantities calculated on raw results
    quantities_on_results = {
                            'min_loss': current_min_loss,
                            'min_loss_ep': current_min_loss_ep,
                            'min_loss_ep_perc': current_min_loss_ep_perc
                            }        
    
    ep = range((stop-start))
    
    return raw_results, quantities_on_results, ep, iterations 

def extend_list(list_to_ext = None, ext = None):
    '''Extend a list of lists as follows:
    Construct a new list of lists ext_list such that the first list of 
    ext_list is a list of li[n]*ext where li is the ith list in list_to_ext
    and n is the nth item of list li. The construction of ext_list proceeds
    from n=0..N-1 where N is the length of li.
    
    Hence all lists li in list_to_ext must have the same length. 

    Example:
    a = [['apple','carrot'],['grape','orange']]
    ext_list = extend_list(list_to_ext=a, ext=5)

    ext_list=[
      ['apple',
       'apple',
       'apple',
       'apple',
       'apple',
       'grape',
       'grape',
       'grape',
       'grape',
       'grape'],
      ['carrot',
       'carrot',
       'carrot',
       'carrot',
       'carrot',
       'orange',
       'orange',
       'orange',
       'orange',
       'orange']
      ] 
       
    list_to_ext: a list of lists to be expanded
    
    ext: int, positive, denoting the amount of expansion of each list item
        li[n]*ext
        
    ext_list: the expanded list with the "expanded" structure explained above.    
    '''
    ext_list = []
    for i, combo in enumerate(list_to_ext):
        for c, item in enumerate(combo):
           if i==0:
               ext_list.append([item] * ext)
           else:
               new_item = [item] * ext
               ext_list[c] = ext_list[c] + new_item
    
    return ext_list

# Clean string from special characters
def clean_str(dirty_string):
    '''Clean string from special characters and return a list with the
    clean strings. This is a tailored cleaning that corresponds to a 
    specific input string format.
    
    Example:
    dirty_string= "(0.1, 'sign', 'of course', 4)\n" 
    clean_strings = clean_str(dirty_string)
    clean_strings -> ['0.1', 'sign', 'of course', '4']
    '''
    clean_strings = re.sub("'", "", dirty_string)# remove '   
    clean_strings = clean_strings.replace("(","")# remove parentheses
    clean_strings = clean_strings.replace(")","")
    clean_strings = clean_strings.split(',')# split strings
    clean_strings = [i.strip() for i in clean_strings]# remove whitespaces
    
    return clean_strings

def get_activation_model(model, 
                         data_generator = None, 
                         device = 'cpu'): 
    '''Obtain the activations of the last hidden layer of an Elman RNN.
    
    Input
    -----
    model: an RNN model, an instance of class nn.Module
    data_generator: data generator, torch.utils.data.dataloader.DataLoader,
        that feeds the model data to get the  activations 
    device: str, specifying the device used for performing the forward pass
        'cpu' or 'gpu'
        
    Output
    ------
    all_hidden: list of tensors, len N, of shape (B,T,H)
        N depends on the batch size (data generator parameter) and the 
        number of data
        B is the batch size, T the time dimension of the data and H are the 
        nr of hidden units of the reccurent networks.
        NOTE that (B,T,H) corresponds to batch_first is True
        otheriwise the shape of the tensors is (T,B,H)
    '''        
    all_hidden = []
    # Make sure that the model is not at the training mode
    model.train(False)
    for X_batch, Y_batch in data_generator:
        # Send tensors to the device used
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device) 
        output, hidden = model(X_batch)
        # Output is hidden state for eact timepoint - that is what we need
        # to keep in a list
        all_hidden.append(output.data)
                
    return all_hidden

def compute_coactivation(batch_time_activations):
    '''Compute coactivations based on activity in the ndarrray 
    batch_time_activations  
    
    Input
    -----
    batch_time_activations: ndarray of shape (B,T,H) with: 
        B the batch size
        T the nr of time steps
        H the activations
        
    Output
    ------
    coactivations: ndarray of shape (B,H,H) 
        this is the coactivation matrix (H,H) for each datapoint in the batch
        
    '''
    all_coact_np = np.zeros((batch_time_activations.shape[0],
                             batch_time_activations.shape[2],
                             batch_time_activations.shape[2])
                            )
    for item in range(batch_time_activations.shape[0]): 
        res = spearmanr(batch_time_activations[item])
        current_coact = res[0]
        all_coact_np[item,:,:] = current_coact 
    
    return all_coact_np

def get_mean_coactivation(all_hidden):
    '''Compute mean coactivations based on activity in the list of tensors 
    all_hidden  
    
    Input
    -----
    all_hidden: list of len N containing tensors of shape (B,T,H) with: 
        B the batch size
        T the nr of time steps
        H the activations
        
    Output
    ------
    mean_coactivations: ndarray of shape (H,H) 
        this is the mean coactivation matrix (H,H) across N datapoint in the 
        batch
    
    '''
    all_coactivations = None
    for hidden in all_hidden:
        coact = compute_coactivation(hidden.data.numpy())   
        mean_coact = np.mean(coact, axis=0)
        idx = np.where(~np.eye(mean_coact.shape[0], dtype=bool))#get all but the diagonal elements
        coactivations = mean_coact[idx]
        all_coactivations = concatenate_arrays(master_container=all_coactivations,
                                               leech=coactivations,
                                               mode='v')
    mean_coactivations = np.mean(all_coactivations, axis=0)
    
    return mean_coactivations 
        
        