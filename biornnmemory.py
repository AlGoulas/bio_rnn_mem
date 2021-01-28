#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import configparser
import numpy as np
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils import data

from bio2art import importnet

import rnnmodels
import auxfun
import tasks
import modeltraineval
import networkmetrics

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

path_to_connectome_folder = Path(config['paths']['path_to_network'])
results_folder = Path(config['paths']['path_to_results'])
data_name = config['net']['net_name']
freeze_layer = config['trainvalidate'].getboolean('freeze_layer')
remap_w = config['trainvalidate'].getboolean('remap_w')
random_w = config['net'].getboolean('random_w')
rand_partition = config['net'].getboolean('rand_partition')
epochs = int(config['trainvalidate']['epochs']) 
iterations = int(config['trainvalidate']['iterations']) 
rnn_size =  int(config['net']['rnn_size'])  
nr_neurons = int(config['net']['nr_neurons'])  
pretrained_folder = config['paths']['pretrained']
if len(pretrained_folder)==0: 
    pretrained_folder = [] 
else: 
    pretrained_folder = Path(pretrained_folder)  
init = config['trainvalidate']['init'] 
trial_params = eval(config['trialparams']['trial_params'])
params_for_combos = eval(config['combosparams']['combos_params'])
# Pretrained parameters - which epoch and which iteration should be used to 
# load a pretrained model (specified by pretrained_folder)
# NOT used in the report of this project but added for completeness
pretrained_epoch = int(config['pretrainedparams']['pretrained_epoch']) 
pretrained_iteration = int(config['pretrainedparams']['pretrained_iteration'])    
           
# Add the "trial_matching":True to trial_params 
trial_params['trial_matching'] = True
# Decide on the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          
# Generators parameters
params_generators = {
                     'batch_size': 128,
                     'shuffle': False,
                     'num_workers': 0
                     }
 
neuron_density = np.zeros((rnn_size,), dtype=int)
neuron_density[:] = nr_neurons 

C, C_Neurons, Region_Neuron_Ids = importnet.from_conn_mat(
    data_name, 
    path_to_connectome_folder = path_to_connectome_folder, 
    neuron_density = neuron_density, 
    target_sparsity = 0.8,
    intrinsic_conn = True, 
    target_sparsity_intrinsic = 1.,
    rand_partition = rand_partition,
    keep_diag = False
    )

C_Neurons = torch.Tensor(C_Neurons).double()
C_Neurons.to(device)

# Spacify the common model parameters across tasks.
# Task-specific model parameters will be specified below.
model_params = {
               'hidden_dim': C_Neurons.shape[0],
               'random_w': random_w, #shuffle topology or not
               'w' : C_Neurons
               }
 
# Parameters for the model and trials automatically assigned based on the 
# set parameters above
if trial_params['task_name'] == 'pic_latent_mem':     
    # Update model params based on task 
    model_params['input_size'] = 51
    model_params['output_size'] = 3 
      
if trial_params['task_name'] == 'pic_mem':  
    # Update model params based on task
    model_params['input_size'] = 785
    model_params['output_size'] = 3 
          
if trial_params['task_name'] == 'nback_mem':  
    # Update model params based on task
    model_params['input_size'] = 2
    model_params['output_size'] = 3 
          
if trial_params['task_name'] == 'seq_mem':  
    # Update model params based on task
    model_params['input_size'] = 2
    model_params['output_size'] = 1
       
# Parameters for training that do not change (maybe place them in tuple in
# future updates) 
train_params_immutable = {
                          'epochs' : epochs,
                          'iterations' : iterations,
                         }

# Generate the list with all the combos of parameters to iterate
all_combos, all_keys = auxfun.combo_params(params_for_combos)

# Keep a file with the map of folders to combos
file_name = 'folder_index.txt'
file_to_open = results_folder / file_name
    
with open(file_to_open, 'w') as file:
    for combo_index, combination in enumerate(all_combos):
        file.write(str(combo_index) + ' ' + str(combination) + '\n')

# Calculate accuracy if we have to do with a decision task
if trial_params['task_name'] == 'nback_mem' or trial_params['task_name'] == 'pic_latent_mem':
    metrics = ['acc']
else:
    metrics = []

# Run the experiment   
# Loop through all the combos and train and validate the model with the given
# params dictated by each entry in the all_combos
for combo_index, combination in enumerate(all_combos):
    print('\nFreezing of layer | Weight re-mapping | Random w')
    print('\nFreeze of layer...:', freeze_layer)
    print('\nWeight re-mapping...:', remap_w)
    print('\nRandom w...:', random_w)       
    # mk dir for storing the results of the current combo - name the folder
    # based on the combo_index (0 ,1... nr of combos)
    # create a txt file to use as index to track which folder corresponds 
    # to what combination of parameters        
    os.makedirs(results_folder / str(combo_index))
    
    # Assign lr, nonlinearity, init and optimizer
    lr = combination[all_keys.index('lr')]
    nonlinearity = combination[all_keys.index('nonlinearity')]
    
    # Keep losses across iterations
    train_loss_all=[]
    validate_loss_all=[]
    
    # Initialize lists in case metrics (e.g., accuracy) are needed
    train_metrics_all=[]
    validate_metrics_all=[]
     
    if trial_params['task_name'] == 'seq_mem': 
        trial_params['pattern_length'] = combination[all_keys.index('pattern_length')]
        
    if (trial_params['task_name'] == 'pic_mem' 
        or trial_params['task_name'] == 'nback_mem' 
        or trial_params['task_name'] == 'pic_latent_mem'): 
        trial_params['n_back'] = combination[all_keys.index('n_back')]
      
    print('\nPerforming task...:',trial_params['task_name'])    
    print('\nTraining for combo nr...:', combo_index+1, '/', len(all_combos), 'combination...:', combination)
    # Train and test iter times
    for it in range(train_params_immutable.get('iterations')):   
            # Generate trials
            X, Y, trials_idx = tasks.create_trials(trial_params)
            #Set dataloaders
            training_set = tasks.Dataset(X[0], Y[0], trials_idx[0])
            training_generator = data.DataLoader(training_set, **params_generators)
            validate_set = tasks.Dataset(X[1], Y[1], trials_idx[1])
            validate_generator = data.DataLoader(validate_set, **params_generators)
             # Based on the nonlinearity, specify initialization of weights
            if init != 'default':
                if nonlinearity == 'relu': init = 'he'
                if nonlinearity == 'tanh': init = 'xavier'
            if model_params.get('random_w') is False:
                w = model_params.get('w')
                w = w.to(device)
            else:
                w = model_params.get('w')
                
                # Randomize the 2D tensors corresponding to the topology and
                # weights. Function rand_net freezes the diagonal elements
                # for symmetric tensors and, with the defaults, 
                # for non-symmetric tensors as well. Thus, randomization do
                # not involve the diagonal entries. See rand_net documentation
                # to unfreeze the diagonal if desired.
                w = networkmetrics.rand_net(w)
                w = w.to(device)
                                
            # Initiate model
            if not pretrained_folder:    
                model = rnnmodels.ModelBio(
                                            input_size = model_params.get('input_size'), 
                                            output_size = model_params.get('output_size'), 
                                            hidden_dim = model_params.get('hidden_dim'), 
                                            n_layers = 1,
                                            w = w,
                                            remap_w = remap_w,
                                            init = init,
                                            nonlinearity = nonlinearity,
                                            device = device
                                            )
                                
            # if pretrained is selected, then load the pretrained and modify it
            # to perform the task
            if pretrained_folder:
                # if the task is nback_mem, then we have to instantiate
                # a model for the seq_mem task
                if trial_params.get('task_name') == 'nback_mem':
                    output_size = 1
                    n_last_layers = -1#remove last layer
                    
                    # Plug in a linear and softmax layer
                    new_modules = [
                            nn.Linear(model_params.get('hidden_dim'),
                                      3),
                            nn.LogSoftmax(dim=1)
                            ]
                
                # if the task is seq_mem, then we have to instantiate
                # a model for the nback_mem task 
                if trial_params.get('task_name') == 'seq_mem':
                    output_size = 3 
                    n_last_layers = -2#remove 2 last layers
                    
                    # Plug in a linear layer
                    new_modules = [
                            nn.Linear(model_params.get('hidden_dim'),
                                      1),
                            ]
                    
                model = rnnmodels.ModelBio(
                                           input_size = model_params.get('input_size'), 
                                           output_size = output_size, 
                                           hidden_dim = model_params.get('hidden_dim'), 
                                           n_layers = 1,
                                           w = w,
                                           remap_w = remap_w,
                                           init = init,
                                           nonlinearity = nonlinearity,
                                           device = device
                                           )
                            
                # load pretrained model that has performed seq_mem
                print('\nLoading pretrained model...!\n') 
                model = auxfun.load_pretrained(
                                                model,
                                                pretrained_folder = pretrained_folder, 
                                                epoch = pretrained_epoch,
                                                it = pretrained_iteration,
                                                combo_nr = combo_index
                                                )
                
                # Modify the pretrained loaded model, so that it
                # can perform the nback_mem task  
                print('\nModifying model...\n') 
                model = rnnmodels.ModelBio_Modified(
                                                     model = model, 
                                                     n_last_layers = n_last_layers,
                                                     new_modules = new_modules
                                                     )
            
            # If asked freeze the hidden layer
            # Freeze the hidden layer
            if freeze_layer is True:  
                print('\nFreezing hidden layer of the model...!\n') 
                values, names = auxfun.get_model_params(model)
            
                # Hidden is the parameter with names[1]
                model = auxfun.freeze_params(
                                              model,
                                              params_to_freeze=names[1]
                                              )
            
            # Change all params of model to double 
            # Also send model to device - this has to be prior to the 
            # optimizer definition! 
            #model = model.double()             
            model = model.to(device)            
            
            # Intialize optimizer and cost based on task
            if trial_params['task_name'] == 'seq_mem':
                criterion = nn.MSELoss()
            
            if (trial_params['task_name'] == 'pic_mem' 
                or trial_params['task_name'] == 'nback_mem' 
                or trial_params['task_name'] == 'pic_latent_mem'):
                criterion = nn.NLLLoss()
            
            if combination[all_keys.index('optimizer')] == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
                
            if combination[all_keys.index('optimizer')] == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)  
                
            if combination[all_keys.index('optimizer')] == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr) 
            
            # Train and validate
            (loss, 
            loss_null,
            metrics,
            metrics_null) = modeltraineval.train_validate_epochs( 
                                                model,  
                                                epochs = train_params_immutable.get('epochs'),
                                                training_generator = training_generator,
                                                validate_generator = validate_generator, 
                                                optimizer = optimizer, 
                                                criterion = criterion, 
                                                batch_size = params_generators.get('batch_size'),
                                                trim = True,
                                                calc_null = False,
                                                w = w,
                                                task_name = trial_params['task_name'],
                                                device = device,
                                                metrics = metrics,
                                                store_every_epoch = 10,
                                                folder_save_model = results_folder / str(combo_index),
                                                iteration = it
                                                )
     
            # Keep the train and val losses for the model for this iteration 
            train_loss_all.append(loss[0])
            validate_loss_all.append(loss[1])  
            
            # the metrics list is not empty if no metrics were used - it 
            # contains two empty lists (no traina nd validation metrics).
            # Thus check if the first element is an empty list to decide
            # if it should be saved or not 
            if metrics[0]:
                train_metrics_all.append(metrics[0])
                validate_metrics_all.append(metrics[1])
                                                                
    # Store results for the current combo in the current folder  
    file_to_save = results_folder / str(combo_index) / 'train_loss_all'
    np.save(file_to_save, train_loss_all)
    
    file_to_save = results_folder / str(combo_index) / 'validate_loss_all'
    np.save(file_to_save, validate_loss_all)
    
    # the metrics list is not empty if no metrics were used - it 
    # contains two empty lists (no train and validation metrics).
    # Thus check if the first element is an empty list to decide
    # if it should be saved or not.  
    if metrics[0]:
        file_to_save = results_folder / str(combo_index) / 'train_metrics_all'
        np.save(file_to_save, train_metrics_all)
        
        file_to_save = results_folder / str(combo_index) / 'validate_metrics_all'
        np.save(file_to_save, validate_metrics_all)
        