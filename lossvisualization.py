#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

import auxfun
import visfun

# Make labels in svg editable
new_plt_params = {
                  'text.usetex': False,
                  'svg.fonttype': 'none'
                 }

plt.rcParams.update(new_plt_params)      
# Load, process and visualize results 
    
# Folders with the results
#results_folder_bio_w_p = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/development/memory-bio-art/results/bin_mem/topology_and_weight_map_partitioned/marmoset_4x/bio')  
#results_folder_rand_w_p = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/development/memory-bio-art/results/bin_mem/topology_and_weight_map_partitioned/marmoset_4x/rand')        

results_folder_bio_no = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/nback_mem/marmoset_4x/no_w_map/bio') 
results_folder_rand_no = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/nback_mem/marmoset_4x/no_w_map/rand')        
results_folder_bio_eq = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/nback_mem/marmoset_4x/w_map_eq/bio') 
results_folder_rand_eq = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/nback_mem/marmoset_4x/w_map_eq/rand')        
results_folder_bio_div = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/nback_mem/marmoset_4x/w_map_div/bio') 
results_folder_rand_div = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/nback_mem/marmoset_4x/w_map_div/rand')  

all_results_folders = [
                       results_folder_bio_no,
                       results_folder_rand_no,
                       results_folder_bio_eq,
                       results_folder_rand_eq,
                       results_folder_bio_div,
                       results_folder_rand_div
                      ]

# Give labels to the results folders to keep track of the results origin
results_labels = [
                  'b_w_no', 
                  'r_w_no',
                  'b_w_eq', 
                  'r_w_eq',
                  'b_w_div', 
                  'r_w_div'
                 ]

# Specify folder to store figs

path_save = Path('/Users/alexandrosgoulas/Data/work-stuff/python-code/projects/rnn-bio2art-sequence-mem/figs/nback_mem/marmoset_4x/all')

# List to keep all the results for all_results_folders
all_current_results_quantities = []
all_current_results_raw = []

for current_folder in all_results_folders:  
    file_name = 'folder_index.txt'
    file_to_open = current_folder / file_name
    
    file_folder_index = open(file_to_open, 'r')
    folder_indexes = file_folder_index.readlines()
    file_folder_index.close()
    
    #List keeping all the strings denoting the combination of params
    all_combinations_str = [] 
    all_combinations_str_cleaned = []
    
    # Lists to keep all computed quantities on results for each results folder
    # specified in all_results_folders 
    current_results_quantities = []
    
    # This is initialized as a four-lists list: 
    # 1st list numpy arrays with validate loss
    # 2nd list numpy arrays with train loss
    # 3rd list numpy arrays with validate metrics
    # 4th list numpy arrays with train metrics
    current_results_raw = [[],[],[],[]]
    
    # Loop through the folder with the results for each combo of params
    for folder_count in range(len(folder_indexes)):
        # Get params to be used for title and in the data frame
        current_str = folder_indexes[folder_count]
        folder_id = current_str.split(' ', 1)[0]
        combinations_str = current_str.split(' ', 1)[1]
        combinations_str.rstrip()# This creates a string with all the params in parentheses
            
        # Keep the params combinations for all the results
        combinations_str_cleaned = auxfun.clean_str(combinations_str)#make list of strings
        all_combinations_str.append(combinations_str)
        all_combinations_str_cleaned.append(combinations_str_cleaned)
        
        # Get metrics/loss
        (raw_results,  
        quantities_on_results,
        ep,
        iterations) = auxfun.read_results(
                                           results_folder = current_folder,  
                                           results_id = folder_count,
                                           start = 1,
                                           stop = None
                                           )
    
        # Store in the list the quantities_on_results 
        if current_results_quantities:                               
            current_results_quantities[0].extend(quantities_on_results['min_loss'])
            current_results_quantities[1].extend(quantities_on_results['min_loss_ep'])
            current_results_quantities[2].extend(quantities_on_results['min_loss_ep_perc'])
        else:
            current_results_quantities.append(quantities_on_results['min_loss'])
            current_results_quantities.append(quantities_on_results['min_loss_ep'])
            current_results_quantities.append(quantities_on_results['min_loss_ep_perc']) 

        # 1st list numpy arrays with validate loss
        # 2nd list numpy arrays with train loss
        # 3rd list numpy arrays with validate metrics
        # 4th list numpy arrays with train metrics 
        current_results_raw[0].append(raw_results['validate_loss'])
        current_results_raw[1].append(raw_results['train_loss'])   
        
        if 'validate_metrics' in raw_results.keys():
            current_results_raw[2].append(raw_results['validate_metrics'])
            current_results_raw[3].append(raw_results['train_metrics'])
                                         
    all_current_results_quantities.append(current_results_quantities)
    all_current_results_raw.append(current_results_raw)

# Visualize loss for each results per combination
for folder_count in range(len(folder_indexes)):       
    # Visualize loss with seaborn
    values_to_plot = []
    eval_labels = []#labels for train/test  
    cat_labels = []#labels for results folders  
    for results_folder in range(len(all_results_folders)):
        values_to_plot.append(all_current_results_raw[results_folder][0][folder_count])
        eval_labels.append('test')
        values_to_plot.append(all_current_results_raw[results_folder][1][folder_count])
        eval_labels.append('train')
        cat_labels.extend([results_labels[results_folder]] * 2)#labels for results folders  
    
    visfun.visualize_mean_std_mult(
                                    values_to_plot = values_to_plot, 
                                    cat_labels = cat_labels, 
                                    bin_labels = eval_labels,
                                    ep = ep, 
                                    title = all_combinations_str[folder_count],
                                    xlabel = 'epochs',
                                    ylabel = 'loss (NLL)',
                                    path_save = path_save,
                                    file_prefix = '_loss',
                                    palette = sns.color_palette('mako_r', 
                                                                len(all_results_folders))
                                   )        
   
# Visualize metrics (e.g., accuracy) for each results per combination
# If len > 2, it means we have to visualize metrics as well    
#if len(all_current_results_raw) > 2:
if 'validate_metrics' in raw_results.keys():    
    for folder_count in range(len(folder_indexes)):       
        # Visualize metrics with seaborn
        values_to_plot = []
        eval_labels = []#labels for train/test  
        cat_labels = []#labels for results folders  
        for results_folder in range(len(all_results_folders)):
            values_to_plot.append(all_current_results_raw[results_folder][2][folder_count])
            eval_labels.append('test')
            values_to_plot.append(all_current_results_raw[results_folder][3][folder_count])
            eval_labels.append('train')
            cat_labels.extend([results_labels[results_folder]] * 2)#labels for results folders  
        
        visfun.visualize_mean_std_mult(
                                        values_to_plot = values_to_plot, 
                                        cat_labels = cat_labels, 
                                        bin_labels = eval_labels,
                                        ep = ep, 
                                        title = all_combinations_str[folder_count],
                                        xlabel = 'epochs',
                                        ylabel = 'acc',
                                        path_save = path_save,
                                        file_prefix = '_metrics',
                                        palette = sns.color_palette('mako_r', 
                                                                    len(all_results_folders))
                                       )    
    
# Visualize by filtering params
new_labels = auxfun.extend_list(list_to_ext = all_combinations_str_cleaned, 
                                 ext = iterations
                                 )

extend_by = len(all_results_folders)

data_frame = {
             'lr': new_labels[0] * extend_by,
             'activation': new_labels[1] * extend_by,
             'optimizer': new_labels[2] * extend_by,
             'task_param': new_labels[3] * extend_by,
             }

# filters = {
#           'activation': 'relu'
#           }

filters = None

# Visualize loss boxplot
values = None
category =[]
for results_folder in range(len(all_results_folders)):
     values = auxfun.concatenate_arrays(
                                master_container = values,
                                leech = all_current_results_quantities[results_folder][0],
                                mode = 'h'
                                )
     
     category.extend([results_labels[results_folder]] * len(all_current_results_quantities[results_folder][0]))
     
data_frame['values'] = values 
data_frame['grouping'] = category

file_name = 'loss'
# Add extensions to the name of the file so we keep track the filters used
if filters is not None:
    for key, value in filters.items():
        file_name = file_name + '_' + value

df = pd.DataFrame(data_frame)
visfun.visualize_data_frame(df = df, 
                             filters = filters,
                             xlabel = 'topology', 
                             ylabel = 'loss (NLL)',
                             file_name = file_name, 
                             path_save = path_save,
                             palette = sns.color_palette('mako_r', 
                                                         len(all_results_folders))
                             )

# Visualize min epoch where min loss was achieved
values = None
category =[]
for results_folder in range(len(all_results_folders)):
     values = auxfun.concatenate_arrays(
                                master_container = values,
                                leech = all_current_results_quantities[results_folder][1],
                                mode = 'h'
                                )
     
     category.extend([results_labels[results_folder]] * len(all_current_results_quantities[results_folder][1]))
     
data_frame['values'] = values 
data_frame['grouping'] = category

file_name = 'min_epoch_loss'
# Add extensions to the name of the file so we keep track the filters used
if filters is not None:
    for key, value in filters.items():
        file_name = file_name + '_' + value

df = pd.DataFrame(data_frame)
visfun.visualize_data_frame(df = df, 
                             filters = filters,
                             xlabel = 'topology', 
                             ylabel = 'min epoch loss',
                             file_name = file_name, 
                             path_save = path_save,
                             palette = sns.color_palette('mako_r', 
                                                         len(all_results_folders))
                             )

# Visualize min perc epoch where min loss was achieved
values = None
category =[]
for results_folder in range(len(all_results_folders)):
     values = auxfun.concatenate_arrays(
                                master_container = values,
                                leech = all_current_results_quantities[results_folder][2],
                                mode = 'h'
                                )
     
     category.extend([results_labels[results_folder]] * len(all_current_results_quantities[results_folder][2]))
     
data_frame['values'] = values 
data_frame['grouping'] = category

file_name = '99% min_epoch_loss'
# Add extensions to the name of the file so we keep track the filters used
if filters is not None:
    for key, value in filters.items():
        file_name = file_name + '_' + value

df = pd.DataFrame(data_frame)
visfun.visualize_data_frame(df = df, 
                             filters = filters,
                             xlabel = 'topology', 
                             ylabel = '99% min epoch loss',
                             file_name = file_name, 
                             path_save = path_save,
                             palette = sns.color_palette('mako_r', 
                                                         len(all_results_folders))
                             )
