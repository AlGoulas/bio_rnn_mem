#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import auxfun

# Visualize boxplot with seaborn
def vis_boxplot(values = None, 
                grouping = None,
                xlabel = None, ylabel = None,
                file_name = None, path_save = None):
    '''Visualize values as boxplots based on the categorical variable
    
    Input
    -----
    
    values: ndarray of shape (N,), int or float, with the values to be plotted
    
    grouping: list or ndarray of shape (N,) (or length N) of unique categorical
        variables (int or str)
        
    xlabel: str, denoting the label of the x-axis
    
    ylabel: str, denoting the label of the y-axis  
    
    file_name: str, denoting the name of the figure file that will be stored
        (if path_save is not None).
        NOTE: the figure will be saved as .svg
       
    path_save: object of class pathlib.PosixPath, default None 
        The path where the figure will be stored (if not None) 
        path_save = Path('path_to_folder') 
        
    '''
    data_frame = {
                  'values': values, 
                  'grouping': grouping
                 }
    
    # Aseemble the values and grouping variables in a dataframe to be used 
    #from seaborn
    df = pd.DataFrame(data_frame)
    fig = plt.figure()
    fig.set_size_inches(10, 10)  
    ax = sns.boxplot(x='grouping', 
                     y='values', 
                     data=df, 
                     palette='ch:2.5,.25')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    # If a path is specififed, then save the figure as .svg
    if path_save is not None:
        file_name = file_name + '.svg'
        file_to_save= path_save / file_name
        plt.savefig(file_to_save, format='svg')

def visualize_data_frame(df = None, 
                         filters = None,
                         xlabel = None, 
                         ylabel = None,
                         file_name = None, 
                         path_save = None,
                         palette = None):
    '''Visualize dataframe df with seaborn as a boxplot. 
    
    If a filter dictionary is specified, then the dataframe df will be 
    filtered to only visualize data corresponding to the dataframe keys 
    (columns) that match the values of the keys in the dictionary filters.
    
    Input
    -----
    
    df: dataframe of pandas.core.frame.DataFrame
        It must contain at least two columns with keys: 'values' and 
        'grouping', with values denoting the values that will be plotted as a 
        boxplot and grouping a categorical variable (str or int) that 
        identifies the groups to be used for the grouping the values.
     
    filters: dict, with keys and values such that the dict keys match at least
        one dataframe key and values are contained in the dataframe in the 
        column denoted by the key (see example).
        
    xlabel: str, denoting the label of the x-axis
    
    ylabel: str, denoting the label of the y-axis  
    
    file_name: str, default None, denoting the name of the figure file 
        that will be stored (if path_save is not None).
        NOTE: the figure will be saved as .svg
       
    path_save: object of class pathlib.PosixPath, default None
        If not None, the path where the figure will be stored 
        path_save = Path('path_to_folder') 
        
    palette: seaborn seaborn.palettes._ColorPalette object 
        See sns.color_palette
    '''    
    # Reduce the dataframe by keeping only the rows with the column
    # values specified in filters
    if filters is not None:
        for key, value in filters.items():
            df = df[(df[key] == value)]
    
    fig = plt.figure()
    fig.set_size_inches(10, 10)  
    
    sns.set(font_scale=2)
    sns.set_style('white') 
    
    ax = sns.boxplot(x = 'grouping', 
                     y = 'values', 
                     data = df, 
                     palette = palette)
    if xlabel is not None and ylabel is not None:
        ax.set(xlabel = xlabel, ylabel = ylabel)
    
    # If a path is specififed, then save the figure as .svg
    if path_save is not None:
        file_name = file_name + '.svg'
        file_to_save= path_save / file_name
        plt.savefig(file_to_save, format='svg')

# Visualize the mean and std of multiple quantities 
def visualize_mean_std_mult(values_to_plot = None, 
                            cat_labels = None, 
                            bin_labels = None,
                            ep = None, 
                            title = None, 
                            xlabel = None, 
                            ylabel = None,
                            path_save = None, 
                            file_prefix = None,
                            palette = None):
    '''Plot with seaborn the values_to_plot as a lineplot (mean±std).
    
    Input
    -----
    values_to_plot: list of len N of ndarrays of shape (K,L)
        The function will create lineplots based on the values in the ndarrays,
        in such a way that the mean and std for each ndarray a in the list
        is calculated as np.mean(a, axis=0), np.std(a, axis=0), that is,
        column-wise.
        
        N such mean±std lineplots will be plotted corresponding to N ndarray
        in the list.
        
    cat_labels: list of str of len N with a string labeling each ndarray in 
        values_to_plot.
        
    bin_labels: list of str of len N with a string labeling each ndarray in 
        values_to_plot. The list is "binary": it must contain two unique 
        str/labels.
        
    ep: range function with start=0 and stop=L, where L is part of the shape of 
        the ndarrays (K,L) in values_to_plot.     
    
    title: str, default None, specifying the title of the lineplot
    
    xlabel: str, denoting the label of the x-axis
    
    ylabel: str, denoting the label of the y-axis
    
    path_save: object of class pathlib.PosixPath, default None
        If not None, the path where the figure will be stored 
        path_save = Path('path_to_folder') 
     
   file_prefix: str, prefix for the figure to be saved (if path_save is not None)     
        
    palette: seaborn seaborn.palettes._ColorPalette object 
        See sns.color_palette
    '''
    #Unpack epochs so that each iteration has an epoch for visualization
    iterations=values_to_plot[0].shape[0]
        
    epochs = np.kron(np.ones((iterations, 1)), ep)
    epochs = np.reshape(epochs, 
                       (epochs.size), 
                       'C')
    
    all_values = None
    all_epochs = None
    all_bin_labels = []
    all_cat_labels = []
    
    for v, values in enumerate(values_to_plot):
          current_values = np.reshape(values, 
                                     (values.size), 
                                     'C')
          all_values = auxfun.concatenate_arrays(master_container = all_values,
                                                 leech = current_values,
                                                 mode='h')
          
          all_epochs = auxfun.concatenate_arrays(master_container = all_epochs,
                                                 leech = epochs,
                                                 mode = 'h')
          
          b = [bin_labels[v]] * len(current_values) 
          c = [cat_labels[v]] * len(current_values)
          
          all_bin_labels.extend(b)
          all_cat_labels.extend(c)  
     
    # Make data frame and visualize data             
    data = {
             'loss': all_values, 
             'epochs': all_epochs,
             'topology': all_cat_labels,
             'train/test': all_bin_labels,
            }
    
    df = pd.DataFrame(data)
    
    fig = plt.figure()  
    fig.set_size_inches(10, 10)

    sns.set(font_scale=2)
    sns.set_style('white') 
    
    ax = sns.lineplot(x='epochs', 
                      y='loss', 
                      hue='topology',
                      style='train/test',
                      data=df,
                      palette=palette,
                      ci='sd')
    
    ax.set(xlabel = xlabel, ylabel = ylabel)
    if title is not None:
        ax.set_title(title)

    # If a path is specififed, the save the figure as .svg
    if path_save is not None:
        file_name = title.rstrip() + file_prefix + '.svg'
        file_to_save= path_save / file_name
        plt.savefig(file_to_save, format='svg')