#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random

from keras.datasets import mnist
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils import data

import auxfun

# Dataset class
class Dataset(data.Dataset):
  '''
  Map-style dataset object to be used from the DataLoader
  It implements the __get_item()__ and __len()__ methods
  See https://pytorch.org/docs/stable/data.html
  '''
  def __init__(self, trials_in, trials_out, trial_ids):
        # Initialize
        self.trials_in = trials_in
        self.trials_out = trials_out
        self.trial_ids = trial_ids
        unique_trial_ids = np.unique(trial_ids)# get the unique ids
        
        #permute to ensure mixed labels/trials in the batch and assign to object!!!
        self.unique_trial_ids = unique_trial_ids[np.random.permutation(len(unique_trial_ids))]

  def __len__(self):
        # Size of unique trials
        return len(self.unique_trial_ids)

  def __getitem__(self, index):
        # Generates one trial
        # Select sample
        current_id = self.unique_trial_ids[index]

        idx = np.where(current_id == self.trial_ids)[0]    
        
        # Load data and get label
        X = self.trials_in[idx, :]
        Y = self.trials_out[idx, :]

        return X, Y

# Task functions 
def generate_sequence_patterns(
                               pattern_length = 3, 
                               low = 0., 
                               high = 1., 
                               nr_of_trials = 100
                               ): 
    '''
    Generate pattern to memorize with length N and from a uniform distribution
    between low and high values. 
    
    Trials have a memorize period (the generated numbers=pattern_length) 
    and a recall period, that is, 0s=pattern_length. The trials are padded with
    zeros and ones with 1 denoting "recall cue". Thus, trials are 2D arrays. 
    
    Input
    -----
    pattern_length: int, default 3, indicating the lenght of the pattern to
        be memorized
        
    low: float, default 0., corresponding to the lowest value of the uniform
        distribution from which random numbers are drawn
        
    high: float, default 1., corresponding to the highest value of the uniform
        distribution from which random numbers are drawn    
        
    nr_of_trials: int, default 100, indicating the total number of trials
    
    Output
    ------
    all_input_trials: ndarray of shape (N,2) where:
        N = ((pattern_length*2) + 1) * nr_of_trials
        
    all_output_trials: ndarray of shape (N,1) where:
        N = ((pattern_length*2) + 1) * nr_of_trials
        
    all_trials_index: ndarray of shape (N,) where:
        N = ((pattern_length*2) + 1) * nr_of_trials 
    '''
    all_input_trials = None
    all_output_trials = None
    all_trials_index = None
        
    for tr in range(nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input 
        # (the patterns to be memorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns. 
        
        # 1 is presented only once, with zeros following it for the "null input"   
        null_input = np.zeros((2, pattern_length+1))
        
        # Assign the cue at the upper left corner so that the first column of the 
        # null input is actually the recall cue.
        null_input[0,0] = 1
        padding_for_trial = np.zeros((pattern_length,))
        
        #Generate one trial based on the specifications
        trial = np.random.uniform(low, high, pattern_length)
    
        # Add the padding that corresponds to a cue=0 (that means no replaying yet,
        # but leanrning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        input_trial = np.hstack((trial, null_input))
        
        # Now we can construct the desired ouput. This is basically a "mirrored"
        # version of the input, so construct accordingly: where null_input put
        # the current trial and vice versa. 
        
        # We need no padding for the output (no "cue needed"). Just require 0s
        # when the pattern is being learned.
        null_output = np.zeros((1, pattern_length+1))#Add 1 column to have the same length with input
        trial = trial[1:,:]   
        output_trial = np.hstack((null_output, trial))
        
        # Concatanate the generated input/output trials to the the overall 
        # trials array 
        if all_input_trials is None:            
            all_input_trials = input_trial
            all_output_trials = output_trial            
        else:            
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))
            
        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1],)
        current_index[:] = tr
               
        if all_trials_index is None:            
            all_trials_index = current_index            
        else:            
            all_trials_index = np.hstack((all_trials_index, current_index))
        
    
    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T
    
    all_trials_index = all_trials_index.T
        
    
    return all_input_trials, all_output_trials, all_trials_index   

def generate_pic_wm_trials(
                           images = None,
                           trial_length = 5, 
                           nr_of_trials = 100, 
                           n_back = None,
                           trial_matching = False,
                           rescale = True
                           ): 
    '''
    Generate trials of a N-Back working memory task based on 2D images
    
    The trials consists of consecutive images and the task is to decide
    if the image presented last matches or not (in terms of category e.g. 
    number) the image N timesteps ago
    
    The trials are padded with zeros and ones with 1 denoting "target image" 
    cue. Thus, trials are 2D arrays
    
    Input
    -----
    images: ndarray of shape (N,K,L) with N the number of images, and K and L
        the dimensions of the image
        
        images contain values in the [0 255] range
        
        As a default the MNIST dataset, asprepackaged in Keras, is used.
        Thus, N=10000 K=28 L=28 and datatype is unit8
        
        However, any image dataset can be passed as an argument given that it
        complies with the aformentioned format

    trial_length: int, default 5, indicating the lenght of the trial, 
        specifically the number of images that are used in the memorization 
        phase of the task. The image corresponding to the latest timestep is 
        always the target image. For instance if trial_length=5, them the 5th
        image is the target
     
    nr_of_trials: int, default 100, indicating the total number of trials    
    
    n_back: int specifying the match N timesteps before the last image. For
        instance, if n_back=2, then the image that should be compared with the 
        target image is the 3rd image in the trial
        
    trial_matching: bool, default False, specifying if the trials to be 
        generated should be matching trials(=True) or non-matching trials 
        (=False) 
        
    rescale: bool, default True, specifying if the images should be rescaled
        by dividing with 255
        
    Output
    ------
    all_input_trials: ndarray of shape (N,M) where:
        N = ((trial_length*2) + 1) * nr_of_trials
        M = (K*L)+1 (image size + 1)
        
    all_output_trials: ndarray of shape (N,1) where:
        N = ((trial_length*2) + 1) * nr_of_trials
        
    all_trials_index: ndarray of shape (N,) where:
        N = ((trial_length*2) + 1) * nr_of_trials        
    '''
    if ((trial_length-n_back) <=0) and (n_back is not None):
        raise ValueError('N-Back value must be less than the trial length')
           
    all_input_trials = None
    all_output_trials = None
    all_trials_index = None
        
    img_pixels = images.shape[1]*images.shape[2] 
    
    for tr in range(nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input 
        # (the patterns to be memomorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns. 
        
        # 1 is presented only once, with zeros following it for the "null input" 
        null_input = np.zeros((img_pixels+1, 2))
        
        # Assign the cue at the upper left corner so that the first column of the 
        # null input is actually the recall cue.
        null_input[0,0] = 1
        
        padding_for_trial = np.zeros((trial_length,))
        
        #Generate one trial based on the specifications
        # The last pic in the trial must be also the pic n_back steps
        target_pic_idx = random.randrange(images.shape[0])
        trial = np.zeros((img_pixels, trial_length))
        
        #Mark the positions of the target picture with 1s and assign the 
        #target pic in the correct indexes indicated by len(trial_idxs) and
        #n_back. Take into account if the trials are 
        #trial_matching = True or False
        trial_idxs = np.zeros((trial_length,))
        
        if trial_matching is True:
            trial_idxs[len(trial_idxs)-1] = 1
            trial_idxs[(len(trial_idxs)-1) - n_back] = 1
            trial[:, len(trial_idxs)-1] =  images[target_pic_idx, :, :].reshape(img_pixels,)
            trial[:, (len(trial_idxs)-1) - n_back] =  images[target_pic_idx, :, :].reshape(img_pixels,)
        else:
            trial_idxs[len(trial_idxs)-1] = 1
            trial[:, len(trial_idxs)-1] =  images[target_pic_idx, :, :].reshape(img_pixels,) 
        
        random_pic_idx = random.sample(range(0, images.shape[0]-1), 
                                       len(np.where(trial_idxs == 0)[0]))
    
        rand_pic_idxs = np.where(trial_idxs == 0)[0]
    
        for i in range (len(random_pic_idx)):
           trial[:,rand_pic_idxs[i]] = images[random_pic_idx[i], :, :].reshape(img_pixels,)
    
        #Rescale to 255 if rescale True
        if rescale is True:
            trial = trial/255
    
        # Add the padding that corresponds to a cue=0 
        #(that means no replaying yet, but learning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        
        input_trial = np.hstack((trial, null_input))
        
        # Now we can construct the desired ouput. 
        # What we need is an array with input_trial shape with 3 discrete
        # values: 
        # 0:fixation 
        # 1:n_back matches=False
        # 2:n_back matches=True
        # Hence, the output trials correspond to the correct labels of 
        # a 3-class classification problem
        output_trial = np.zeros((1, trial_length+2))#Add 1 column to have the same length with input
        
        #Assign the correct labeling
        
        if trial_matching is True:
            output_trial[0, (trial_length+1):] = 2
        else:
            output_trial[0, (trial_length+1):] = 1
        
        # Concatanate the generated input/output trials to the overall 
        # trials array 
        if all_input_trials is None:       
            all_input_trials = input_trial
            all_output_trials = output_trial       
        else:       
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))
            
        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1],)
        current_index[:] = tr
        
        
        if all_trials_index is None:          
            all_trials_index = current_index          
        else:          
            all_trials_index = np.hstack((all_trials_index, current_index))
            
    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T
    
    all_trials_index = all_trials_index.T
          
    return all_input_trials, all_output_trials, all_trials_index  

def generate_pic_latent_wm_trials(
                                  images = None,
                                  labels = None,
                                  trial_length = 5, 
                                  nr_of_trials = 100, 
                                  n_back = None,
                                  trial_matching = False,
                                  rescale = True
                                  ):   
    if ((trial_length-n_back) <=0) and (n_back is not None):
        raise ValueError('N-Back value must be less than the trial length')
           
    all_input_trials = None
    all_output_trials = None
    
    all_trials_index = None
        
    img_size = images.shape[1]
    
    #Rescale to [0 1] if rescale True
    if rescale is True:
        images = auxfun.scale_tensor(images, 
                                     global_scaling=False
                                     )# rescale each trial seperately (row-wise rescaling)
    
    # Convert to numpy
    images = images.clone().numpy()
    labels = labels.clone().numpy()

    # Keep unique labels
    unique_labels = np.unique(labels)    
    
    for tr in range(nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input 
        # (the patterns to be memomorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns. 
        
        # 1 is presented only once, with zeros following it for the "null input" 
        null_input = np.zeros((img_size+1, 2))
        
        # Assign the cue at the upper left corner so that the first column of the 
        # null input is actually the recall cue.
        null_input[0,0] = 1
        
        padding_for_trial = np.zeros((trial_length,))
        
        # Generate one trial based on the specifications
        # The last pic in the trial must be also the pic n_back steps
        # Select the targets and the non-targets based on the labels
        target_pic_label = random.randrange(unique_labels.shape[0])
        trial = np.zeros((img_size, trial_length))
        
        potential_target_pic_idxs = np.where(target_pic_label == labels)[0]
        potential_non_target_pic_idxs = np.where(target_pic_label != labels)[0]
        
        target_pic_idx = random.sample(list(potential_target_pic_idxs), 1)
        target_pic = images[target_pic_idx, :]
        
        # Remove the index of the used pic from the pool of potential targets
        potential_target_pic_idxs = potential_target_pic_idxs[np.where(
                                                              potential_target_pic_idxs != target_pic_idx
                                                              )[0]
                                                             ]
        
        #Mark the positions of the target picture with 1s and assign the 
        #target pic in the correct indexes indicated by len(trial_idxs) and
        #n_back. Take into account if the trials are 
        #trial_matching = True or False
        trial_idxs = np.zeros((trial_length,))
        
        if trial_matching is True:
            trial_idxs[len(trial_idxs)-1] = 1
            trial_idxs[(len(trial_idxs)-1) - n_back] = 1
            trial[:, len(trial_idxs)-1] = target_pic
            trial[:, (len(trial_idxs)-1) - n_back] = target_pic
        else:
            trial_idxs[len(trial_idxs)-1] = 1
            trial[:, len(trial_idxs)-1] = target_pic
        
        rand_pic_in_trial_idxs = np.where(trial_idxs == 0)[0]
        
        random_pic_idx = random.sample(list(potential_non_target_pic_idxs), 
                                       len(rand_pic_in_trial_idxs)
                                       )
  
        for i in range (len(random_pic_idx)):
           trial[:, rand_pic_in_trial_idxs[i]] = images[random_pic_idx[i], :]
     
        # Remove from the pool of potential random pic_idx the used pics   
        remove_pic_idxs = np.in1d(potential_non_target_pic_idxs, random_pic_idx).nonzero()[0]
        potential_non_target_pic_idxs = np.delete(potential_non_target_pic_idxs, 
                                                  remove_pic_idxs
                                                  )   
                     
        # Add the padding that corresponds to a cue=0 
        #(that means no replaying yet, but learning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        
        input_trial = np.hstack((trial, null_input))
        
        # Now we can construct the desired ouput. 
        # What we need is an array with input_trial shape with 3 discrete
        # values: 
        # 0:fixation 
        # 1:n_back matches=False
        # 2:n_back matches=True
        # Hence, the output trials correspond to the correct labels of 
        # a 3-class classification problem
        output_trial = np.zeros((1, trial_length+2))#Add 1 column to have the same length with input
        
        #Assign the correct labeling
        
        if trial_matching is True:
            output_trial[0, (trial_length+1):] = 2
        else:
            output_trial[0, (trial_length+1):] = 1
        
        # Concatanate the generated input/output trials to the overall 
        # trials array 
        if all_input_trials is None:          
            all_input_trials = input_trial
            all_output_trials = output_trial          
        else:          
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))
            
        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1],)
        current_index[:] = tr
              
        if all_trials_index is None:           
            all_trials_index = current_index           
        else:           
            all_trials_index = np.hstack((all_trials_index, current_index))
            
    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T
    
    all_trials_index = all_trials_index.T
          
    return all_input_trials, all_output_trials, all_trials_index  

def generate_nback_wm_trials(
                            trial_length = 5, 
                            nr_of_trials = 100, 
                            n_back = None,
                            trial_matching = False
                            ):
    '''
    Generate trials of a N-Back working memory task based on a sequence of
    numbers from a random uniform distribution [0 1]
    
    The trials consists of numbers and the task is to decide
    if the number presented last matches or not the number N timesteps ago
    
    The trials are padded with zeros and ones with 1 denoting "target number" 
    cue. Thus, trials are 2D arrays

    trial_length: int, default 5, indicating the lenght of the trial, 
        specifically the number of number that are used in the memorization 
        phase of the task. The number corresponding to the latest timestep is 
        always the target number. For instance if trial_length=5, them the 5th
        number is the target
    
    nr_of_trials: int, default 100, indicating the total number of trials  
    
    n_back: int specifying the match N timesteps before the last number. For
        instance, if n_back=2, then the number that should be compared with the 
        target number is the 3rd number in the trial
        
    trial_matching: bool, default False, specifying if the trials to be 
        generated should be matching trials(=True) or non-matching trials 
        (=False) 
    
    Output
    ------
    all_input_trials: ndarray of shape (N,2) where:
        N = ((trial_length*2) + 1) * nr_of_trials
        
    all_output_trials: ndarray of shape (N,1) where:
        N = ((trial_length*2) + 1) * nr_of_trials
        
    all_trials_index: ndarray of shape (N,) where:
        N = ((trial_length*2) + 1) * nr_of_trials
    '''
    if ((trial_length-n_back) <=0) and (n_back is not None):
        raise ValueError('N-Back value must be less than the trial length')
           
    all_input_trials = None
    all_output_trials = None
    
    all_trials_index = None
    
    for tr in range(nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input 
        # (the patterns to be memomorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns. 
        
        # 1 is presented only once, with zeros following it for the "null input" 
        null_input = np.zeros((2, 1))
        
        # Assign the cue at the upper left corner so that the first column of the 
        # null input is actually the recall cue.
        #null_input[0,0] = 1
        
        padding_for_trial = np.zeros((trial_length,))
        padding_for_trial[-1] = 1.
        
        #Generate one trial based on the specifications
        # The last pic in the trial must be also the pic n_back steps
        #target_pic_idx = random.randrange(x_train.shape[0])
        
        #trial = np.zeros((1, trial_length))
        trial = np.random.uniform(0., 1., trial_length)
        trial = np.reshape(trial, (1, trial_length))
        
        #Mark the positions of the target picture with 1s and assign the 
        #target pic in the correct indexes indicated by len(trial_idxs) and
        #n_back. Take into account if the trials are 
        #trial_matching = True or False
        #trial_idxs = np.zeros((trial_length,))
        
        #target_value = random.sample(range(0, 2), 1)
        target_value = np.random.uniform(0., 1., 1)
        
        if trial_matching is True:
            #trial_idxs[len(trial_idxs)-1] = 1
            #trial_idxs[(len(trial_idxs)-1) - n_back] = 1
            trial[0, trial_length-1] = target_value[0]
            trial[0, (trial_length-n_back-1)] = target_value[0]
        else:
            #trial_idxs[len(trial_idxs)-1] = 1
            #trial_idxs[(len(trial_idxs)-1) - n_back] = 1
            trial[0, trial_length-1] =  target_value[0]
            
            # Put the wrong value in n_back position since the trial is False
            # if target_value[0] == 0:
            #     trial[0, len(trial_idxs) - n_back] = 1
            # else:
            #     trial[0, len(trial_idxs) - n_back] = 0
    
        #rand_trials_idxs = np.where(trial_idxs == 0)[0]
        #random_values = [random.randint(0, 1) for x in range(0, len(rand_trials_idxs))] 
        #random_values = np.random.uniform(0., 1., len(rand_trials_idxs))
    
        # for i in range (0, len(rand_trials_idxs)):
        #    trial[:,rand_trials_idxs[i]] = random_values[i]
    
        # Add the padding that corresponds to a cue=0 
        #(that means no replaying yet, but learning the input patterns)
        trial = np.vstack((padding_for_trial, trial))
        
        input_trial = np.hstack((trial, null_input))
        
        # Now we can construct the desired ouput. 
        # What we need is an array with input_trial shape with 3 discrete
        # values: 
        # 0:fixation 
        # 1:n_back matches=False
        # 2:n_back matches=True
        # Hence, the output trials correspond to the correct labels of 
        # a 3-class classification problem
        output_trial = np.zeros((1, trial_length+1))#Add 1 column to have the same length with input
        
        #Assign the correct labeling
        
        if trial_matching is True:
            output_trial[0, trial_length] = 2
        else:
            output_trial[0, trial_length] = 1
        
        # Concatanate the generated input/output trials to the the overall 
        # trials array 
        if all_input_trials is None:     
            all_input_trials = input_trial
            all_output_trials = output_trial 
        else:
            
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))
            
        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1],)
        current_index[:] = tr
               
        if all_trials_index is None:           
            all_trials_index = current_index           
        else:           
            all_trials_index = np.hstack((all_trials_index, current_index))
            
    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T
    
    all_trials_index = all_trials_index.T
           
    return all_input_trials, all_output_trials, all_trials_index 

def wrapper_trials(func):
    '''
    Decorator function used to generate trials by setting the 
    trial_matching parameter to False and subsequently True
    (Used from create_trials()) 
    '''
    def internal(**kwargs):
        print('Generating trials with params:')
        for key in kwargs:
            if key not in ['images', 'labels']:
                print('%s: %s' % (key, kwargs[key]))
        
        # Call the function with 'trial_matching':True
        kwargs.update({'trial_matching':True})    
        X_match, Y_match, indexes_match=func(**kwargs)  
            
        # Call the function with 'trial_matching':False
        kwargs.update({'trial_matching':False})  
        X_non_match, Y_non_match, indexes_non_match=func(**kwargs)
        
        print('Generating trials with params:')
        for key in kwargs:
             if key not in ['images', 'labels']:
                print('%s: %s' % (key, kwargs[key]))
        
        # Increase the indexes_non_match in such a way so that they are 
        # a continuation of the indexes_match 
        indexes_non_match  +=  np.max(indexes_match)+1
        
        # Concatanate all true and false trials
        X = np.vstack((X_match, X_non_match))
        Y = np.vstack((Y_match, Y_non_match))
        indexes = np.hstack((indexes_match, indexes_non_match))
        
        return  X,Y,indexes
        
    return internal

def create_train_test_trials(
                             X = None,
                             Y = None,
                             indexes = None,
                             train_size = 0.8
                             ):
    '''
    Create train and test sets
    (this is just a wrapper around scikit-learn's GroupShuffleSplit)
    
    Input
    -----
    X: ndarray of shape (N,M) representing a feature matrix
        with N observations and M features (representing trials)
    
    Y: ndarray of shape (N,) of labels (representing task output)

    indexes: ndarray of shape (N,) with unique numbers (float or int) grouping
        together the observations constituting one trial
        These indexes will be used by  thr group shuffle split
     
    train_size: float (0 1), default 0.8, denoting the percentage of X to be 
        used for the train set 
        
    Output
    ------
    X: list of len 2 containing the train X[0] and test set X[1]
        
    Y: list of len 2 containing the train Y[0] and test set Y[1] labels 
        
    indexes: list of len 2 containing the train indexes[0] and test set 
    indexes[1] indexes. These indexes correspond to the indexes parameter  
    '''
    # Create train and validate tests by ensuring that only complete 
    # trials are shuffled!
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size)
    
    for train_idx, validate_idx in gss.split(X, Y, indexes):
        X_train = X[train_idx]   
        Y_train = Y[train_idx]  
        
        X_validate = X[validate_idx]   
        Y_validate = Y[validate_idx]
    
    # Seperate idx for train and validate sets 
    train_trial_idxs = indexes[train_idx]
    validate_trial_idxs = indexes[validate_idx] 

    # Wrap up results in lists. Each position corresponds to train
    # validate sets   
    X = [X_train, X_validate]
    Y = [Y_train, Y_validate]
    indexes = [train_trial_idxs, validate_trial_idxs]
    
    return X, Y, indexes
    
def trim_zeros_from_trials(actual=None, predicted=None):
    '''
    Remove zeros in-between experimental trials. This is useful to remove 
    "fixation" periods from trials and thus do not take into account 
    such perios in the calculation of the error (and thus the training of the 
    network)
    
    Input
    -----
    actual: torch.Tensor of shape (N,M) where N is the length of the experiement
        and M the dimensions involved with a specific experiment. 
        NOTE: the function will remove the observations from N that are 
        equal to 0
        
    predicted: torch.Tensor of shape (N,K) here N is the length of the experiement
        and K the dimensions involved with a specific experimental 
        output/prediction 
        
    Output
    ------
    actual_trimmed:  torch.Tensor of shape (N-P,M) where N-P is the length of 
        the experiement after the removal of 0s from actual[:,0],
        and M the dimensions involved with a specific experiment.
        
    predicted_trimmed: torch.Tensor of shape (N-P,K) where N-P is the length of 
        the experiement after the removal of 0s from actual[:,0],
        and K the dimensions involved with a specific experimental 
        output/prediction        
    '''
    ind = torch.where(actual != 0)[0]#use the actual trials for tracking non 0s 
    actual_trimmed = actual[ind]
    predicted_trimmed = predicted[ind]    
    
    return actual_trimmed, predicted_trimmed  

def create_trials(trial_params):
    '''
    Create trials that correspond to the experimental design specified in the 
    dictionary trial_params
    
    Input
    -----
    trial_params: dict with trial parameters for a task
        key: 'task_name'    value: {'nback_mem','seq_mem', 'pic_mem', 'pic_latent_mem'}
             'nr_of_trials'        int
             'trial_length'        int
             'trial_matching'      bool
             'train_size'          float (0 1)
             'n_back'              int (if 'task_name'={'nback_mem', 'pic_mem', 'pic_latent_mem'})
    
    '''
    if trial_params['task_name'] == 'seq_mem':
        pattern_length = trial_params['pattern_length']
        low = trial_params['low']
        high = trial_params['high']
        nr_of_trials = trial_params['nr_of_trials']
        
        train_size = trial_params['train_size']
        
        # Train and validation test
        (X, 
         Y,
         indexes) = generate_sequence_patterns(pattern_length = pattern_length, 
                                               low = low, 
                                               high = high, 
                                               nr_of_trials = nr_of_trials
                                               ) 
                                               
    if trial_params['task_name'] == 'pic_mem':  
        # Load the data
        # Use only the test set - it has 
        # less but sufficient samples than the train
        _ , (x_test, y_test) = mnist.load_data() 
        
        nr_of_trials = trial_params['nr_of_trials']
        trial_length = trial_params['trial_length']
        n_back = trial_params['n_back']
        trial_matching = trial_params['trial_matching']
        rescale = trial_params['rescale']
        
        train_size = trial_params['train_size'] 
        
        generate_pic_wm_trials_boosted = wrapper_trials(generate_pic_wm_trials)
        
        X, Y, indexes = generate_pic_wm_trials_boosted(images = x_test,
                                                       trial_length = trial_length, 
                                                       nr_of_trials = nr_of_trials, 
                                                       n_back = n_back,
                                                       trial_matching = trial_matching,
                                                       rescale = rescale
                                                       )
        
    if trial_params['task_name'] == 'pic_latent_mem':  
        # Load the data
        # Use only the test set - it has 
        # less but sufficient samples than the train
        output, labels = torch.load('latent_mnist/mnist_latent_output_labels.pt') 
        
        nr_of_trials = trial_params['nr_of_trials']
        trial_length = trial_params['trial_length']
        n_back = trial_params['n_back']
        trial_matching = trial_params['trial_matching']
        rescale = trial_params['rescale']
        
        train_size = trial_params['train_size'] 
        
        generate_pic_wm_trials_boosted = wrapper_trials(generate_pic_latent_wm_trials)
        
        X, Y, indexes = generate_pic_wm_trials_boosted(images = output,
                                                       labels = labels,
                                                       trial_length = trial_length, 
                                                       nr_of_trials = nr_of_trials, 
                                                       n_back = n_back,
                                                       trial_matching = trial_matching,
                                                       rescale = True
                                                       )    
        
    if trial_params['task_name'] == 'nback_mem': 
        nr_of_trials = trial_params['nr_of_trials']
        trial_length = trial_params['trial_length']
        n_back = trial_params['n_back']
        trial_matching = trial_params['trial_matching']
        
        train_size = trial_params['train_size'] 
        
        generate_bin_wm_trials_boosted = wrapper_trials(generate_nback_wm_trials)
        
        X, Y, indexes = generate_bin_wm_trials_boosted(trial_length = trial_length, 
                                                       nr_of_trials = nr_of_trials, 
                                                       n_back = n_back,
                                                       trial_matching = trial_matching,
                                                       )
                                                                                                                                                                                                                
    # Create train and validate tests by ensuring that only complete trials
    # are shuffled!                                        
    X, Y, indexes = create_train_test_trials(X = X,
                                             Y = Y,
                                             indexes = indexes,
                                             train_size = train_size
                                             )
    
    X[0], Y[0], indexes[0] = auxfun.group_shuffle(X[0], Y[0], indexes[0])
    X[1], Y[1], indexes[1] = auxfun.group_shuffle(X[1], Y[1], indexes[1])
    
    return X, Y, indexes

