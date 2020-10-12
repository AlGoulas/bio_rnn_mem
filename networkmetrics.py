#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

def rand_net(A, freeze_diag=True):
    ''''
    Randomize the elements of a 2D tensor.
    
    Input:
        
    A: tensor
        a 2D tensor (e.g., from PyTorch)
    
    freeze_diag: bool, default=True
        indicating if diagonal values should be randomized (=True) 
        or not (=False).
    
    Output:
        
    X: tensor
        a 2D tensor with randomized values from tensor A
    
    NOTE:
        
    this option is valid only for non-symmetric tensors, that is,
    not every non-zero entry Ai,j corresponds to a non-zero entry Aj,i.
    For symmetric tensors, always freeze_diag=True.
       
    
    Examples:
        
    # if R is symmetric 
    R_rand = rand_net(R)   
        
    # if R is non-symmetric    
    R_rand = rand_net(R, freeze_diag=False)#Randomize also diag
    R_rand = rand_net(R)   
    '''  
    if check_symmetric(A.data.numpy()):
        nodes = A.shape[0]

        # Get the idx and values of the lower diagonal - since A is symmetric,
        # that is all we need.
        X = torch.ones((nodes, nodes)).double() 
        X = torch.triu(X, diagonal=0)
        idx = torch.where(X==0.)
        values = A[idx]

        # Permute values and assign them to lower triangle + diag positions
        values = values[np.random.permutation(len(values))]
        X = torch.zeros((nodes, nodes)).double()  
        X[idx] = values 
        X = X + X.transpose(0,1)# transpose to symmetrize  
        X = X + torch.diagflat(A.diag())
    else: 
        
        nodes = A.shape[0]
        
        if freeze_diag:
            # Make mask marking the diagonal with 1s - so we work with the 
            # 0 entries
            M = torch.diagflat(torch.ones(nodes,))
            idx = torch.where(M==0.)
            values = A[idx]
            
            # Permute the values and assign them to a 2D tensor
            values = values[np.random.permutation(len(values))]
            X = torch.zeros((nodes, nodes)).double()  
            X[idx] = values 
            # Put the diagonal valeus to the permuted tensor
            X = X + torch.diagflat(A.diag())
        else:
            # Get the idx of each position in the tensor
            idx = torch.where(A)
            values = A[idx]
            
            # Permute the values and assign them to a 2D tensor
            values = values[np.random.permutation(len(values))]
            X = torch.zeros((nodes, nodes)).double()  
            X[idx] = values
    
    return X

#Check is a 2D numpy array is symmetric
def check_symmetric(X):
    ''' 
    Check is a 2D numpy array is symmetric: if A[i,j] !=0 then A[j,i] !=0   
    for every i,j.
    Thus, no weights are taken into account to decide if symmetry exists.
    
    Input:
    X: numpy array, 2D
    
    Output:
    Boolean denoting if the numpy array is symmetric    
    '''
    return np.all(X == X.T)


