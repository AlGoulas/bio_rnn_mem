#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import statistics
import torch

import auxfun
import tasks

# Train/validate model in one epoch
def train_validate_model(
                         model, 
                         data_generator, 
                         optimizer = None, 
                         criterion = None, 
                         train = True,
                         calc_null = True,
                         batch_size = 32,
                         trim = True,
                         w = None,
                         metrics = [],
                         task_name = None,
                         device = 'cpu'
                         ):
       
    batch_loss = []
    batch_loss_null=[]
    
    batch_metrics = []
    batch_metrics_null = []
        
    model.train(train)#train(=True) or validate(=False) mode
    
    for X_batch, Y_batch in data_generator:
        # Send tensors to the device used
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()#Clear existing gradients from previous batch
    
        #model.train(train)#train(=True) or validate(=False) mode
                       
        output, hidden = model(X_batch)
        
        Y_batch = Y_batch.contiguous().view(
            int(batch_size * (len(output)/batch_size)), 
            -1)
        
        if trim is True: 
            Y_batch, output = tasks.trim_zeros_from_trials(actual=Y_batch, 
                                                             predicted=output)
        
        if task_name == 'pic_mem' or task_name == 'bin_mem': 
            Y_batch = Y_batch.view(Y_batch.shape[0],)
            Y_batch = Y_batch.long()
        
        loss = criterion(output, Y_batch)
        batch_loss.append(loss.data.item())
            
        # Calculate also the metrics
        # Accuracy
        if 'acc' in metrics:
            acc = auxfun.calc_accuracy(output=output, labels=Y_batch)
            batch_metrics.append(acc)
             
        if train is True:      
            loss.backward()#backpropagation and calculation of gradients
            
            # Estimate the mask dictated by the topology 
            # and apply it to the gradients 
            if w is not None:
                mask_idx = torch.where(w == 0.)
                
                for name, param in zip(model.named_parameters(), model.parameters()):  
                    if name[0] in ['rnn.weight_hh_l0']:
                        param.grad[mask_idx] = 0.
                               
                optimizer.step()#update the weights
              
            if w is None:
                 optimizer.step()#update the weights 
                 
        # This calculates a null prediction on the current input to the model.
        # Null models could also be constructed only on the train input data
        # Different null predictions for each task.       
        if calc_null is True:
            if task_name == 'seq_mem':
                idx = torch.where(X_batch[:, :, 1] !=0)#create null mean pediction
                length = X_batch[idx[0],idx[1],1].shape[0]
                null_prediction = torch.full((length, 1), 
                                              torch.mean(X_batch[idx[0],idx[1],1]))
                loss_null = criterion(null_prediction, Y_batch)
                
            if task_name == 'pic_mem' or task_name == 'bin_mem':   
                   ind = [0,1,2]
                   random.shuffle(ind)
                   output = output[:, ind]#shuffle the positions so we have null predictions
                   loss_null = criterion(output, Y_batch)
                   
                   if 'acc' in metrics:
                       acc = auxfun.calc_accuracy(output=output, labels=Y_batch)
                       batch_metrics_null.append(acc)
                
            batch_loss_null.append(loss_null.data.item())
                           
    return batch_loss, batch_loss_null, batch_metrics, batch_metrics_null

# Test model
def test_model(
               model, 
               data_generator = None, 
               criterion = None, 
               calc_null = True,
               batch_size = 32,
               trim = True,
               metrics = None,
               task_name = None,
               device = 'cpu',
               save_hidden = False
               ): 
       
    batch_loss = []
    batch_loss_null = []
    batch_metrics = []
    
    all_hidden = []
    
    #Make sure that the model is not at the training mode
    model.train(False)
    for X_batch, Y_batch in data_generator:
        # Send tensors to the device used
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        output, hidden = model(X_batch)
        
        # Store the hidden states if requested
        if save_hidden is True:
            all_hidden.append(output.data)#output is also hidden state!
        
        Y_batch = Y_batch.contiguous().view(
            int(batch_size * (len(output)/batch_size)), 
            -1)
        
        if trim is True:
            Y_batch, output = tasks.trim_zeros_from_trials(actual=Y_batch, 
                                                             predicted=output)
        
        
        if task_name == 'pic_mem' or task_name == 'bin_mem': 
            Y_batch = Y_batch.view(Y_batch.shape[0],)
            Y_batch = Y_batch.long()
        
        loss = criterion(output, Y_batch)
        batch_loss.append(loss.data.item())
        
        if 'acc' in metrics:
            acc = auxfun.calc_accuracy(output=output, labels=Y_batch)
            batch_metrics.append(acc)
    
        # This calculates a null prediction on the current input to the model!
        # Null models could also be constructed only on the train input data                
        if calc_null is True:
            idx = torch.where(X_batch[:, :, 1] !=0)#create null mean pediction
            
            length = X_batch[idx[0],idx[1],1].shape[0]
            
            null_prediction = torch.full((length, 1), 
                                          torch.mean(X_batch[idx[0],idx[1],1]))
            
            loss_null = criterion(null_prediction, Y_batch)
            batch_loss_null.append(loss_null.data.item())
                
    return batch_loss, batch_loss_null, batch_metrics, all_hidden

# train, validate across epochs
def train_validate_epochs(
                          model, 
                          epochs = 100,
                          training_generator = None,
                          validate_generator = None, 
                          optimizer = None, 
                          criterion = None, 
                          batch_size = 32,
                          trim = True,
                          calc_null = True,
                          w = None,
                          metrics = [],
                          task_name = None,
                          device = 'cpu',
                          store_every_epoch = None,
                          folder_save_model = None,
                          iteration = None
                          ):

    epoch_train_loss=[]
    epoch_train_loss_null=[]
    
    epoch_validate_loss=[]
    epoch_validate_loss_null=[]
    
    epoch_train_metrics=[]
    epoch_train_metrics_null=[]
    
    epoch_validate_metrics=[]
    epoch_validate_metrics_null=[]
    
    # TODO N-fold validation can be implemented for each epoch. In addition
    # a random shuffling can be implemented internally as an option so that 
    # the model is not trained/validated in the same order across all epochs. 
    for epoch in range(0, epochs):
        optimizer.zero_grad() #Clear existing gradients from previous epoch        
        # Train
        (batch_loss,
         batch_loss_null,
         batch_metrics,
         batch_metrics_null) = train_validate_model(
                                                   model, 
                                                   training_generator, 
                                                   optimizer = optimizer, 
                                                   criterion = criterion, 
                                                   train = True,
                                                   calc_null = calc_null,
                                                   batch_size = batch_size,
                                                   trim = trim,
                                                   w = w,
                                                   metrics = metrics,
                                                   task_name = task_name, 
                                                   device = device
                                                   )
        
        # Store the mean of all the batch-wise training losses for the current 
        # epoch
        epoch_train_loss.append(statistics.mean(batch_loss)) 
        
        if batch_loss_null:
            epoch_train_loss_null.append(statistics.mean(batch_loss_null))
            
        if batch_metrics:
            epoch_train_metrics.append(statistics.mean(batch_metrics))  
         
        if batch_metrics_null:
            epoch_train_metrics_null.append(statistics.mean(batch_metrics_null))     
                    
        # Validate
        (batch_loss,
        batch_loss_null,
        batch_metrics,
        batch_metrics_null) = train_validate_model(
                                                  model, 
                                                  validate_generator, 
                                                  optimizer = optimizer, 
                                                  criterion = criterion, 
                                                  train = False,
                                                  calc_null = calc_null,
                                                  batch_size = batch_size,
                                                  trim = trim,
                                                  metrics = metrics,
                                                  task_name = task_name, 
                                                  device = device
                                                  )
        
        # If asked, store model every store_every_epoch th epoch, 
        # and for the last epoch 
        if store_every_epoch is not None:
            if epoch % store_every_epoch == 0 or epoch == (epochs-1):
                auxfun.save_model_state(model, 
                                         epoch = epoch, 
                                         iteration = iteration, 
                                         folder_name = folder_save_model)
             
        # Store the mean of all the batch-wise validation losses for the current 
        # epoch
        epoch_validate_loss.append(statistics.mean(batch_loss)) 
      
        if batch_loss_null:
            epoch_validate_loss_null.append(statistics.mean(batch_loss_null))
        
        if batch_metrics:
            epoch_validate_metrics.append(statistics.mean(batch_metrics)) 
            
        if batch_metrics_null:
            epoch_validate_metrics_null.append(statistics.mean(batch_metrics_null))    
           
        # Communicate where we stand
        if calc_null is False and batch_metrics:    
            print('Epoch',epoch+1,'/', epochs,
                  ' Train loss: ', epoch_train_loss[-1], 
                  ' Val loss: ', epoch_validate_loss[-1],
                  ' Train acc: ', epoch_train_metrics[-1], 
                  ' Val acc: ', epoch_validate_metrics[-1]
                  )
                  
        if calc_null is True and batch_metrics:    
            print('Epoch',epoch+1,'/', epochs,
                  ' Train loss: ', epoch_train_loss[-1], 
                  ' Train loss null: ', epoch_train_loss_null[-1],
                  ' Val loss: ', epoch_validate_loss[-1], 
                  ' Val loss null: ', epoch_validate_loss_null[-1],
                  ' Train acc: ', epoch_train_metrics[-1], 
                  ' Train acc null: ', epoch_train_metrics_null[-1], 
                  ' Val acc: ', epoch_validate_metrics[-1],
                  ' Val acc null: ', epoch_validate_metrics_null[-1],
                  )
            
        if calc_null is False and not batch_metrics:    
            print('Epoch',epoch+1,'/', epochs,
                  ' Train loss: ', epoch_train_loss[-1],
                  ' Validate loss: ', epoch_validate_loss[-1]
                  )  
                              
    # Assign the outcome of the training, validation of the model
    # to lists        
    loss = [epoch_train_loss, epoch_validate_loss]
    loss_null = [epoch_train_loss_null, epoch_validate_loss_null]
    metrics = [epoch_train_metrics, epoch_validate_metrics]
    metrics_null = [epoch_train_metrics_null, epoch_validate_metrics_null]

    return loss, loss_null, metrics, metrics_null  
