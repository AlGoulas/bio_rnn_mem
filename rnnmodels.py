#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch
import auxfun 

# RNN with no topology
class Model(nn.Module):
    def __init__(self, 
                 input_size = None, 
                 output_size = None, 
                 hidden_dim = None, 
                 n_layers = None,          
                 init = 'default',
                 nonlinearity = 'tanh',
                 device = 'cpu'
                 ):
        super(Model, self).__init__()

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.device = device

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, 
                          hidden_dim, 
                          n_layers, 
                          batch_first = True,
                          nonlinearity = nonlinearity
                          )
                        
        # TODO:  biases??!    
        if init == 'xavier':
            w_hh = torch.empty(hidden_dim, hidden_dim).double()
            w_ih = torch.empty(hidden_dim, input_size).double() 
            
            w_hh = w_hh.to(self.device)
            w_ih = w_ih.to(self.device)
            
            nn.init.xavier_uniform_(w_hh)
            nn.init.xavier_uniform_(w_ih)
            
            self.rnn.weight_ih_l0=torch.nn.Parameter(w_ih)    
            self.rnn.weight_hh_l0=torch.nn.Parameter(w_hh)
        
        # TODO:  biases??!                   
        if init == 'he':
            w_hh = torch.empty(hidden_dim, hidden_dim).double()
            w_ih = torch.empty(hidden_dim, input_size).double() 
            
            w_hh = w_hh.to(self.device)
            w_ih = w_ih.to(self.device)
            
            nn.init.kaiming_uniform_(w_hh, mode='fan_out')
            nn.init.kaiming_uniform_(w_ih, mode='fan_out')
            
            self.rnn.weight_ih_l0=torch.nn.Parameter(w_ih)    
            self.rnn.weight_hh_l0=torch.nn.Parameter(w_hh)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if init == 'xavier':
            w_fc = torch.empty(output_size, hidden_dim).double()
            nn.init.xavier_uniform_(w_fc)
            self.fc.weight = torch.nn.Parameter(w_fc)
            
        if init == 'he':
            w_fc = torch.empty(output_size, hidden_dim).double()
            nn.init.kaiming_uniform_(w_fc, mode='fan_in')
            self.fc.weight = torch.nn.Parameter(w_fc)
        
        # If the output dimension is >1 then we deal with a classification 
        # task. So add a softmax.
        if output_size > 1:
            self.softmax = nn.LogSoftmax(dim=1)
            
    def forward(self, x):   
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining 
        # outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully 
        # connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        # If the output dimension is >1 then we deal with a classification 
        # task. So make another pass through the softmax.
        if self.output_size > 1:
            out = self.softmax(out)
             
        out = out.to(self.device) 
        hidden = hidden.to(self.device) 
        
        return out, hidden
    
    def init_hidden(self, batch_size): 
        # Generate the first hidden state (all zeros) which we'll 
        # use in the forward pass. This initialization takes place at
        # every batch.
        hidden = torch.zeros(self.n_layers, 
                              batch_size, 
                              self.hidden_dim).double()
        
        hidden = hidden.to(self.device)     

        return hidden

# Tweek the model to instantiate biological network topology.
# This will be implemnented with inheritance to avoid replication of common
# functions with the "normal" Model rnn (forward pass etc).        
class ModelBio(Model):
    def __init__(self, 
                 input_size = None, 
                 output_size = None, 
                 hidden_dim = None, 
                 n_layers = 1, 
                 w = None,
                 remap_w = True,
                 init = 'default',
                 nonlinearity = 'tanh',
                 device = 'cpu'
                 ):
        nn.Module.__init__(self)

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.device = device

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, 
                          hidden_dim, 
                          n_layers, 
                          batch_first = True,
                          nonlinearity = nonlinearity
                          )
        
        # Track the positions to be zeroed in the hh weights tensor
        idx = torch.where(w == 0)
            
        # All we have to do is set the values for w_hh and threshold the 
        # tensor so that it corresponds to the topology of w. 
        #
        # 3 intit schemes: default, xavier(uniform), he(uniform)
        #
        # Note that for the default w_ih, w_hh and linear out (fc) are already
        # initialized automatically when specified. So if default,
        # just access w_hh to apply threshold  
        if init == 'default':
            w_hh = self.rnn.weight_hh_l0.detach().clone().double()
            
        # TODO:  biases??!    
        if init == 'xavier':
            w_hh = torch.empty(hidden_dim, hidden_dim).double()
            w_ih = torch.empty(hidden_dim, input_size).double() 
            
            nn.init.xavier_uniform_(w_hh)
            nn.init.xavier_uniform_(w_ih)
            
            self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=True)
            
        if init == 'he':
            w_hh = torch.empty(hidden_dim, hidden_dim).double()
            w_ih = torch.empty(hidden_dim, input_size).double() 
            
            nn.init.kaiming_uniform_(w_hh, mode='fan_out')
            nn.init.kaiming_uniform_(w_ih, mode='fan_out')
            
            self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, 
                                                       requires_grad=True)    
     
        w_hh[idx] = 0.# threshold to 0 so that it corresponds to topology
        
        # Redistribute the weights so that their rank ordering adheres to
        # the rank ordering of the biological network weights.
        if remap_w is True:
            w_hh = auxfun.map_weights_to_template(w_template = w, 
                                                   w_to_map = w_hh)
        
        # Assign the weights to the hh layer
        self.rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=True)#we have detached it from the graph so mark it as requires grad again
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if init == 'xavier':
            w_fc = torch.empty(output_size, hidden_dim).double() 
            nn.init.xavier_uniform_(w_fc)
            self.fc.weight = torch.nn.Parameter(w_fc, requires_grad=True)
            
        if init == 'he': 
            w_fc = torch.empty(output_size, hidden_dim).double()
            nn.init.kaiming_uniform_(w_fc, mode='fan_out')
            self.fc.weight = torch.nn.Parameter(w_fc, requires_grad=True)
        
        # If the output dimension is >1 then we deal with a classification 
        # task. So add a softmax.
        if self.output_size > 1:
            self.softmax = nn.LogSoftmax(dim=1)   

# This class is used to modify the architecture wih the modules specified
# in the new_modules list. These modules are appended after removing the last
# layer from the model            
class ModelBio_Modified(nn.Module):
    def __init__(self, 
                 model=None, 
                 n_last_layers=-1,
                 new_modules=None
                 ):

        super(ModelBio_Modified, self).__init__()
        # assign values from the model to the new one
        self.hidden_dim = model.hidden_dim
        self.n_layers = model.n_layers
        self.device = model.device
        
        # Get all the modules of the model in a list 
        module_list = nn.ModuleList(model.children())
        
        #Remove the n last layers
        module_list = module_list[:n_last_layers]
        
        # Append the modules (if specified) 
        # that are specified as input args to the modified model
        if new_modules is not None: 
            for i, item in enumerate(new_modules):
                    module_list.append(item)
        
        # Unpack the list of modules in a Sequential module
        self.features = nn.Sequential(*module_list)# Build the new model with the specifications in module_list

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        
        # Loop throught the features of self and make the approariate 
        # forward calculations based on what type of sequential features
        # the model consists of.
        for i, model_curent in enumerate(self.features):
            
            if type(model_curent) is torch.nn.modules.rnn.RNN:
                out, hidden = model_curent(x, hidden)
                
            if type(model_curent) is torch.nn.modules.linear.Linear:
                out = out.contiguous().view(-1, self.hidden_dim)
                out = model_curent(out)  
            
            if type(model_curent) is torch.nn.modules.activation.LogSoftmax:
                out = model_curent(out)
                    
        out = out.to(self.device) 
        hidden = hidden.to(self.device) 
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll 
        # use in the forward pass
        hidden = torch.zeros(self.n_layers, 
                             batch_size, 
                             self.hidden_dim).double()
        
        hidden = hidden.to(self.device)     

        return hidden
    
    