#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch
import auxfun 

# RNN with no topology
class Model(nn.Module):
    '''
    Class for constructing an Elman RNN without a structured topology of the 
    recurrent hidden layer (all-to-all)
    '''
    def __init__(self, 
                 input_size = None, 
                 output_size = None, 
                 hidden_dim = None, 
                 n_layers = 1,          
                 init = 'default',
                 nonlinearity = 'relu',
                 device = 'cpu'
                 ):
        '''
        Constructor for the RNN model
        
        Input
        -----
        input_size: int, specifying the number of input neurons
        
        output_size: int, specifying the number of output neurons 
        
        hidden_dim: int, specifying the number of neurons of the hidden 
            recurrent layer
       
        n_layers: int, default 1, specifying the number of hidden layers
        
        init: str {'default','he','xavier'}, default 'default', specifying 
            what type of weight initialization will be used 
            
        nonlinearity: str {'tanh','relu'}, default 'relu', specifying the 
            activation function
            
        device: str {'cpu','gpu'}, default 'cpu', specifying the device to be
            used
        '''
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
            w_hh = torch.empty(hidden_dim, hidden_dim)
            w_ih = torch.empty(hidden_dim, input_size) 
            
            w_hh = w_hh.to(self.device)
            w_ih = w_ih.to(self.device)
            
            nn.init.xavier_uniform_(w_hh)
            nn.init.xavier_uniform_(w_ih)
            
            self.rnn.weight_ih_l0=torch.nn.Parameter(w_ih)    
            self.rnn.weight_hh_l0=torch.nn.Parameter(w_hh)
        
        # TODO:  biases??!                   
        if init == 'he':
            w_hh = torch.empty(hidden_dim, hidden_dim)
            w_ih = torch.empty(hidden_dim, input_size)
            
            w_hh = w_hh.to(self.device)
            w_ih = w_ih.to(self.device)
            
            nn.init.kaiming_uniform_(w_hh, mode='fan_out')
            nn.init.kaiming_uniform_(w_ih, mode='fan_out')
            
            self.rnn.weight_ih_l0=torch.nn.Parameter(w_ih)    
            self.rnn.weight_hh_l0=torch.nn.Parameter(w_hh)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if init == 'xavier':
            w_fc = torch.empty(output_size, hidden_dim)
            nn.init.xavier_uniform_(w_fc)
            self.fc.weight = torch.nn.Parameter(w_fc)
            
        if init == 'he':
            w_fc = torch.empty(output_size, hidden_dim)
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
                              self.hidden_dim)
        
        hidden = hidden.to(self.device)     

        return hidden

# Tweek the model to instantiate biological network topology.
# This will be implemnented with inheritance to avoid replication of common
# functions with the "normal" Model rnn (forward pass etc).        
class ModelBio(Model):
    '''
    Class for constructing an Elman RNN with a topology of the 
    recurrent hidden layer (specified by parameter w)
    '''
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
        '''
        Constructor for the RNN model with topology
        
        Input
        -----
        input_size: int, specifying the number of input neurons
        
        output_size: int, specifying the number of output neurons 
        
        hidden_dim: int, specifying the number of neurons of the hidden 
            recurrent layer
       
        n_layers: int, default 1, specifying the number of hidden layers
        
        w: torch.Tensor of shape (N,N) denoting the topology of the hidden
            recurrent layer. The topology can be viewed both as binary, i.e.
            neuron-to-neuron conenctions, and weighted,i.e. how strong two
            neurons are connected.
            
            if w(i,j) == 0 then neurons i, j are NOT connected
            if w(i,j) != 0 then neurons i, j are connected with a strength 
            that will be dictated by the magnitude of w(i,j)
            
            NOTE: w(i,j) values are used to define the corresponding weight
            after the initialization corresponding to the parameter init.
            Thus, w(i,j) as such will not be the weight for the RNN,
            but a rank ordered equal (if remap_w=True) or a random weight
            (if remap_w=False)
        
        remap_w: bool, default True, specifying if the valeus of w should be 
            used for rank ordering the weights after the initialization 
            scheme specified by the parameter init.
            
            If True, then rank(w(i,j))==rank(w'(i,j)) where w' the tensor
            with all the weights of the hidden recurrent layer after the 
            initialization based on parameter init.
            
            If False, no such weight remapping is performed.
        
        init: str {'default','he','xavier'}, default 'default', specifying 
            what type of weight initialization will be used 
            
        nonlinearity: str {'tanh','relu'}, default 'relu', specifying the 
            activation function
            
        device: str {'cpu','gpu'}, default 'cpu', specifying the device to be
            used
        '''
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
            w_hh = self.rnn.weight_hh_l0.detach().clone()
            
        # TODO:  biases??!    
        if init == 'xavier':
            w_hh = torch.empty(hidden_dim, hidden_dim)
            w_ih = torch.empty(hidden_dim, input_size) 
            
            nn.init.xavier_uniform_(w_hh)
            nn.init.xavier_uniform_(w_ih)
            
            self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=True)
            
        if init == 'he':
            w_hh = torch.empty(hidden_dim, hidden_dim)
            w_ih = torch.empty(hidden_dim, input_size) 
            
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
            w_fc = torch.empty(output_size, hidden_dim) 
            nn.init.xavier_uniform_(w_fc)
            self.fc.weight = torch.nn.Parameter(w_fc, requires_grad=True)
            
        if init == 'he': 
            w_fc = torch.empty(output_size, hidden_dim)
            nn.init.kaiming_uniform_(w_fc, mode='fan_out')
            self.fc.weight = torch.nn.Parameter(w_fc, requires_grad=True)
        
        # If the output dimension is >1 then we deal with a classification 
        # task. So add a softmax.
        if self.output_size > 1:
            self.softmax = nn.LogSoftmax(dim=1)   
        
class ModelBio_Modified(nn.Module):
    '''
    This class is used to modify the architecture wih the modules specified
    in the new_modules list. These modules are appended after removing the last
    layer (default) from the model.  
    '''
    def __init__(self, 
                 model = None, 
                 n_last_layers = -1,
                 new_modules = None
                 ):
        '''
        Input
        -----
        model: model that is an instantiation of a class of nn.Module or a 
            class with such inheritance
            
        n_last_layers: int, default -1, specifying which N last layers will 
            be removed from the model. The default -1 will remove the last 
            layer, -2 the two last layers, etc    
        
        new_modules: list of modules to be added to the model after the 
            removal of the n_last_layers. Modules of the list must be an object
            from module torch.nn.modules
        '''
        super(ModelBio_Modified, self).__init__()
        # assign values from the model to the new one
        self.hidden_dim = model.hidden_dim
        self.n_layers = model.n_layers
        self.device = model.device
        
        # Get all the modules of the model in a list 
        module_list = nn.ModuleList(model.children())
        
        #Remove the n_last_layers
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
    
    