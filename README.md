# Bio-instantiated recurrent neural networks and working memory

Create and train bio-instantiated recurrent neural networks that perform working memory tasks. 

![bio_rnn_mem](bio_rnn_mem.png) 

#Description
While the interest in the relation between artifical and biological neural networks has generated a plethora of studies, attention is promarily directed towards the neurobiological plausibility of learnign rules and the similarity of visual representation in artificial and natural systems. The similarity and differences of the ***network topology*** of artificial and natural systems is examined only through the lens of abstract analogies, without a direct incorporation of the network topology found in biological neural system, such as the primate brain, in artifical neural networks.

This project instantiates the topology of biological neural networks (monkey and human brain networks) in recurrent neural networks with different strategies. The bio-instantiated artifical networks are tested in working memory tasks, that is, tasks that require tracking of information across time.

The contributions of the project are as follows:

* Highlighting the best strategy for converting empricial network data to artificial recurrent neural networks
* Demonstrating the effects of biological network topology in the performance of the artifical networks (minimization of loss, speed of training)
* Using network topology data from different species, thus examining the effects of biological network topology from a universal, cross-species standpoint 

#Usage
##Installation
Clone the repository and create a virtual environment (e.g., with conda) with the requirements.txt

##Running the experiements
All experiments can be run by executing ***biornnmemory.py***. The parameters needed are specified in the ***config.ini***

For each task, there is a template config.ini file in the ***config*** folder.

Copy the content of the .ini file that corresponds to the tasks that you want to be executed in the experiment to be run and paste it in the confi.ini file in the root directory of the repository.

There are several parameters that exist in the config.ini file, which looks like this:

```
[paths]
path_to_network = /Users/alexandrosgoulas/Bio2Art/connectomes/
path_to_results = /Users/alexandrosgoulas/rnn-bio2art-sequence-mem/test
pretrained = 

[net]
net_name = Marmoset_Normalized
rnn_size = 55
nr_neurons = 4
rand_partition = True
random_w = False

[trainvalidate]
epochs = 500
iterations = 5
init = default
freeze_layer = False
remap_w = True

[trialparams]
trial_params = {"task_name":"seq_mem", "nr_of_trials":500, "train_size":0.8}

[combosparams]
combos_params = {"lr":[0.0001,0.000001], "nonlinearity":["relu","tanh"], "optimizer":["Adam","RMSprop"], "pattern_length": [3, 5, 10]}

[pretrainedparams]
pretrained_epoch = 0
pretrained_iteration = 0
```
 
Let's see what each one of them does.

##Config.ini parameters
```
[paths]
path_to_network = /Users/alexandrosgoulas/Bio2Art/connectomes/
path_to_results = /Users/alexandrosgoulas/rnn-bio2art-sequence-mem/test
pretrained = 
```
Here the path for saving the results ```path_to_results ``` and with the location of the empirical biological networks (connectomes) ```path_to_network ``` are specified. 

 


  

