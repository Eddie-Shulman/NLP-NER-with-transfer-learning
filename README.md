# NLP-NER-with-transfer-learning

# Pre requisites
## Hardware
1. CPU i7-4720HQ
2. 16GB of RAM
3. NVIDIA GPU (The code was tested with NVIDIA GTX 980M)
4. 5GB of hard drive for the project and saved model weights

## Software
1. Python 3.7
2. Pip3 19.0.3+

# Installation
1. Create a python virtual environment for the project (not mandatory but strongly encouraged) 
2. Install the requirements file. From project root run: ```pip install -r requirements.txt```
3. Create a ```train``` folder under the project root

# Project structure
The project consists of the following components:
1. ```Dataset``` module which contains helper methods who return the complete, train or test 
data-sets to be used in the various experiments.
2. ```Models``` module which contains helper methods for dynamically compile a NN model 
by a given configuration and load weights from a TF checkpoint, if such is given
3. ```Experiment``` module which is an abstract class providing platform for conducting experiments. It handle an experiment configuration
consisting of the model to compile, train data-set and test data-set, and previous model weights (checkpoints)
to load then runs the training and the prediction.
4. ```ExperimentSoloTraining, ExperimentTransferCrfTraining, ExperimentTransferTddTraining``` modules 
which extend the ```Experiment``` module and contain the configurations for the 
various experiments.  
5. ```TrainingAnalyzer``` module which is runs the predictions and analyzes the 
results, printing out the F1 scores of the experiment for both the CEG (complete entity
group) and the -OG (CEG without the O entity label) 

# Executing the experiments
## Conventional training
In conventional training experiments I'll train the model over some train data-sets and 
test the results over the test portion of the same data-sets. This will be my base line
for comparison of the transfer learning performance.  

The configuration of the experiments is defined in ```ExperimentSoloTraining``` module 
and consists of the following experiments:
1. Train and test over the Ritter data-set with Elmo and naive embedding, with CRF or TDD 
as the output layer.
2. Train and test over the BTC data-set with Elmo embedding and CRF output layer

To run an experiment you'll need to uncomment it's configuration (all other experiments 
MUST be commented out!). Then run ```python -m NERTranserLearning/ExperimentSoloTraining.py```
from the project root. 

This will trigger the training process, if the training was previously completed - the 
weights for the model will be loaded from the checkpoint file.
After the training is complete, the testing will begin. In the end of the experiment
2 numbers will be printed, those are the F1 scores for the CEG and the -OG (annotated as -O).

## Transfer learning experiments
In transfer learning experiments I'll train the models over various data-sets, complete
or partial, and test the results over the test portion of those data-sets BUT also test
the prediction performance over other data-sets, not related to the training data-sets.

In addition, the weights learned in each experiment are saved and used in the 
following experiment thus the order of the experiments is crucial for the code to
run and MUST be executed one after the other from top to bottom.

The experiments are split into 2 modules: ```ExperimentTransferCrfTraining, ExperimentTransferTddTraining```
and consist of the same experiments except for the output layer in the generated models. 
For ```ExperimentTransferCrfTraining``` the output layer in the model would CRF and in  
```ExperimentTransferTddTraining``` would be TDD.

The configuration of the experiments consists of the following experiments:
1. Train over GMB data-set, with naive and Elmo embeddings, and test the predictions
over the BTC and Ritter data-sets.
2. Continue the training from step 1, and train over the BTC TRAIN data-set, with naive and Elmo embeddings, and test the predictions 
over the BTC TEST and Ritter data-sets.
3. Continue the training from step 1, and train over the BTC data-set, with naive and Elmo embeddings, and test the predictions 
over the Ritter data-sets.
4. Continue the training from step 3, and train over the Ritter TRAIN data-set, with naive and Elmo embeddings, and test the predictions 
over the Ritter TEST data-sets.
5. Continue the training from step 3, and train over the Ritter data-set, with naive and Elmo embeddings, and test the predictions 
over the WSJ data-sets.
6. Continue the training from step 5, and train over the WSJ TRAIN data-set, with naive and Elmo embeddings, and test the predictions 
over the WSJ TEST data-sets.

To run an experiment you'll need to select a module (```ExperimentTransferCrfTraining or ExperimentTransferTddTraining```), uncomment it's configuration (all other experiments 
MUST be commented out!). Then run ```python -m NERTranserLearning/ExperimentTransferCrfTraining.py``` or
```python -m NERTranserLearning/ExperimentTransferTddTraining.py``` 
from the project root. 

This will trigger the training process, if the training was previously completed - the 
weights for the model will be loaded from the checkpoint file.
If the experiment is expecting weights for the model based on previous experiments, it
will try to load them. If they are not found - the experiment will crash.
After the training is complete, the testing will begin. In the end of the experiment
2 numbers will be printed, those are the F1 scores for the CEG and the -OG (annotated as -O).

# Acknowledgements
1. BTC data-set - https://github.com/juand-r/entity-recognition-datasets
2. Ritter data-set - https://github.com/aritter/twitter_nlp
3. GMB data-set - https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus