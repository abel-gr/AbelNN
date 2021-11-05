# MLP_Abel documentation

Multi-Layer Perceptron classifier and regressor made by Abel Garcia.

## Constructor parameters:

* **hidden**: _(Type list of ints, default = [1])_
List of number of neurons of each hidden layers. The ith element will be the number of neurons in the ith hidden layer.

    Example:
    
```python
hiddenLayers = [2, 4, 8]

clf = MLP_Abel(hidden=hiddenLayers)
```
    
    
    The above example instances a MLP with 3 hidden layers of 2, 4 and 8 neurons each.
    

* **nEpochs**: _(Type int, default = 1)_ 
Number of times that the MLP will use all training data to perform the training.


* **learningRate**: _(Type float, default = 0.1)_
Learning rate value that controls the size of the steps when updating the weights.


* **manualWeights**: _(Type list of lists of floats, default = [])_
If it is not empty, it defines the weights of each neuron. If it is empty, all weights will be generated automatically (this is what would be usual).

    Example:
    
    In this example we will start from data of 2 dimensions so first layer size is 2. We decide to have 1 hidden layer with 2 neurons in the _hiddenLayers_ parameter. And the output size is 1 (also determined by data). So manualWeights is a list of 2 lists. The first sub-list contains the weights corresponding to the input of the hidden layer, and the second sub-list contains the weights corresponding to the input of the last layer.
    
```python
hiddenLayers = [2] # 1 hidden layer with 2 neurons
manualWeights = [[0.9,0.5,0.7,-0.7], [0.5, 0.2]]

clf = MLP_Abel(hidden=hiddenLayers, manualWeights=manualWeights)
```


* **debugLevel**: _(Type int, default = 1)_ Depending on the level of debugging, certain learning related values are stored in class instance variables to be able to later show plots with information about the learning. Each level allows all the actions of lower levels than itself. Higher levels correspond to bigger numbers.

| debugLevel    | Debug action performed |
| :-----------: |:----------------------:| 
| 0             | Debug mode disabled    | 
| 1             | The last layer mean error of each epoch is <br> calculated and stored in a list.| 
| 2             | The value of all the weights after each epoch is saved.               |   
| 3             | All output layer neurons errors <br> are saved for each input value and for each epoch. |


* **rangeRandomWeight**:  _(Type tuple of floats, default = None)_
Range of values between which the weights will be randomly initialized. If None, Xavier initialization will be used if activation function is _sigmoid_ and He initialization will be used otherwise.


* **showLogs**: _(Type bool, default = False)_ 
Toggles the display of the network logs, which include information on weight modifications, errors, net outputs and much more.


* **softmax**: _(Type bool, default = False)_ 
If True, the network applies softmax before returning the prediction value.


* **activationFunction** _(Type string, default = 'sigmoid')_ 
Activation function to be used in the neural network. Allowed values: 'sigmoid', 'relu', 'softplus', 'leakyrelu' and 'identity'.


* **verbose** _(Type bool, default = False)_
If True, when the _fit_ method is called, information about the number of training samples will be printed, and at the end of each epoch the finished epoch number and its loss will also be printed.


* **use** _(Type string, default = 'classification')_
Indicates whether the neural network should perform classification or regression, to automatically make internal modifications that allow it to perform this task, such as changing the activation function of the output layer, for example. Allowed values: 'classification' and 'regression'.


* **batch_size** _(Type int, default = 1)_
Batch size to be used.


* **batch_gradient** _(Type string, default = 'average')_
Use of the gradient calculated between the data of a batch. Allowed values: 'average' and 'sum'.


* **batch_mult** _(Type int, default = 1)_
After each epoch, _batch_size_ will be multiplied by the value of _batch_mul_.


* **dropout** _(Type float, default = 0)_
Probability of a neuron to be deactivated in each hidden layer.


* **pre_norm** _(Type bool, default = False)_
If it is equal to True, it will normalize the input to the activation function of all hidden layers.


* **shuffle** _(Type bool, default = True)_
If True, the training dataset indices will be shuffled at each epoch, for random access.


* **iterationDrop** _(Type float, default = 0)_
Probability of a iteration to be skipped in each epoch. That means there will be (1-_iterationDrop_) times fewer iterations in each epoch. For example, if the batch size is 10 and there are 1000 training samples, with _iterationDrop_=0, 100 iterations will be performed per epoch, but with _iterationDrop_=0.4, (1-0.4) * 100 iterations = 60 iterations will be performed. Combined with data shuffling at each epoch, and a sufficient number of epochs, all training data will continue to be used during training, while reducing overall run time.

Example:

```python
clf = MLP_Abel(hidden=[2,3], softmax=True)
clf.fit(X_train, y_train)

probabs = clf.predict_proba(X_test)
```

## Methods:

### fit

Trains the MLP with backpropagation and gradient descent for the input data.

Returns the trained model, but it is not necessary to save it in a new variable since it modifies the variables of the already created instance.

> _fit(self, x, y)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of shape (n_samples, n_features))_ Neural network input data.
>
>        - **y**: _(Type Numpy array of shape (n_samples, n_classes))_ Desired output of the network (groundtruth).
>
>   - **Returns**: _(Type MLP_Abel)_ Returns self (trained MLP model).

Example:

```python
clf = MLP_Abel(hidden=[2,3])
clf.fit(X_train, y_train)
```

### predict

Returns the prediction for the data indicated by parameter.

> _predict(self, x, noProba=1)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of shape (n_samples, n_features))_ Neural network input data.
>
>        - **noProba**: _(Type int)_ If it is 0, the prediction is returned as an estimated probability (its sum is not equal to 1). If it is 1, it returns the predicted classes.
>
>   - **Returns**: _(Type Numpy array of shape (n_samples, n_classes))_ Prediction output for the input data.

Example:

```python
clf = MLP_Abel(hidden=[2,3])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
```

### predict_proba

Returns the prediction output for the input data, as an estimated probability (its sum is not equal to 1).

> _predict_proba(self, x)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of shape (n_samples, n_features))_ Neural network input data.
>
>
>   - **Returns**: _(Type Numpy array of shape (n_samples, n_classes))_ Prediction output for the input data, as an estimated probability (its sum is not equal to 1).

Example:

```python
clf = MLP_Abel(hidden=[2,3])
clf.fit(X_train, y_train)

probabs = clf.predict_proba(X_test)
```

### plot_mean_error_last_layer

It shows a figure from the Matplotlib library in which it plots the mean error made by the network in the output layer in which the X axis is the epoch number and the Y axis is the mean error.

MLP debug mode must be level 1 or higher to be able to use this method, as lowers levels does not store the information it uses to do not decrease performance.

> _plot_mean_error_last_layer(self, customLabels=[], byClass=False)_
>
>
>   - **Parameters**:
>        
>        - **customLabels**: _(Type list of strings, default = [])_ List of labels to be displayed in the plot legend. If empty, in the legend it will be displayed: Neuron 0, Neuron 1...
>
>        - **byClass**: _(Type bool, default = False)_ If True, it plots the error for each neuron in the last layer separately (but in the same figure). If it is False, it plots the average error of all last layer neurons.
>
>
>   - **Returns**: None.

### plot_weights_by_epoch

It shows a figure from the Matplotlib library in which it plots the value of the final layer weights for each of the epochs. The X axis corresponds to the epoch number and the Y axis corresponds to the value of the weights.

MLP debug mode must be level 2 or higher to be able to use this method, as lowers levels does not store the information it uses to do not decrease performance.

> _plot_weights_by_epoch(self, max_weights=-1)_
>
>
>   - **Parameters**:
>        
>        - **max_weights**: _(Type int, default = -1)_ Maximum number of weights to be displayed on the plot. If it is -1, all the final layer weights will be displayed.
>
>
>   - **Returns**: None.

### draw

Draws a figure with the neurons and connections.

> _draw(self, showWeights=False, textSize=9, customRadius=0)_
>
>   - **Parameters**:
>        
>       - **showWeights**: _(Type bool, default = False)_ If True, it shows the value of the weights next to each connection.
>
>       - **textSize**: _(Type int, default = 9)_ The size of the text to be displayed of the weights of the network.
>
>       - **customRadius**: _(Type int, default = 0)_ If it is 0 or negative, the radius of the neuron drawings (drawn as circles) is automatically calculated from the total number of neurons in order to adapt the size of the drawn neurons to any size. But if you want you can indicate another positive number and that will be the radius used.
>        
>
>   - **Returns**: None

Example:

```python
clf = MLP_Abel(hidden=[2,3])
clf.fit(X_train, y_train)

clf.draw(showWeights=True, textSize=10, customRadius=0)
```
     

### exportModel

Export the network and its variables to a disk file so that it persists after execution and can be used again, even without the need to retrain. It performs the export using the Numpy _load_ and _save_ procedures, and generates various _npy_ files with the content of the MLP variables. The _filename_ parameter is used for all files during that export as the name prefix.

> _exportModel(self, path='', filename='MLP_Abel_model')_
>
>
>   - **Parameters**:
>        
>        - **path**: _(Type string, default = '')_ Path where the model will be exported.
>
>        - **filename**: _(Type string, default = 'MLP_Abel_model')_ Prefix of the files to be generated.
>
>   - **Returns**: None

Example:

```python
mpath = 'exportedModels/'
mfilename = 'myModel'

clf = MLP_Abel(hidden=[2,3])
clf.exportModel(mpath, mfilename)
```
     
### importModel

Imports an MLP model of _MLP_Abel_ class from a disk file, which will keep the variables of the instance that was exported.

> _importModel(self, path='', filename='MLP_Abel_model')_
>
>
>   - **Parameters**:
>        
>        - **path**: _(Type string, default = '')_ Path where the exported model files are located.
>
>        - **filename**: _(Type string, default = 'MLP_Abel_model')_ Prefix of the exported files.
>
>   - **Returns**: None

Example:

```python
mpath = 'exportedModels/'
mfilename = 'myModel'

clf = MLP_Abel() # Instance a default MLP
clf.importModel(mpath, mfilename) # The variables will be loaded from the file.
```

## Internal attributes:

The following are internal attributes of the class, which do not need to be used by a user making conventional use of my network. However, they can be used as well.

* **hiddenL**: _(Type list of ints)_ List of number of neurons of each hidden layers. The ith element will be the number of neurons in the ith hidden layer.
        
        
* **learningRate**: _(Type float)_ Learning rate value that controls the size of the steps when updating the weights.
        
        
* **numEpochs**: _(Type int)_ Number of times that the MLP will use all training data to perform the training.
        
        
* **hiddenWeights**: _(Type List of Numpy arrays)_ The ith element of the list is an array with the weights of the ith layer.
        
        
* **debugMode**: _(Type int)_ Contains the value of the debug level (see constructor info).
        
        
* **rangeRandomWeight**: _(Type tuple of floats)_
Range of values between which the weights will be randomly initialized. If None, Xavier initialization will be used if activation function is _sigmoid_ and He initialization will be used otherwise.
        
        
* **showLogs**: _(Type bool)_ Toggles the display of the network logs, which include information on weight modifications, errors, net outputs and much more.
        
        
* **softmax**: _(Type bool)_ If True, the network applies softmax before returning the prediction value.
        
        
* **n_layer0**: _(Type int)_ Contains the number of neurons in the input layer.


* **activationFunction** _(Type string)_ 
Activation function to be used in the neural network. Allowed values: 'sigmoid', 'relu', 'softplus', 'leakyrelu' and 'identity'.


* **lastLayerNeurons**: _(Type int)_ Contains the number of neurons in the output layer.


* **meanCostByEpoch**: _(Type list of Numpy arrays)_ It contains all the last layer neurons mean errors for each epoch. If the debug level is less than 1, the variable will be an empty list.


* **debugWeights**: _(Type list of lists of Numpy arrays)_ It contains the value of all the weights after each epoch. If the debug level is less than 2, the variable will be an empty list.


* **costs**: _(Type list of floats)_ List of all output layer neurons errors for each input value and for each epoch. If the debug level is less than 3, the variable will be an empty list.


* **manualWeights**: _(Type list of lists of floats)_ If they have been specified, it contains the weights that the user wants to manually set. If it is empty, the weights have been randomly initialized. The value of this variable is only used once in network initialization.


* **verbose** _(Type bool)_
If True, when the _fit_ method is called, information about the number of training samples will be printed, and at the end of each epoch the finished epoch number and its loss will also be printed.


* **use** _(Type string)_
Indicates whether the neural network should perform classification or regression, to automatically make internal modifications that allow it to perform this task, such as changing the activation function of the output layer, for example. Allowed values: 'classification' and 'regression'.


* **batch_size** _(Type int)_
Batch size to be used.


* **batch_gradient** _(Type string)_
Use of the gradient calculated between the data of a batch. Allowed values: 'average' and 'sum'.


* **batch_mult** _(Type int)_
After each epoch, _batch_size_ will be multiplied by the value of _batch_mul_.


* **dropout** _(Type float)_
Probability of a neuron to be deactivated in each hidden layer.


* **pre_norm** _(Type bool)_
If it is equal to True, it will normalize the input to the activation function of all hidden layers.


* **shuffle** _(Type bool)_
If True, the training dataset indices will be shuffled at each epoch, for random access.


* **iterationDrop** _(Type float)_
Probability of a iteration to be skipped in each epoch. That means there will be (1-_iterationDrop_) times fewer iterations in each epoch. For example, if the batch size is 10 and there are 1000 training samples, with _iterationDrop_=0, 100 iterations will be performed per epoch, but with _iterationDrop_=0.4, (1-0.4) * 100 iterations = 60 iterations will be performed. Combined with data shuffling at each epoch, and a sufficient number of epochs, all training data will continue to be used during training, while reducing overall run time.


## Internal methods:

The following are internal methods of the class, which do not need to be used by a user making conventional use of my network. However, they can be used as well.

### initializeWeight

Initializes the weights of the inputs to the neurons of one layer.

> _initializeWeight(self, n, i, lastN)_
>
>
>   - **Parameters**:
>        
>        - **n**: _(Type int)_ Number of neurons of the current layer (the one that we want to initialize its weights).
>
>        - **i**: _(Type int)_ Position of the layer in the network (0 for the first layer, 1 for the second layer, etc).
>
>        - **lastN**: _(Type int)_ Number of neurons of the previous layer.
>
>
>   - **Returns**: _(Type Numpy array)_ Array with the initialized weights of the inputs to the neurons of the indicated layer.

### ActivationFunction

Returns the output of the selected activation function for the input data.

> _ActivationFunction(self, x, activ_type=0)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of any shape)_ Input data.
>
>        - **activ_type**: _(Type string, default = 'sigmoid')_ Activation function selected. Allowed values: 'sigmoid', 'relu', 'softplus', 'leakyrelu' and 'identity'.
>
>
>   - **Returns**: _(Type Numpy array of same shape that input)_ Output of the selected activation function for the input data.

### functionDerivative

Returns the output of the derivative of the selected activation function for the input data.

> _functionDerivative(self, x, activ_type=0)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of any shape)_ Input data.
>
>        - **activ_type**: _(Type string, default = 'sigmoid')_ Activation function selected. Allowed values: 'sigmoid', 'relu', 'softplus', 'leakyrelu' and 'identity'.
>
>
>   - **Returns**: _(Type Numpy array of same shape that input)_ Output of the derivative of the selected activation function for the input data.


### cost

Returns the error made by the network in a prediction.

> _cost(self, y_true, y_pred)_
>
>
>   - **Parameters**:
>        
>        - **y_true**: _(Type Numpy array of shape (n_classes))_ Desired output of the network (groundtruth).
>
>        - **y_pred**: _(Type Numpy array of shape (n_classes))_ Output predicted by the network.
>
>
>   - **Returns**: _(Type Numpy array of shape (n_classes))_ Error made by the network in the prediction.

### softmaxF

Applies softmax to the values indicated by parameter, returning the converted values to probability.

This means that if you sum the returned values of each column in the same row, all the rows would be equal to 1.

> _softmaxF(self, x)_
>
>
>   - **Parameters**:
>
>        - **x**: _(Type Numpy array of shape (n_samples, n_classes) or (n_values))_ Network output values.
>
>
>   - **Returns**: _(Type Numpy array of shape (n_samples, n_classes) or (1, n_values))_ Network output values converted to probability.

Example 1:

```python
import numpy as np

sx = np.asarray([[2.3, 1.5, 2.2, 0.7]])
self.softmaxF(sx)
```
> **Output**: array([[0.39122668, 0.17578948, 0.35399654, 0.0789873 ]])

Example 2:

```python
import numpy as np

sx = np.asarray([[2.3, 1.5, 2.2, 0.7], [1.1, 5.6, 3.9, 0.2]])
self.softmaxF(sx)
```

> **Output**: array([[0.39122668, 0.17578948, 0.35399654, 0.0789873 ], [0.00927056, 0.83450923, 0.15245109, 0.00376913]])

### log

If the variable _self.showLogs_ is True, it prints the string passed by parameter.

This function is called in key zones of the network, for example to show the updates of the weights or the error made by the network. But the print will only occur if the user indicated it with the aforementioned variable.

> _log(self, *m)_
>
>
>   - **Parameters**:
>        
>        - **m**: _(Type string)_ String to print.
>
>
>   - **Returns**: None

### printVerbose

If the variable _self.verbose_ is True, it prints the string passed by parameter.

This function is called at the beginning of the _fit_ method to print the number of samples, and at the end of each epoch to print the finished epoch number and its loss.

> _printVerbose(self, *m)_
>
>
>   - **Parameters**:
>        
>        - **m**: _(Type string)_ String to print.
>
>
>   - **Returns**: None

### pre_norm_forward_FC

Normalizes the result of multiplying the input of a hidden layer by its weights and returns the result.

> _pre_norm_forward_FC(self, v_layer)_
>
>
>   - **Parameters**:
>        
>        - **v_layer**: _(Type Numpy array)_ Result of multiplying the input of a layer by its weights.
>
>
>   - **Returns**: _(Type Numpy array of same shape than input)_ Normalized layer values.