# ConvNetAbel documentation

Convolutional Neural Network made by Abel Garcia.

## Constructor parameters:

* **convFilters**: _(Type list of ints, default = [32, 64, 128])_
List of number of filters of each convolutional layer. The ith element will be the number of kernels in the ith convolutional layer.

* **convStride**: _(Type int or list of ints, default = 2)_
If _convStride_ is an int, it is the stride that will be used in all convolutional layers. For example, if the value is 2, the stride will be 2 in all convolutional layers. If _convStride_ is a list, it contains the strides that will be used in each convolutional layer. The ith element will be the stride of the ith convolutional layer. For example, if the value of _convStride_ is the list [1,2,4], the first convolutional layer will have a stride of 1, the second a stride of 2, and the third a stride equal to 4. If _convStride_ is a list, it must be the same size as the list passed to the _convFilters_ parameter.

* **convFilterSizes**: _(Type int or list of ints, default = 3)_
If _convFilterSizes_ is an int, it is the size of kernels dimensions that will be used in all convolutional layers. For example, if the value is 5 and it is detected that the input images are in grayscale, the kernels size will be 5x5, but if three-dimensional kernels are required (color images as input), all kernels of the first layer will automatically be 5x5xN where N is the number of elements in the third dimension of the input images (3 if RGB, but can be any number) and all the kernels of the rest of the layers will be 5x5 since color is only in the first layer. If _convFilterSizes_ is a list, it will contain the size of kernels dimensions that will be used in each convolutional layer. The ith element will be the size of kernels dimensions of the ith convolutional layer. For example, if the value is the list [7,5,3] and it is detected that input images are in grayscale, automatically all kernels of the first convolutional layer will be 7x7. If input images are in RGB, all kernels of the first layer will be of shape 7x7x3, kernels of the second 5x5 (color is only in the first layer) and kernels of the third 3x3. If _convFilterSizes_ is a list, it must be the same size as the list passed to the _convFilters_ parameter.

    **Example:**
    
    ```python
    clf = ConvNetAbel(convFilters=[16, 32, 64], 
                      convStride=[4,2,1], 
                      convFilterSizes=[7,5,3],
                      hidden = [10, 5])
    ```
    
    
    The example above instantiates a convolutional neural network with 3 convolutional layers and 2 fully connected hidden layers. The first convolutional layer has 16 filters, a stride of 4 and a kernel of 7x7 if it detects that input images are in grayscale, or 7x7x3 if it detects that are color images. The second convolutional layer has 32 filters with a stride of 2 and a kernel of 5x5. The third convolutional layer has 64 filters with a stride of 1 and a kernel of 3x3. After the convolutional layers it has a first fully connected hidden layer of 10 neurons and then another of 5 neurons. Finally, it has an output layer but its number of neurons will depend on the number of classes, an amount that will be known when using the fit method.

* **kernel_initializer** _(Type string, default = 'he_normal')_
Method that will be used to initialize the values of the filters of the convolutional layers. Allowed values: 

|  Parameter value | Random initialization distribution                                                          |
| :--------------: |:-------------------------------------------------------------------------------------------:| 
| 'xavier_normal'  | Normal distribution centered on 0 with a <br> standard deviation equal to sqrt(2 / (n + m)) | 
| 'xavier_uniform' | Uniform distribution in range [-k, k] <br> where k = sqrt(6 / (n + m))                      | 
| 'he_normal'      | Normal distribution centered on 0 with a <br> standard deviation equal to sqrt(2 / n)       |
| 'he_uniform'     | Uniform distribution in range [-k, k] <br> where k = sqrt(6 / n)                            |

    Where n is the size of the input to that convolutional layer and m is the size of the output of that convolutional layer.

* **hidden**: _(Type list of ints, default = [1])_
List of number of neurons of each hidden layers. The ith element will be the number of neurons in the ith hidden layer.

    Example:
    
    ```python
    hiddenLayers = [2, 4, 8]

    clf = ConvNetAbel(hidden=hiddenLayers)
    ```
    
    
    The example above instances a ConvNetAbel with 3 hidden layers of 2, 4 and 8 neurons each.
    

* **nEpochs**: _(Type int, default = 1)_ 
Number of times that the neural network will use all training data to perform the training.

* **convEpochs**: _(Type int, default = 10)_
Number of epochs in which the backpropagation will also take place in the convolutional layers. This value must be less than or equal to _nEpochs_ for this to take effect, otherwise backpropagation will also be performed in the convolutional layers during all epochs. Backpropagation is performed always in all epochs in fully connected layers.

* **learningRate**: _(Type float, default = 0.1)_
Learning rate value that controls the size of the steps when updating the weights of the fully connected layers.

* **learningRateConv** _(Type float, default = 0.001)_
Learning rate value that controls the size of the steps when updating the values of the convolutional layer filters.

* **manualWeights**: _(Type list of lists of floats, default = [])_
If it is not empty, it defines the weights of each neuron. If it is empty, all weights will be generated automatically (this is what would be usual).

    Example:
    
    In this example we will start from data of 2 dimensions so first layer size is 2. We decide to have 1 hidden layer with 2 neurons in the _hiddenLayers_ parameter. And the output size is 1 (also determined by data). So manualWeights is a list of 2 lists. The first sub-list contains the weights corresponding to the input of the hidden layer, and the second sub-list contains the weights corresponding to the input of the last layer.
    
    ```python
    hiddenLayers = [2] # 1 hidden layer with 2 neurons
    manualWeights = [[0.9,0.5,0.7,-0.7], [0.5, 0.2]]

    clf = ConvNetAbel(hidden=hiddenLayers, manualWeights=manualWeights)
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

* **activationFunction** _(Type string, default = 'leakyrelu')_ 
Activation function to be used in the fully connected layers of the neural network. Allowed values: 'sigmoid', 'relu', 'softplus', 'leakyrelu' and 'identity'.

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
If it is equal to True, it will normalize the input to the activation function of all convolutional layers and all fully-connected hidden layers. It will also normalize the value of the derivative of the error with respect to the filter prior to updating the value of the kernels.

* **shuffle** _(Type bool, default = True)_
If True, the training dataset indices will be shuffled at each epoch, for random access.

* **iterationDrop** _(Type float, default = 0)_
Probability of a iteration to be skipped in each epoch. That means there will be (1-_iterationDrop_) times fewer iterations in each epoch. For example, if the batch size is 10 and there are 1000 training samples, with _iterationDrop_=0, 100 iterations will be performed per epoch, but with _iterationDrop_=0.4, (1-0.4) * 100 iterations = 60 iterations will be performed. Combined with data shuffling at each epoch, and a sufficient number of epochs, all training data will continue to be used during training, while reducing overall run time.

Example:

```python

clf = ConvNetAbel(convFilters=[32, 64, 64], convStride=2, convFilterSizes=5,
                  hidden=[2,3], softmax=True, learningRate=0.5, nEpochs=10)

clf.fit(X_train, y_train)

probabs = clf.predict_proba(X_test)
```

## Methods:

### fit

Trains the convolutional neural network with backpropagation and gradient descent for the input images.

Returns the trained model, but it is not necessary to save it in a new variable since it modifies the variables of the already created instance.

> _fit(self, x, y)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of shape (n_images, image_dim0, image_dim1) or (n_images, image_dim0, image_dim1, image_dim2))_ Input images. Allowed in the 3 formats (color, grayscale or binary).
>
>        - **y**: _(Type Numpy array of shape (n_samples, n_classes))_ Desired output of the network (groundtruth).
>
>   - **Returns**: _(Type ConvNetAbel)_ Returns self (trained model).

Example:

```python
clf = ConvNetAbel(convFilters=[16, 32, 64], 
                  convStride=[4,2,1], 
                  convFilterSizes=[7,5,3],
                  hidden = [10, 5])

clf.fit(X_train, y_train)
```

### predict

Returns the prediction for the data indicated by parameter.

> _predict(self, x, noProba=1)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of shape (n_images, image_dim0, image_dim1) or (n_images, image_dim0, image_dim1, image_dim2))_ Input images. Allowed in the 3 formats (color, grayscale or binary).
>
>        - **noProba**: _(Type int)_ If it is 0, the prediction is returned as an estimated probability (its sum is not equal to 1). If it is 1, it returns the predicted outputs for each class.
>
>   - **Returns**: _(Type Numpy array of shape (n_samples, n_classes))_ Prediction output for the input data.

Example:

```python
clf = ConvNetAbel(convFilters=[16, 32, 64], 
                  convStride=[4,2,1], 
                  convFilterSizes=[7,5,3],
                  hidden = [10, 5])

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
>        - **x**: _(Type Numpy array of shape (n_images, image_dim0, image_dim1) or (n_images, image_dim0, image_dim1, image_dim2))_ Input images. Allowed in the 3 formats (color, grayscale or binary).
>
>
>   - **Returns**: _(Type Numpy array of shape (n_samples, n_classes))_ Prediction output for the input data, as an estimated probability (its sum is not equal to 1).

Example:

```python
clf = ConvNetAbel(convFilters=[16, 32, 64], 
                  convStride=[4,2,1], 
                  convFilterSizes=[7,5,3],
                  hidden = [10, 5])

clf.fit(X_train, y_train)

probabs = clf.predict_proba(X_test)
```

### plot_mean_error_last_layer

It shows a figure from the Matplotlib library in which it plots the mean error made by the network in the output layer in which the X axis is the epoch number and the Y axis is the mean error.

ConvNetAbel debug mode must be level 1 or higher to be able to use this method, as lowers levels does not store the information it uses to do not decrease performance.

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

ConvNetAbel debug mode must be level 2 or higher to be able to use this method, as lowers levels does not store the information it uses to do not decrease performance.

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

Draws a figure with the neurons and connections of the fully connected part of the neural network.

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
clf = ConvNetAbel(hidden=[2,3])
clf.fit(X_train, y_train)

clf.draw(showWeights=True, textSize=10, customRadius=0)
```
     

### exportModel

Export the network and its variables to a disk file so that it persists after execution and can be used again, even without the need to retrain. It performs the export using the Numpy _load_ and _save_ procedures, and generates various _npy_ files with the content of the ConvNetAbel variables. The _filename_ parameter is used for all files during that export as the name prefix.

> _exportModel(self, path='', filename='ConvNetAbel_model')_
>
>
>   - **Parameters**:
>        
>        - **path**: _(Type string, default = '')_ Path where the model will be exported.
>
>        - **filename**: _(Type string, default = 'ConvNetAbel_model')_ Prefix of the files to be generated.
>
>   - **Returns**: None

Example:

```python
mpath = 'exportedModels/'
mfilename = 'myModel'

clf = ConvNetAbel(convFilters=[8, 16], hidden=[2,3])
clf.exportModel(mpath, mfilename)
```
     
### importModel

Imports an ConvNetAbel model of _ConvNetAbel_ class from a disk file, which will keep the variables of the instance that was exported.

> _importModel(self, path='', filename='ConvNetAbel_model')_
>
>
>   - **Parameters**:
>        
>        - **path**: _(Type string, default = '')_ Path where the exported model files are located.
>
>        - **filename**: _(Type string, default = 'ConvNetAbel_model')_ Prefix of the exported files.
>
>   - **Returns**: None

Example:

```python
mpath = 'exportedModels/'
mfilename = 'myModel'

clf = ConvNetAbel() # Instance a default ConvNetAbel
clf.importModel(mpath, mfilename) # The variables will be loaded from the file.
```

## Internal attributes:

The following are internal attributes of the class, which do not need to be used by a user making conventional use of my network. However they can be used as well.

* **convFilters**: _(Type list of ints)_ List of number of filters of each convolutional layer. The ith element will be the number of kernels in the ith convolutional layer.

* **filtersValues**: _(Type list of Numpy arrays)_ Contains the arrays with the filters values of all the convolutional layers.

* **convStride**: _(Type int or list of ints, default = 2)_
If _convStride_ is an int, it is the stride that will be used in all convolutional layers. For example, if the value is 2, the stride will be 2 in all convolutional layers. If _convStride_ is a list, it contains the strides that will be used in each convolutional layer. The ith element will be the stride of the ith convolutional layer. For example, if the value of _convStride_ is the list [1,2,4], the first convolutional layer will have a stride of 1, the second a stride of 2, and the third a stride equal to 4. If _convStride_ is a list, it must be the same size as the list passed to the _convFilters_ parameter.

* **convFilterSizes**: _(Type int or list of ints, default = 3)_
If _convFilterSizes_ is an int, it is the size of kernels dimensions that will be used in all convolutional layers. For example, if the value is 5 and it is detected that the input images are in grayscale, the kernels size will be 5x5, but if three-dimensional kernels are required (color images as input), all kernels of the first layer will automatically be 5x5xN where N is the number of elements in the third dimension of the input images (3 if RGB, but can be any number) and all the kernels of the rest of the layers will be 5x5 since color is only in the first layer. If _convFilterSizes_ is a list, it will contain the size of kernels dimensions that will be used in each convolutional layer. The ith element will be the size of kernels dimensions of the ith convolutional layer. For example, if the value is the list [7,5,3] and it is detected that input images are in grayscale, automatically all kernels of the first convolutional layer will be 7x7. If input images are in RGB, all kernels of the first layer will be of shape 7x7x3, kernels of the second 5x5 (color is only in the first layer) and kernels of the third 3x3. If _convFilterSizes_ is a list, it must be the same size as the list passed to the _convFilters_ parameter.

* **kernel_initializer** _(Type string)_
Method that will be used to initialize the values of the filters of the convolutional layers. Allowed values: 

|  Parameter value | Random initialization distribution                                                          |
| :--------------: |:-------------------------------------------------------------------------------------------:| 
| 'xavier_normal'  | Normal distribution centered on 0 with a <br> standard deviation equal to sqrt(2 / (n + m)) | 
| 'xavier_uniform' | Uniform distribution in range [-k, k] <br> where k = sqrt(6 / (n + m))                      | 
| 'he_normal'      | Normal distribution centered on 0 with a <br> standard deviation equal to sqrt(2 / n)       |
| 'he_uniform'     | Uniform distribution in range [-k, k] <br> where k = sqrt(6 / n)                            |

    Where n is the size of the input to that convolutional layer and m is the size of the output of that convolutional layer.

* **convInputs**: _(Type list of Numpy arrays)_
Contains the input values for each convolutional layer so that they can be used in backpropagation. Each time the _convLayersFeedForward_ method is called, the list is flushed and populated again.

* **convOutputs** _(Type list of Numpy arrays)_
In case the number of total epochs (_numEpochs_) is greater than the number of epochs in which the backpropagation will be performed in the convolutional layers (_convEpochs_), the output value of the last convolutional layer for each input data is stored in this variable once the epoch number equal to _convEpochs_ is reached, to avoid to recalculate all the convolutions in the remaining _numEpochs_-_convEpochs_ epochs, because the value of the filters will not change any more and therefore the output of said data will not change either.

* **hiddenL**: _(Type list of ints)_ List of number of neurons of each hidden layers. The ith element will be the number of neurons in the ith hidden layer.
        
* **numEpochs**: _(Type int, default = 1)_ 
Number of times that the neural network will use all training data to perform the training.

* **convEpochs**: _(Type int, default = 10)_
Number of epochs in which the backpropagation will also take place in the convolutional layers. This value must be less than or equal to _nEpochs_ for this to take effect, otherwise backpropagation will also be performed in the convolutional layers during all epochs. Backpropagation is performed always in all epochs in fully connected layers.

* **learningRate**: _(Type float, default = 0.1)_
Learning rate value that controls the size of the steps when updating the weights of the fully connected layers.

* **learningRateConv** _(Type float, default = 0.001)_
Learning rate value that controls the size of the steps when updating the values of the convolutional layer filters.
        
* **hiddenWeights**: _(Type List of Numpy arrays)_ The ith element of the list is an array with the weights of the ith layer.
        
* **debugMode**: _(Type int)_ Contains the value of the debug level (see constructor info).
        
* **rangeRandomWeight**: _(Type tuple of floats)_
Range of values between which the weights will be randomly initialized. If None, Xavier initialization will be used if activation function is _sigmoid_ and He initialization will be used otherwise.
                
* **showLogs**: _(Type bool)_ Toggles the display of the network logs, which include information on weight modifications, errors, net outputs and much more.
        
* **softmax**: _(Type bool)_ If True, the network applies softmax before returning the prediction value.
        
* **n_layer0**: _(Type int)_ Contains the number of neurons in the input layer.

* **activationFunction** _(Type string)_
Activation function to be used in the fully connected layers of the neural network. Allowed values: 'sigmoid', 'relu', 'softplus', 'leakyrelu' and 'identity'.

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
If it is equal to True, it will normalize the input to the activation function of all convolutional layers and all fully-connected hidden layers. It will also normalize the value of the derivative of the error with respect to the filter prior to updating the value of the kernels.

* **shuffle** _(Type bool)_
If True, the training dataset indices will be shuffled at each epoch, for random access.

* **iterationDrop** _(Type float)_
Probability of a iteration to be skipped in each epoch. That means there will be (1-_iterationDrop_) times fewer iterations in each epoch. For example, if the batch size is 10 and there are 1000 training samples, with _iterationDrop_=0, 100 iterations will be performed per epoch, but with _iterationDrop_=0.4, (1-0.4) * 100 iterations = 60 iterations will be performed. Combined with data shuffling at each epoch, and a sufficient number of epochs, all training data will continue to be used during training, while reducing overall run time.


## Internal methods:

The following are internal methods of the class, which do not need to be used by a user making conventional use of my network. However they can be used as well.

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

### conv2

Returns the result of the 2d convolution for an input kernel and image.

> _conv2(self, x, kernel, stride=1)_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Numpy array of shape (image_dim0, image_dim1))_ Image to be convolved.
>
>        - **kernel**: _(Type Numpy array of shape (kernel_dim0, kernel_dim1))_ Kernel array.
>
>        - **stride**: _(Type int, default = 1)_ Number of pixels that the kernel moves in each step.
>
>
>   - **Returns**: _(Type Numpy array of shape (conv_dim0, conv_dim1))_ Result of the 2d convolution.

Example:

```python
random_image = np.random.uniform(low=0.0, high=1.0, size=(28,28))

kernel = np.random.uniform(low=0.0, high=1.0, size=(3,3))
conv_result = c.conv2(random_image, kernel)
```

### conv_filters

It can be used in three ways. The first way is passing a 2-dimensional image as a parameter _x_. It will calculate convolutions with a series of kernels (filters) specified in the _filters_ parameter.

The second way is to pass it a color image (3 dimensions). If you want to do this, you must pass it a 3 or 4-dimensional filter matrix, since if the image is 3-dimensional, the kernels must be 3-dimensional too (and a fourth dimension is only if you want to use multiple kernels at the same time). The method itself will take care of adding the 3 channels, returning an array of the same size as if a grayscale or binary image had been entered.

The third way of using this method is to pass as a parameter _x_ the result of this own method to chain convolutional layers. In this way, it will calculate the convolutions with the filters specified in the _filters_ parameter to the result of the previous convolutions (that is a 3D array passed by parameter _x_). The returned result of this method using it in this way will always be an array whose last dimension will be equal to the number of filters being used (that matches the last dimension of the _filters_ input parameter).

> _conv_filters(self, x, filters, stride=1, relu=False, mode='same')_
>
>
>   - **Parameters**:
>        
>        - **x**: _(Type Image as a Numpy array of shape (image_dim0, image_dim1) or shape (image_dim0, image_dim1, image_dim2) or previous filter convolutions outputs as Numpy array of shape (image_dim0, image_dim1, number_of_filters))_ Image to be convolved or filters to be convolved.
>
>        - **filters**: _(Type Numpy array of shape (kernel_dim0, kernel_dim1, number_of_filters_to_convolve) or (kernel_dim0, kernel_dim1))_ Filters or filter with which the image will be convolved. If array has 3 dimensions, the last dimension corresponds to the number of filters, which does not need to coincide with the number of filters in the _x_ parameter, since the input image or filters are automatically repeated to always use the number of filters defined in the kernel parameter (in its last dimension, and only if it has 3 or more dimensions).
>
>        - **stride**: _(Type int, default = 1)_ Number of pixels that the kernel moves in each step.
>
>        - **mode**: _(Type string, default = 'same')_ Convolution mode: 'same', 'valid' or 'full'.
>
>   - **Returns**: _(Type Numpy array of shape (conv_dim0, conv_dim1))_ Result of the convolutions.

**Example 1:**

To a 28x28 image, apply convolutions with 32 filters of 3x3 size. Convolve the result with 64 filters of size 3x3, and convolve that result with 128 filters of size 3x3. All this using a stride value of 2:

```python
random_image = np.random.uniform(low=0.0, high=1.0, size=(28,28))
print(random_image.shape)

filters1 = np.random.uniform(low=0.0, high=1.0, size=(3,3,32))
em = self.conv_filters(random_image, filters1, relu=True, stride=2)
print(em.shape)

filters2 = np.random.uniform(low=0.0, high=1.0, size=(3,3,64))
em = self.conv_filters(em, filters2, relu=True, stride=2)
print(em.shape)

filters3 = np.random.uniform(low=0.0, high=1.0, size=(3,3,128))
em = self.conv_filters(em, filters3, relu=True, stride=2)
print(em.shape)
```

> **Output**:
> (28, 28)
> (14, 14, 32)
> (7, 7, 64)
> (4, 4, 128)

**Example 2:**

To a 28x28x3 image (RGB colored), apply convolutions with 32 filters of 3x3x3 size. The result will be a set of convolved images in 3D, since the first layer is the one that works with colors and these 3 channels are added the first time, so 2 dimensions are of the convolutions and the third is for the number of filters. 

Convolve the result with 64 filters of size 3x3, and convolve that result with 128 filters of size 3x3. All this using a stride value of 2:

```python
random_image = np.random.uniform(low=0.0, high=1.0, size=(28,28,3))
print(random_image.shape)

filters1 = np.random.uniform(low=0.0, high=1.0, size=(3,3,3,32))
em = c.conv_filters(random_image, filters1, relu=True, stride=2)
print(em.shape)

filters2 = np.random.uniform(low=0.0, high=1.0, size=(3,3,64))
em = c.conv_filters(em, filters2, relu=True, stride=2)
print(em.shape)

filters3 = np.random.uniform(low=0.0, high=1.0, size=(3,3,128))
em = c.conv_filters(em, filters3, relu=True, stride=2)
print(em.shape)
```

> **Output**:
> (28, 28, 3)
> (14, 14, 32)
> (7, 7, 64)
> (4, 4, 128)

### kernelInitializer

Initializes the values of the filters of the ith convolutional layer according to the initialization indicated in the variable _self.kernel_initializer_.

> _kernelInitializer(self, i, ksize, inSize, outSize)_
>
>
>   - **Parameters**:
>        
>        - **i**: _(Type int)_ Convolutional layer in which it is wanted to initialize the filters.
>
>        - **ksize**: _(Type tuple of ints)_ Desired size of the filters array to be initialized in the ith layer.
>
>        - **inSize**: _(Type int)_ Total size of the input values to the ith convolutional layer.
>
>        - **outSize**: _(Type int)_ Total size of the output values of the ith convolutional layer.
>
>
>   - **Returns**: None.

### convLayersFeedForward

It calculates the feedforward part of the convolutional layers, chaining convolutions with different number of filters that had been specified in the constructor or by modifying the variable _self.convFilters_.

Filters are randomly generated if they are not initialized yet.

> _convLayersFeedForward(self, im)_
>
>
>   - **Parameters**:
>        
>        - **im**: _(Type Numpy array of shape (image_dim0, image_dim1) or (image_dim0, image_dim1, image_dim2))_ Image to be convolved.
>
>
>   - **Returns**: _(Type Numpy array of shape (n_conv_features))_ Result of the last convolutional layer.

### convLayersBackpropagation

Updates the value of the filters of the convolutional layers using ConvNet backpropagation.

> _convLayersBackpropagation(self, last_layer_output, prev_cost)_
>
>
>   - **Parameters**:
>        
>        - **last_layer_output**: _(Type Numpy array of shape (n_dim0, n_dim1, n_filters))_ Output values of the last convolutional layer.
>
>        - **prev_cost**: _(Type Numpy array of shape (n_costs))_ Vector with the errors propagated from the first fully connected layer (the one to the right of the last convolutional layer).
>
>
>   - **Returns**: None

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