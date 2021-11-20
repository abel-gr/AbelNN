# AutoML_Abel documentation

Auto Machine Learning hyper-parameter auto tuner implementation made by Abel Garcia.

Made from scratch as an evolutionary algorithm in the form of a tree, with mutations of children (machine learning models) and pruning of branches to optimize.

**Example:**

```python
import AbelNN

# Instance a default model.
c = AbelNN.ConvNet()
    
# Pass the ConvNet to the AutoML. It returns the best model for your data.
clf = AbelNN.AutoML(c).fit(x_train, y_train, x_test, y_test)

# Train the auto tuned model.
clf.fit(x_train, y_train)

# Use the model to predict your test data.
clf.predict(x_test, y_test)
```

## Constructor parameters:

All constructor parameters are optional, except *model*.

* **model**: _(Type class)_
Model to be auto tuned.


* **hyp_space**: _(Type Dictionary, default = None)_
Dictionary where keys are names of hyperparameters to be tuned and values are lists with two elements. The first element can be 'range', 'rangeint' or 'option'. The second item is a list whose values and structure depend on the first element, as indicated in the following table. The third column shows the resulting mutation if the AutoML chooses to mutate that hyperparameter. The AutoML module will choose random mutations among the different hyperparameters specified in the dictionary. If its value is *None*, it uses a predefined search space for neural networks from the AbelNN library.

| First item (string) | Second item (list)                        | Type of the list members <br> of the second item |   Mutation result    |
|:-------------:|:----------------------------------------:|:-------------------:|:--------------------:|
| 'range'       | [minValue, maxValue]                     | Any numerical type  | Random value within <br> the specified range |
| 'rangeint'    | [minValue, maxValue]                     | int                 | Random value within <br> the specified range as type int |
| 'option'      | [value_0, value_1, ... , value_n]        | Any type            | Random choice from list |

    Example:

```python

hyp_space = {'learningRate': ['range', [1e-5, 1.5]], 'batch_size': ['rangeint', [1, 64]],
             'activationFunction': ['option', ['sigmoid', 'relu', 'leakyrelu', 'softplus']],
             'dropout': ['range', [0, 0.8]], 'pre_norm': ['option', [False, True]]}
```


* **train_subset_elements**: _(Type int, default = 20)_
Number of elements in the train subset to be used to train each model. Random shuffle is always performed before each subsampling. If the number is greater than the number of items, all items will be used.


* **test_subset_elements**: _(Type int, default = 20)_
Number of elements in the test subset to be used to test each model. Random shuffle is always performed before each subsampling. If the number is greater than the number of items, all items will be used.


* **maxChildrenPerNode**: _(Type int, default = 3)_
Maximum number of children that each node (model) of the tree can have.


* **maxTreeDepth**: _(Type int, default = 3)_
Maximum depth of the tree, starting to count from zero, that is, also counting the root.


* **prunedRatio**: _(Type float, default = 0.2)_
Proportion of nodes that are pruned at each depth level of the tree. It must be a float between 0 and 1.


* **pruningsNumber**: _(Type int, default = 10)_
Number of prunings that will be performed, that is, number of iterations that the algorithm will execute.


* **pruningDepthStep**: _(Type int, default = 5)_
Number of iterations until a new depth level can be pruned. For instance, if it is equal to 100 and _pruningsNumber_ is 300, during the first 100 iterations only leaves can be removed, for the next 100 iterations, leaves and nodes at level (n-1) are removable, and during the last 100 iterations depths (n-2) and higher can be pruned.


* **verbose**: _(Type bool, default = False)_
If its value is True, information on the progress of the algorithm's execution will be displayed.


* **debugMode**: _(Type bool, default = False)_
If its value is True, the best accuracy found will be stored in a list in each iteration.


* **nEpochs**: _(Type int, default = 10)_
Number of epochs that the final model will have.


## Class methods:

### fit

Finds the best model with auto hyper-parameter tuning and returns it initialized with its hyperparameters.

> _fit(self, x_train, y_train, x_test, y_test)_
>
>
>   - **Parameters**:
>        
>    * **x_train**: _(Type Numpy array of shape (n_train_samples, dim0, dim1, ..., dimN))_ X values of the train subset of the dataset.
>
>    * **y_train**: _(Type Numpy array of shape (n_train_samples, n_classes))_ Y values of the train subset of the dataset.
>
>    * **x_test**: _(Type Numpy array of shape (n_test_samples, dim0, dim1, ..., dimN))_ X values of the test subset of the dataset.
>
>    * **y_test**: _(Type Numpy array of shape (n_test_samples, n_classes))_ Y values of the test subset of the dataset.
>
>
>   - **Returns**: _(Type class)_ New model instantiated and initialized with the best hyperparameter configuration found for the data.


## Internal methods:

The following are internal methods of the class, which do not need to be used by a user making conventional use of it. However, they can be used as well.

### initModel

Finds the best model with auto hyper-parameter tuning.

> _initModel(self, hyperparameters)_
>
>
>   - **Parameters**:
>        
>        - **hyperparameters**: _(Type dictionary)_ Dictionary where keys are hyperparameter names and values are the value of that hyperparameter.
>
>
>   - **Returns**: _(Type class)_ Model instantiated and initialized with the hyperparameter values specified in the _hyperparameters_ variable.

### generateMutation

Finds the best model with auto hyper-parameter tuning.

> _generateMutation(self, parent_hyperparameters)_
>
>
>   - **Parameters**:
>        
>        - **parent_hyperparameters**: _(Type dictionary)_ Dictionary where keys are hyperparameter names and values are the value of that hyperparameter in parent.
>
>
>   - **Returns**: _(Type dictionary)_ The same dictionary of _parent_hyperparameters_ but with one of the hyperparameters (randomly selected) modified with a random mutation.

**Example:**

```python

hyp_default = {'learningRate': 0.1, 'batch_size': 1,
               'activationFunction': 'sigmoid', 'hidden': [10],
               'dropout': 0, 'pre_norm': False, 'nEpochs': 5}

mutated = self.generateMutation(hyp_default)
```


### modelFitPredict

Trains and tests the model passed by parameter.

> _modelFitPredict(self, model)_
>
>
>   - **Parameters**:
>        
>        * **model**: _(Type class)_ Model to be trained and tested.
>
>
>   - **Returns**: _[model, accuracy]_
>
>        * **model**: _(Type class)_ Trained model.
>        * **accuracy**: _(Type float)_ Accuracy obtained after classifying a shuffled subset of the test set with the trained model within the function.


### generateChild

Generates a mutated child of a node, then initializes it, trains it, and tests it. All this is done by calling the _initModel_, _generateMutation_ and _modelFitPredict_ functions. Finally it adds the node to the list of nodes.

> _generateChild(self, parentNode)_
>
>
>   - **Parameters**:
>        
>        * **parentNode**: _(Type class ModelNode)_ Node of which you want to create a mutated child.
>
>
>   - **Returns**: _[model, accuracy]_
>
>        * **mutatedNode**: _(Type class ModelNode)_ Node of trained model.
>        * **accuracy**: _(Type float)_ Accuracy obtained after classifying a shuffled subset of the test set with the trained model within the function.



## Internal attributes:

The following are internal attributes of the class, which do not need to be used by a user making conventional use of it. However, they can be used as well.

* **modelBase**: _(Type class)_
Copied instance of the model specified in the constructor.


* **hyp_space**: _(Type Dictionary)_
Dictionary where keys are names of hyperparameters and values are lists with two elements. The first element can be 'range', 'rangeint' or 'option'. The second element is a list whose values and structure depend on the first element, as indicated in the following table:

| First element | Second element                           | Second element type |   Mutation result    |
|:-------------:|:----------------------------------------:|:-------------------:|:--------------------:|
| 'range'       | [minValue, maxValue]                     | Any numerical type  | Random value within <br> the specified range |
| 'rangeint'    | [minValue, maxValue]                     | int                 | Random value within <br> the specified range as type int |
| 'option'      | [value_0, value_1, ... , value_n]        | Any type            | Random choice from list |

    Example:

```python

hyp_space = {'learningRate': ['range', [1e-5, 1.5]], 'batch_size': ['rangeint', [1, 64]],
             'activationFunction': ['option', ['sigmoid', 'relu', 'leakyrelu', 'softplus']],
             'dropout': ['range', [0, 0.8]], 'pre_norm': ['option', [False, True]]}
```


* **x_train**: _(Type Numpy array of shape (n_train_samples, dim0, dim1, ..., dimN))_
X values of the train subset of the dataset.


* **y_train**: _(Type Numpy array of shape (n_train_samples, n_classes))_
Y values of the train subset of the dataset.


* **x_test**: _(Type Numpy array of shape (n_test_samples, dim0, dim1, ..., dimN))_
X values of the test subset of the dataset.


* **y_test**: _(Type Numpy array of shape (n_test_samples, n_classes))_
Y values of the test subset of the dataset.


* **train_subset_elements**: _(Type int)_
Number of elements in the train subset to be used to train each model. Random shuffle is always performed before each subsampling. If the number is greater than the number of items, all items will be used.


* **test_subset_elements**: _(Type int)_
Number of elements in the test subset to be used to test each model. Random shuffle is always performed before each subsampling. If the number is greater than the number of items, all items will be used.


* **maxChildrenPerNode**: _(Type int)_
Maximum number of children that each node (model) of the tree can have.


* **maxTreeDepth**: _(Type int)_
Maximum depth of the tree, starting to count from zero, that is, also counting the root.


* **prunedRatio**: _(Type float)_
Proportion of nodes that are pruned at each depth level of the tree. It must be a float between 0 and 1.


* **pruningsNumber**: _(Type int)_
Number of prunings that will be performed, that is, number of iterations that the algorithm will execute.


* **pruningDepthStep**: _(Type int)_
Number of iterations until a new depth level can be pruned. For instance, if it is equal to 100 and _pruningsNumber_ is 300, during the first 100 iterations only leaves can be removed, for the next 100 iterations, leaves and nodes at level (n-1) are removable, and during the last 100 iterations depths (n-2) and higher can be pruned.


* **verbose**: _(Type bool)_
If its value is True, information on the progress of the algorithm's execution will be displayed.


* **debugMode**: _(Type bool)_
If its value is True, the best accuracy found will be stored in a list in each iteration.


* **nEpochs**: _(Type int)_
Number of epochs that the final model will have.


* **treeNodes**: _(Type list)_
List containing all nodes of the tree models.


* **accuraciesPerDepth**: _(Type dictionary)_
Dictionary in which the keys are the depth number (except level 0 of the root which is not added to this dictionary because the root is never removed from the tree) and the values are lists of two elements, the first contains an accuracy and the second contains the node of the model that has obtained that accuracy.


