import numpy as np
import math

import copy
import random


class ModelNode:
    
    def __init__(self, model, hyperparameters, parentNode, accuracy, depth):
        
        self.model = model
        
        self.hyperparameters = hyperparameters
        
        self.parent = parentNode
        
        self.accuracy = accuracy
        
        self.depth = depth
        
        self.children = 0
        

class AutoML_Abel:
    
    def __init__(self, model, hyp_space=None, train_subset_elements=20, test_subset_elements=20, maxChildrenPerNode=3,
                 maxTreeDepth=3, prunedRatio=0.2, pruningsNumber=10, pruningDepthStep=5, verbose=False,
                 debugMode=False, nEpochs=10):
        
        self.modelBase = model
        
        self.train_subset_elements = train_subset_elements
        
        self.test_subset_elements = test_subset_elements
        
        self.maxChildrenPerNode = maxChildrenPerNode
        
        self.maxTreeDepth = maxTreeDepth
        
        self.prunedRatio = prunedRatio
        
        self.pruningsNumber = pruningsNumber
        
        self.pruningDepthStep = pruningDepthStep
        
        self.verbose = verbose
        
        self.debugMode = debugMode
        
        self.nEpochs = nEpochs
        

        
        if hyp_space is None:
            
            self.hyp_space = {'learningRate': ['range', [1e-5, 1.5]], 'batch_size': ['rangeint', [1, 64]],
                         'activationFunction': ['option', ['sigmoid', 'relu', 'leakyrelu', 'softplus']],
                         'dropout': ['range', [0, 0.8]], 'pre_norm': ['option', [False, True]]}


            if("Conv" in str(self.modelBase.__class__.__name__)):

                self.hyp_space['learningRateConv'] = ['range', [1e-5, 1.5]]
                self.hyp_space['kernel_initializer'] = ['option', ['xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform']]
                self.hyp_space['convFilterSizes'] = ['rangeint', [2, 7]]
                self.hyp_space['convStride'] = ['rangeint', [2, 3]]
                self.hyp_space['convFilters'] = ['rangeint', [8, 64]]
                self.hyp_space['convLayers'] = ['rangeint', [1, 3]]
                
        else:
            
            self.hyp_space = hyp_space
            
            
        
        if self.pruningDepthStep < math.ceil(self.pruningsNumber / (self.maxTreeDepth - 1)):
            print('pruningDepthStep must be greater than or equal to: math.ceil(pruningsNumber / (maxTreeDepth - 1))')
    
    
    def fit(self, x_train, y_train, x_test, y_test):
        
        if self.pruningDepthStep < math.ceil(self.pruningsNumber / (self.maxTreeDepth - 1)):
            print('pruningDepthStep must be greater than or equal to: math.ceil(pruningsNumber / (maxTreeDepth - 1))')
            return
        
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        

        hyp_default = {'learningRate': 0.1, 'batch_size': 1,
                       'activationFunction': 'sigmoid', 'hidden': [25],
                       'dropout': 0, 'pre_norm': False, 'nEpochs': 10}
        
        if("Conv" in str(self.modelBase.__class__.__name__)):
            
            hyp_default['learningRateConv'] = 0.1
            hyp_default['kernel_initializer'] = 'xavier_normal'
            hyp_default['convFilterSizes'] = 3
            hyp_default['convStride'] = 2
            hyp_default['convFilters'] = 8
            hyp_default['convLayers'] = 1
        
        
        treeRoot = self.initModel(hyp_default)
        treeRoot, accuracy = self.modelFitPredict(treeRoot)
        rootNode = ModelNode(treeRoot, hyp_default, None, accuracy, 0)
        
        self.treeNodes = [rootNode]
        
        bestAccuracy = accuracy
        bestModel = treeRoot
        
        if self.debugMode:
            self.bestAccuracies = []
        
                
        self.accuraciesPerDepth = {}
        
        pruneLevel = (self.maxTreeDepth - 1) # Leaves
        pruneCount = 0
        
        for pruningN in range(0, self.pruningsNumber):
            
            # Generate child nodes
            for node in self.treeNodes:

                if node.depth < (self.maxTreeDepth - 1) and node.children < self.maxChildrenPerNode:

                    for i in range(0, self.maxChildrenPerNode - node.children):
                        mutatedNode, accuracy = self.generateChild(node)
                        
                        if accuracy > bestAccuracy:
                            bestAccuracy = accuracy
                            bestModel = mutatedNode.model
                            
                        
                        if self.debugMode:
                            self.bestAccuracies.append(bestAccuracy) # To be able to display a plot

                        if (node.depth + 1) in self.accuraciesPerDepth:
                            self.accuraciesPerDepth[node.depth + 1].append([accuracy, mutatedNode])
                        else:
                            self.accuraciesPerDepth[node.depth + 1] = [[accuracy, mutatedNode]]



            if pruningN < (self.pruningsNumber - 1):
                
                # Prune worst children per depth level
                newaccuraciesPerDepth = {}
                for depthLevel, am in self.accuraciesPerDepth.items():

                    if depthLevel >= pruneLevel:

                        am.sort(key=lambda a: a[0]) # Sort accuracies per depth level

                        pruneIndex = int(len(am) * (1 - self.prunedRatio))

                        for worstAcc, worstModel in am[0:pruneIndex]:

                            worstModel.parent.children = worstModel.parent.children - 1
                            self.treeNodes.remove(worstModel) # We remove the worst models from each level

                        newaccuraciesPerDepth[depthLevel] = am[pruneIndex:]

                    else:

                        newaccuraciesPerDepth[depthLevel] = am

                self.accuraciesPerDepth = newaccuraciesPerDepth
            
            if pruneCount == (self.pruningDepthStep - 1):
                pruneCount = 0
                pruneLevel = pruneLevel - 1
            else:
                pruneCount = pruneCount + 1
                
                
            if self.verbose:
                print('AutoML pruning', str(pruningN + 1) + '/' + str(self.pruningsNumber), ' - Best accuracy found:', bestAccuracy)
          
        
        bestModel.numEpochs=self.nEpochs
        
        return bestModel
                
    
    
    def initModel(self, hyperparameters):
        
        newModel = copy.deepcopy(self.modelBase)
        
        
        if("Conv" in str(self.modelBase.__class__.__name__)):
            
            convF = [int(hyperparameters['convFilters'])] * int(hyperparameters['convLayers'])
                        
            newModel.__init__(learningRate=hyperparameters['learningRate'], 
                              batch_size=hyperparameters['batch_size'],
                              activationFunction=hyperparameters['activationFunction'], 
                              dropout=hyperparameters['dropout'],
                              pre_norm=hyperparameters['pre_norm'], 
                              hidden=hyperparameters['hidden'],
                              nEpochs=hyperparameters['nEpochs'],
                              learningRateConv=hyperparameters['learningRateConv'],
                              kernel_initializer=hyperparameters['kernel_initializer'],
                              convFilterSizes=hyperparameters['convFilterSizes'],
                              convStride=hyperparameters['convStride'],
                              convFilters=convF)
                                    
        else:
        
            newModel.__init__(learningRate=hyperparameters['learningRate'], 
                              batch_size=hyperparameters['batch_size'],
                              activationFunction=hyperparameters['activationFunction'], 
                              dropout=hyperparameters['dropout'],
                              pre_norm=hyperparameters['pre_norm'], 
                              hidden=hyperparameters['hidden'],
                              nEpochs=hyperparameters['nEpochs'])
        
        return newModel
        
        
    def generateMutation(self, parent_hyperparameters):

        modelValues = copy.deepcopy(parent_hyperparameters)

        hyp, values = random.choice(list(self.hyp_space.items())) # Choose one random mutation

        hyp_type = values[0]
        hyp_values = copy.deepcopy(values[1])

        if hyp_type == 'option':
            
            if parent_hyperparameters[hyp] in hyp_values:
                
                hyp_values.remove(parent_hyperparameters[hyp])
                            
            value = random.choice(hyp_values)
            
        else:
            value = np.random.uniform(low=hyp_values[0], high=hyp_values[1], size=(1))[0]

            if 'int' in hyp_type:
                value = value.astype(int)
            else:
                value = value.astype(float)

        modelValues[hyp] = value # Apply mutation
        
        return modelValues
    
    
    def modelFitPredict(self, model):
        
        positions = np.arange(self.x_train.shape[0])
        np.random.shuffle(positions)
        
        positions = positions[0:min(positions.shape[0], self.train_subset_elements)]

        sub_x_train = self.x_train[positions]
        sub_y_train = self.y_train[positions]

        try:
            
            model.fit(sub_x_train, sub_y_train)
        
            positions = np.arange(self.x_test.shape[0])
            np.random.shuffle(positions)

            positions = positions[0:min(positions.shape[0], self.test_subset_elements)]

            sub_x_test = self.x_test[positions]
            sub_y_test = self.y_test[positions]


            probabs = model.predict_proba(sub_x_test)
            probabs_results = np.argmax(probabs, axis=1)

            accuracy = np.sum(probabs_results == np.argmax(sub_y_test, axis=1)) / probabs_results.shape[0]
            
        except:
            
            accuracy = 0
                
        return [model, accuracy]
        
        
        
    def generateChild(self, parentNode):
                
        parentNode.children = parentNode.children + 1
            
        mutation = self.generateMutation(parentNode.hyperparameters)
        mutatedModel = self.initModel(mutation)
            
        mutatedModel, accuracy = self.modelFitPredict(mutatedModel)
            
        mutatedNode = ModelNode(mutatedModel, mutation, parentNode, accuracy, parentNode.depth + 1)
                        
        self.treeNodes.append(mutatedNode)
                
        return [mutatedNode, accuracy]
            
            
        
        