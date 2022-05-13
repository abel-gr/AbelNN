# Copyright Abel Garcia. All Rights Reserved.
# https://github.com/abel-gr/AbelNN

import numpy as np
import copy as copy
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import text
import math


class MLP_Abel:

    version = 1.2
    
    def __init__(self, hidden = [1], nEpochs = 1, learningRate=0.1, manualWeights=[], 
                 debugLevel=1, rangeRandomWeight=None, showLogs=False, softmax=False,
                 activationFunction='sigmoid', verbose=False, use='classification',
                 batch_size=1, batch_gradient='average', batch_mult=1, dropout=0, pre_norm=False,
                 shuffle=True, iterationDrop=0):
        
        self.hiddenL = copy.deepcopy(hidden)
        self.hiddenL2 = copy.deepcopy(hidden)
        
        self.learningRate = learningRate
        
        self.numEpochs = nEpochs
        
        self.costs = [] # Costs list to check performance
        
        self.debugWeights = []
        
        self.meanCostByEpoch = []
        
        self.hiddenWeights = []
        
        self.manualWeights = manualWeights
        
        self.debugMode = debugLevel
        
        self.rangeRandomWeight = rangeRandomWeight
        
        self.showLogs = showLogs
        
        self.softmax = softmax
        
        self.n_layer0 = -1
        
        self.activationFunction = activationFunction
        
        self.verbose = verbose
        
        self.use = use
        
        self.batch_size = batch_size
        
        self.batch_gradient = batch_gradient
        
        self.batch_mult = batch_mult
        
        self.dropout = dropout
        
        self.pre_norm = pre_norm
        
        self.shuffle = shuffle
        
        self.iterationDrop = iterationDrop
        
        self.XavierInitialization = '1'

        self.lastLayerNeurons = -1
        
        
    def draw(self, showWeights=False, textSize=9, customRadius=0):
        plt.figure(figsize=(10,8))

        fig = plt.gcf()
        ax = fig.gca()

        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0, ymax=1)

        xmin, xmax, ymin, ymax = ax.axis()

        xdim = xmax - xmin
        ydim = ymax - ymin

        space_per_layer = xdim / (len(self.hiddenL) + 1)

        x0 = xmin
        x1 = xmin + space_per_layer

        medio_intervalo = space_per_layer / 2

        if customRadius <= 0:
            radio = 1 / ((sum(self.hiddenL) + self.n_layer0) * 5)
        else:
            radio = customRadius

        lista_lineas_xy = []
        
        lasth = self.n_layer0

        for capa,h in enumerate([self.n_layer0] + self.hiddenL):
            space_per_neuron = ydim / h
            y0 = ymin
            y1 = ymin + space_per_neuron
            medio_intervalo_n = space_per_neuron / 2
            lista_lineas_xy_pre = []
            ne = (lasth * h) - 1
            neY = h - 1
            for j in range(0, h):
                ax.add_patch(plt.Circle(((medio_intervalo + x0), (medio_intervalo_n + y0)), radio, color='r'))
                
                neX = lasth - 1

                for xy in lista_lineas_xy:
                    if True: #j == 2:
                        plt.plot([xy[0],(medio_intervalo + x0)],[xy[1], (medio_intervalo_n + y0)])
                        #print(capa, ne, self.hiddenWeights[capa-1][ne])

                        my = ((medio_intervalo_n + y0) - xy[1])
                        mx = ((medio_intervalo + x0) - xy[0])
                        pendiente = my / mx
                        ordenada_origen = xy[1] - pendiente * xy[0]
                        margen_ord = 0.015
                        if pendiente < 0:
                            margen_ord = -0.045 # para compensar la rotacion del texto                        
                        ordenada_origen = ordenada_origen + margen_ord # para evitar que el texto salga encima de la linea no sobre ella
                        
                        # aleatorio entre las x del segmento de la recta (menos un margen para que no salga demasiado cerca de la neurona)
                        mx2 = random.uniform(xy[0] + 0.04, (medio_intervalo + x0) - 0.04)
                        my2 = pendiente*mx2 + ordenada_origen

                        alfa = math.degrees(math.atan(pendiente))
                        
                        if showWeights:
                            #print(h, capa-1, neX, neY)
                            text(mx2, my2, round(self.hiddenWeights[capa-1][neX][neY],3), rotation = alfa, fontsize = textSize)
                            
                    ne = ne - 1
                    neX = neX - 1

                lista_lineas_xy_pre.append([(medio_intervalo + x0), (medio_intervalo_n + y0)])
                
                neY = neY - 1

                y0 = y0 + space_per_neuron
                y1 = y1 + space_per_neuron
                
                lasth = h
                #print('\n')

            x0 = x0 + space_per_layer
            x1 = x1 + space_per_layer

            #print('-------------\n')

            lista_lineas_xy = lista_lineas_xy_pre

        plt.show()
        
    def importModel(self, path='', filename='MLP_Abel_model'):
        
        self.hiddenWeights = np.load(path + filename + '_weights.npy', allow_pickle=True)
            
        mConfig = np.load(path + filename + '_config.npy', allow_pickle=True)
            
        self.n_layer0 = int(mConfig[0])
        self.showLogs = bool(mConfig[1])
        self.lastLayerNeurons = int(mConfig[2])
        self.numEpochs = int(mConfig[3])
        self.learningRate = float(mConfig[4])
        self.debugMode = int(mConfig[5])
        self.softmax = bool(mConfig[6])
        self.activationFunction = str(mConfig[7])
        self.verbose = bool(mConfig[8])
        self.use = str(mConfig[9])
        self.batch_size = int(mConfig[10])
        self.batch_gradient = str(mConfig[11])
        self.batch_mult = int(mConfig[12])
        self.dropout = float(mConfig[13])
        self.pre_norm = bool(mConfig[14])
        self.shuffle = bool(mConfig[15])
        self.iterationDrop = float(mConfig[16])
        self.version_importedModel = mConfig[17]
        self.hiddenL2 = mConfig[18]
        self.hiddenL = mConfig[19]
            
            
        if self.debugMode > 0:
            
            self.meanCostByEpoch = np.load(path + filename + '_meanCostByEpoch.npy', allow_pickle=True).tolist()
            
        if self.debugMode > 1:

            self.debugWeights = np.load(path + filename + '_debugWeights.npy', allow_pickle=True).tolist()
    
    def exportModel(self, path='', filename='MLP_Abel_model'):
        
        np.save(path + filename + '_weights.npy', np.asarray(self.hiddenWeights, dtype=object))
            
            
        mConfig = []
        mConfig.append(self.n_layer0)
        mConfig.append(self.showLogs)
        mConfig.append(self.lastLayerNeurons)
        mConfig.append(self.numEpochs)
        mConfig.append(self.learningRate)
        mConfig.append(self.debugMode)
        mConfig.append(self.softmax)
        mConfig.append(self.activationFunction)
        mConfig.append(self.verbose)
        mConfig.append(self.use)
        mConfig.append(self.batch_size)
        mConfig.append(self.batch_gradient)
        mConfig.append(self.batch_mult)
        mConfig.append(self.dropout)
        mConfig.append(self.pre_norm)
        mConfig.append(self.shuffle)
        mConfig.append(self.iterationDrop)
        mConfig.append(self.version)
        mConfig.append(self.hiddenL2)
        mConfig.append(self.hiddenL)

        mConfig = np.asarray(mConfig, dtype=object)
        
        np.save(path + filename + '_config.npy', mConfig)
            
            
        if self.debugMode > 0:
            
            np.save(path + filename + '_meanCostByEpoch.npy', self.meanCostByEpoch)
            
        if self.debugMode > 1:

            np.save(path + filename + '_debugWeights.npy', np.asarray(self.debugWeights, dtype=object))

        
    def log(self, *m):
        if self.showLogs:
            print(*m)
            
    def printVerbose(self, *m):
        if self.verbose:
            print(*m)
        
    def initializeWeight(self, n, i, lastN):
        if len(self.manualWeights) == 0:
            numW = n * lastN
            
            if self.rangeRandomWeight is None:
                
                if self.activationFunction == 'sigmoid':

                    if self.XavierInitialization == 'normalized': # Normalized Xavier initialization
                        
                        highVal = math.sqrt(6.0) / math.sqrt(lastN + n) 
                        lowVal = -1 * highVal

                        mnar = np.random.uniform(low=lowVal, high=highVal, size=(numW,1))
                    
                    else: # Xavier initialization
                        
                        mnar = np.random.randn(numW, 1) * math.sqrt(1.0 / lastN)
                    
                else:
                    
                    mnar = np.random.randn(numW, 1) * math.sqrt(2.0 / lastN) # He initialization
                
            else:
                
                highVal = self.rangeRandomWeight[1]
                lowVal = self.rangeRandomWeight[0]
            
                mnar = np.random.uniform(low=lowVal, high=highVal, size=(numW,1))
            
        else:
            mnar = np.asarray(self.manualWeights[i])
            #mnar = mnar.reshape(mnar.shape[0], 1)
        
        #ns = int(mnar.shape[0] / lastN)
        
        mnar = mnar.reshape(lastN, n, order='F')
        
        return mnar
    
    def ActivationFunction(self, x, activ_type='sigmoid'):
        if activ_type=='sigmoid':
            return 1.0/(1 + np.exp(-1*x))
        elif activ_type=='relu':
            return np.where(x > 0, x, 0)
        elif activ_type=='softplus':
            return np.log(1 + np.exp(x))
        elif activ_type=='leakyrelu':
            return np.where(x > 0, x, 0.01 * x)
        elif activ_type=='identity':
            return np.copy(x)
        else:
            x[x>0.5] = 1
            x[x<=0.5] = 0
            return x
    
    def functionDerivative(self, x, activ_type='sigmoid'):
        if activ_type=='sigmoid':
            return self.ActivationFunction(x,activ_type) * (1-self.ActivationFunction(x,activ_type))
        elif activ_type=='relu':
            return np.where(x >= 0, 1, 0)
        elif activ_type=='softplus':
            return 1.0/(1 + np.exp(-1*x))
        elif activ_type=='leakyrelu':
            return np.where(x >= 0, 1, 0.01)
        elif activ_type=='identity':
            return 1
        else:
            return 1
    
    def cost(self, y_true, y_pred):
        c = y_true - y_pred
        return c
    
    def softmaxF(self, x):
        if np.max(np.abs(x)) < 500: # prevent overflow
            expX = np.exp(x)
            return expX / np.sum(expX, axis=-1).reshape(-1, 1)
        else:
            return x / np.maximum(1, np.sum(x, axis=-1).reshape(-1, 1))
        
        
    def pre_norm_forward_FC(self, v_layer):

        if self.batch_size == 1 or len(v_layer.shape) == 1:
            v_layer_norm = (v_layer - v_layer.mean()) / (v_layer.std() + 1e-7)
        else:
            v_layer_norm = ((v_layer.T - np.mean(v_layer, axis=1)) / (np.std(v_layer, axis=1) + 1e-7)).T
        
        return v_layer_norm
        
        
        
    def fit(self, x, y):
        n_layer0 = x.shape[1]
        self.n_layer0 = n_layer0
        
        self.hiddenL = copy.deepcopy(self.hiddenL2)
                
        hiddenW = [None] * (len(self.hiddenL) + 1)
        
        self.lastLayerNeurons = y.shape[1]
        
        self.hiddenL.append(y.shape[1])
        
        self.printVerbose('Training started with', x.shape[0], 'samples')
        
        if self.batch_size == 1:
            numIterations = x.shape[0]
        else:
            numIterations = math.ceil(x.shape[0] / self.batch_size)
            
        numIterations = int(numIterations * (1 - self.iterationDrop))
        
        
        for epochs in range(0, self.numEpochs):
            
            meanCostByEpochE = 0
            
            batch_pos = 0
            
            xy_ind = np.arange(x.shape[0])
            
            if self.shuffle:
                np.random.shuffle(xy_ind)
                
            classMaxError = [0, 0]
                        
            for x_pos in range(0, numIterations):
                
                if self.batch_size == 1:
                    c_positions = xy_ind[x_pos]
                else:
                    if (batch_pos + self.batch_size) < xy_ind.shape[0]:
                        c_positions = xy_ind[batch_pos:batch_pos+self.batch_size]
                    else:
                        c_positions = xy_ind[batch_pos:]

                x_val = x[c_positions]
                    
            
                v_layer = x_val #np.asarray(x_val)
                #v_layer = v_layer.reshape(1, v_layer.shape[0])
                lastN = n_layer0

                layerValues = []
                preActivateValues = []
                
                f_vlayer = self.ActivationFunction(v_layer, 'identity')
                layerValues.append(f_vlayer)
                preActivateValues.append(v_layer)
                
                f_vlayer = v_layer
                
                dropout_values = []

                for i, hiddenLayer in enumerate(self.hiddenL):
                    entries = hiddenLayer * lastN

                    if hiddenW[i] is None:
                        hiddenW[i] = self.initializeWeight(hiddenLayer, i, lastN) # Initialize weights

                    valuesForPerc = int(entries / hiddenLayer)

                    firstPos = 0
                    lastPos = valuesForPerc
                    
                    self.log('x_j: ', f_vlayer)
                    self.log('w_j: ', hiddenW[i])
                    
                    
                    v_layer = f_vlayer.dot(hiddenW[i])
                    
                    if self.pre_norm and (i < (len(self.hiddenL) - 1)):
                        v_layer = self.pre_norm_forward_FC(v_layer)
                        
                    
                    if self.dropout != 0 and (i < (len(self.hiddenL) - 1)):
                        dropout_v = np.random.binomial(1, 1-self.dropout, size=hiddenLayer) / (1-self.dropout)
                        v_layer = v_layer * dropout_v
                        
                        dropout_values.append(dropout_v)
                        
                    #print('v_layer:, ', v_layer, '\n')
                    #print(f_vlayer.shape, hiddenW[i].shape, v_layer.shape)

                    #print("\n\n\n")

                    #print('nuevosValores: ', nuevosValores, '\n', 'v_layer: ', v_layer)
                    
                    #v_layer = v_layer.reshape(1, v_layer.shape[0])
                    
                    self.log('net_j:', v_layer, '\n')
                    
                    if (i == (len(self.hiddenL) - 1)):
                        if(self.softmax):
                            f_vlayer = self.softmaxF(v_layer).reshape(-1)
                        else:
                            if self.use == 'classification':
                                f_vlayer = self.ActivationFunction(v_layer, 'sigmoid') # use sigmoid on last layer if classification
                            else:
                                f_vlayer = self.ActivationFunction(v_layer, 'identity') # use identity on last layer if regression
                    else:
                        f_vlayer = self.ActivationFunction(v_layer, self.activationFunction)#.reshape(-1)
                    
                    layerValues.append(f_vlayer)
                    preActivateValues.append(v_layer)
                    v_layer = f_vlayer
                    
                    self.log('f(net_j):', f_vlayer, '\n')
                    
                    #print("\n\n\n")
                    
                    #print('\n\nNuevos pesos: ', hiddenW)
                    
                    lastN = hiddenLayer
                    
                coste_anterior = None

                i = len(self.hiddenL) - 1
                
                #print('max i: ', i)
                
                """
                if(self.softmax):
                    f_vlayer = self.softmaxF(f_vlayer).reshape(-1)
                    self.log('f_vlayer (Softmax output):', f_vlayer)
                """
                
                self.log('-----------------\nBackPropagation: \n')
                
                # backpropagation:
                for hiddenLayer in ([n_layer0] + self.hiddenL)[::-1]:
                    self.log('Neurons in this layer: ', hiddenLayer)
                    #print('i: ', i, '\n')

                    if coste_anterior is None:
                        if(self.softmax):
                            derivf_coste = self.functionDerivative(v_layer, self.activationFunction)
                        else:
                            if self.use == 'classification':
                                derivf_coste = self.functionDerivative(v_layer, 'sigmoid')
                            else:
                                derivf_coste = self.functionDerivative(v_layer, 'identity')
                            
                        f_cost = self.cost(y[c_positions], f_vlayer)
                        #if self.batch_size != 1:
                            #f_cost = f_cost / v_layer.shape[0]
                            
                        coste = f_cost * derivf_coste
                        
                        if self.batch_size != 1:
                            batch_pos = batch_pos + self.batch_size
                                
                        #print(y[x_pos].shape, f_vlayer.shape, coste.shape)
                        #coste = coste.reshape(-1)
                        #coste = coste.reshape(coste.shape[0], 1)
                        
                        #if self.batch_size != 1:
                            #coste = np.sum(coste, axis=0)
                            #derivf_coste = np.sum(derivf_coste, axis=0)
                            
                            
                             
                        if self.debugMode > 0:
                            meanCostByEpochE = meanCostByEpochE + (abs(coste) if self.batch_size == 1 else np.mean(np.absolute(coste), axis=0))
                            
                            """
                            if self.fM is not None:
                                
                                mclass = np.argmax(meanCostByEpochE)

                                if mclass == classMaxError[0]:
                                    classMaxError[1] = classMaxError[1] + 1
                                else:
                                    classMaxError = [mclass, 1]


                                if classMaxError[1] > self.fM:
                                    drop = np.ones(self.lastLayerNeurons)
                                    drop[classMaxError[0]] = 1.05

                                    coste = coste * drop

                                    classMaxError[1] = 0
                            """
                                                            
                            
                        if self.debugMode > 2:
                            self.costs.append(coste)                            
                        
                        self.log('derivf_coste: ', derivf_coste, 'cost: ', coste, '\n')

                    else:

                        entries = hiddenLayer * nextN
                        valuesForPerc = int(entries / hiddenLayer)
                        firstPos = 0
                        lastPos = valuesForPerc

                        #coste = []
                        #coste = np.zeros(shape=(hiddenLayer))
                        
                        self.log('prev_error: ', coste_anterior)
                            
                        pesos_salientes = hiddenW[i+1].T
                            
                        #print('hiddenW[i+1][j::hiddenLayer]: ', pesos_salientes)
                            
                        preActivateValueM = preActivateValues[i+1]
                        
                            
                        preDeriv = self.functionDerivative(preActivateValueM, self.activationFunction)
                        self.log('preDeriv: ', preDeriv)
                            
                        costeA = coste_anterior.dot(pesos_salientes) # coste por los pesos que salen de la neurona
                        #costeA = np.asarray(costeA)
                        self.log("preCostA: ", costeA)
                        costeA = costeA * (preDeriv)
                        
                        #costeA = costeA.reshape(-1)
                        #costeA = costeA.T

                        
                        if self.dropout != 0 and i > -1: # dropout is not done on input layer
                            costeA = costeA * dropout_values[i]
                            
                        self.log('costA: ', costeA)
                            
                            
                        layerValueM = layerValues[i+1]
                          
                        #print("coste_anterior: ", coste_anterior)
                        self.log("layer values: ", layerValueM)
                        

                        
                        
                        if self.batch_gradient == 'sum':
                            
                            preT1 = coste_anterior.reshape((1 if self.batch_size==1 else coste_anterior.shape[0]), (coste_anterior.shape[0] if self.batch_size==1 else coste_anterior.shape[1]))
                            preT2 = layerValueM.reshape((layerValueM.shape[0] if self.batch_size==1 else layerValueM.shape[1]), (1 if self.batch_size==1 else layerValueM.shape[0]))
                        
                        elif self.batch_size == 1:
                            
                            preT1 = coste_anterior.reshape(1, coste_anterior.shape[0])
                            preT2 = layerValueM.reshape(layerValueM.shape[0], 1)
                        
                        else:
                            
                            preT1 = np.mean(coste_anterior, axis=0)
                            preT1 = preT1.reshape(1, preT1.shape[0])
                            
                            preT2 = np.mean(layerValueM, axis=0)
                            preT2 = preT2.reshape(preT2.shape[0], 1)
                            
                                                
                        pre = preT2.dot(preT1)
                        
                        #if self.batch_size != 1:
                            #pre = pre * (1.0 / layerValueM.shape[0])
                          
                        
                        pre = pre * self.learningRate
                        
                        #print(coste_anterior.shape, layerValueM.shape, preT2.shape, preT1.shape, pre.shape)
                        
                        self.log('pre: ', pre, '\n')
                        self.log('Old weight: ', hiddenW[i+1])
                        hiddenW[i+1] = (hiddenW[i+1] + pre)
                        self.log('New weight: ', hiddenW[i+1], '\n\n')
                            
                        coste = costeA
                                                        
                        self.log('\n\n')

                    #coste = coste.reshape(-1)
                    #print(coste.shape)
                    
                    #if len(coste.shape) == 3:
                        #coste = coste.reshape(coste.shape[0] * coste.shape[1], coste.shape[2])
                        
                    #print('Coste: ' , coste, coste.shape)
                    #print("\n\n")
                    coste_anterior = coste
                    nextN = hiddenLayer
                    i = i - 1
                    
                    #print('------------------')
                    
                    #print('\n\nNuevos pesos: ', hiddenW)
            
            self.printVerbose('\nEpoch', str(epochs+1) + '/' + str(self.numEpochs), 'completed')
            
            if self.debugMode > 0:
                self.meanCostByEpoch.append(meanCostByEpochE / numIterations)
                
                self.printVerbose('--- Epoch loss:', round(np.mean(self.meanCostByEpoch[-1]),4))
                
            if self.debugMode > 1:
                self.debugWeights.append(copy.deepcopy(hiddenW))
                
                
            self.batch_size = int(self.batch_size * self.batch_mult)
            
                            
        self.hiddenWeights = hiddenW
        #print('\n\nNuevos pesos: ', hiddenW)
        
        self.printVerbose('\n\nTraining finished\n\n')
        
        return self
                
    def predict(self, x, noProba=1):
            layerValues = np.zeros(shape=(x.shape[0],self.lastLayerNeurons))
            #preActivateValues = np.zeros(shape=(x.shape[0],self.lastLayerNeurons))
            n_layer0 = x.shape[1]
        
            for x_pos, x_val in enumerate(x):
            
                v_layer = x_val #np.asarray(x_val)
                #v_layer = v_layer.reshape(1, v_layer.shape[0])
                lastN = n_layer0
                
                f_vlayer = self.ActivationFunction(v_layer, 'identity')
                                
                #f_vlayer = v_layer

                for i, hiddenLayer in enumerate(self.hiddenL):
                    entries = hiddenLayer * lastN

                    valuesForPerc = int(entries / hiddenLayer)

                    firstPos = 0
                    lastPos = valuesForPerc

                    #print('ns: ', ns)
                    #print('f_vlayer: ', f_vlayer)
                    #print('w: ', self.hiddenWeights[i].reshape(lastN, ns, order='F'))
                    v_layer = f_vlayer.dot(self.hiddenWeights[i])            
                    #print('v_layer:, ', v_layer, '\n')
                    
                    if self.pre_norm and (i < (len(self.hiddenL) - 1)):
                        v_layer = self.pre_norm_forward_FC(v_layer)
                    
                    #v_layer = v_layer.reshape(1, v_layer.shape[0]) 
                    if (i == (len(self.hiddenL) - 1)):
                        if(self.softmax):
                            f_vlayer = self.softmaxF(v_layer).reshape(-1)
                        else:
                            if self.use == 'classification':
                                f_vlayer = self.ActivationFunction(v_layer, 'sigmoid') # use sigmoid on last layer if classification
                            else:
                                f_vlayer = self.ActivationFunction(v_layer, 'identity') # use identity on last layer if regression
                    else:
                        f_vlayer = self.ActivationFunction(v_layer, self.activationFunction)#.reshape(-1)
                    
                    #print('f_vlayer:, ', f_vlayer, '\n')
                    v_layer = f_vlayer
                    
                    #print("\n\n")
                    
                    lastN = hiddenLayer
                    
                
                layerValues[x_pos] = f_vlayer
                #preActivateValues[x_pos] = v_layer
                
                #print('Salida: ', f_vlayer)
                
            """    
            if(self.softmax):
                layerValues = self.softmaxF(layerValues)
            """
                    
            
            if noProba==1:
                if self.use == 'classification':
                    return self.ActivationFunction(layerValues, 2).astype(int)
                else:
                    return layerValues
            else:
                return layerValues
        
    
    def predict_proba(self, x):
        return self.predict(x, 0)
    
    
    def plot_mean_error_last_layer(self, customLabels=[], byClass=False):

        if self.debugMode > 0:

            meancost = np.asarray(self.meanCostByEpoch)

            if len(meancost.shape) > 1 and not byClass:
                meancost = np.mean(meancost, axis=1)

            ptitle = 'Last layer mean error by epoch'   

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(range(0, meancost.shape[0]), meancost)
            ax.set(xlabel='Epoch', ylabel='Mean error', title=ptitle)
            ax.grid()

            if len(meancost.shape) > 1:
                if meancost.shape[1] > 1:
                    if len(customLabels) == 0:
                        neur = [("Neuron " + str(i)) for i in range(0, meancost.shape[1])]
                    else:
                        neur = customLabels

                    plt.legend(neur, loc="upper right")

            plt.show()

        else:
            print('MLP debug mode must be level 1 or higher')

    def plot_weights_by_epoch(self, max_weights=-1):

        if self.debugMode > 1:

            dw = self.debugWeights

            dwx = dw[0][len(dw[0]) - 1][:]

            fig, ax = plt.subplots(figsize=(8,6))

            ygrafico = {}

            for jposH, posH in enumerate(range(0, len(dw))): # for each epoch

                dwF = dw[jposH][len(dw[0]) - 1][:]
                #print(dwF.shape)
                for posg, neu in enumerate(dwF):
                    #print(neu.shape)        
                    if posg in ygrafico:
                        ygrafico[posg].append(neu[0])
                    else:
                        ygrafico[posg] = [neu[0]]

            if max_weights == -1:

                for ygrafico2 in ygrafico.values():
                    ax.plot(range(0, len(ygrafico2)), ygrafico2)
            else:

                if max_weights < 1:

                    print('max_weights must be bigger than 0')

                elif max_weights > len(ygrafico.values()):

                    print('max_weights must be lower than total weights of last layer')

                else:

                    ygrafico3 = []
                    
                    # Gets the weights that have changed the most from beginning to end.

                    for yi, ygrafico2 in enumerate(ygrafico.values()):
                        a = abs(ygrafico[yi][0] - ygrafico[yi][-1])
                        #print(ygrafico[yi][0], a)
                        ygrafico3.append([ygrafico2, a])

                    for ygrafico4 in sorted(ygrafico3, key=lambda tupval: -1*tupval[1])[0:max_weights]:
                        #print(ygrafico4)
                        plt.plot(range(0, len(ygrafico4[0])), ygrafico4[0])

            ax.set(xlabel='Epoch', ylabel='Weight', title='Last layer weights by epoch')
            ax.grid()

            plt.show()

        else:
            print('MLP debug mode must be level 2 or higher')