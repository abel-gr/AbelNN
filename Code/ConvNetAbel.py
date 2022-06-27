# Copyright Abel Garcia. All Rights Reserved.
# https://github.com/abel-gr/AbelNN

import numpy as np
import copy as copy
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


class ConvNetAbel:

    __version__ = '1.2.1'
    
    def __init__(self, hidden = [1], nEpochs = 1, learningRate=0.1, manualWeights=[],
                 debugLevel=1, rangeRandomWeight=None, showLogs=False, softmax=False,
                 activationFunction='leakyrelu', verbose = False, use='classification',
                 batch_size=1, batch_gradient='average', batch_mult=1, dropout=0, pre_norm=False,
                 shuffle=True, iterationDrop=0, convFilters = [32, 64, 128], convStride=2,
                 convFilterSizes=3, learningRateConv=0.001, convEpochs=10, kernel_initializer='he_normal'):
        
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
        
        
        # ConvNet:
        self.convFilters = convFilters
        self.filtersValues = [None] * len(convFilters)
        self.convStride = convStride
        self.convFilterSizes = convFilterSizes
        self.learningRateConv = learningRateConv
        self.convEpochs = convEpochs
        self.kernel_initializer = kernel_initializer
        
    # Conv2 with only one kernel
    def conv2(self, x, kernel, stride=1):
        output = [] #np.zeros((kernel.shape), dtype=np.float32)
        kernel_l = kernel.shape[0]
        kernel_size = kernel.shape[0] * kernel.shape[1]
        
        c = int(kernel_l / 2)
        
        for i in range(c, x.shape[0] - c, stride):
            
            o = []
            
            for j in range(c, x.shape[1] - c, stride):
                
                i0 = i - c
                j0 = j - c
                
                i1 = i + c + 1
                j1 = j + c + 1
                
                o.append(np.sum(x[i0:i1, j0:j1] * kernel))
                
            output.append(o)
            
        output = np.asarray(output)
                
        return output
    
    # Convolution with multi-filters
    def conv_filters(self, x, filters, stride=1, relu=False, mode='same'):
        
        lex = len(x.shape)
        lef = len(filters.shape)
        
        if lex > lef:
            print('conv_filters: The input array cannot have more dimensions than the filter array.')
            return 0
        
        output = []
        kernel_l = filters.shape[0]
        kernel_size = filters.shape[0] * filters.shape[1]
        
        if lef == 2:
            num_filters = 1
        else:
            num_filters = filters.shape[-1]
        
        c = int(kernel_l / 2)
                
        dim3 = False
        
        evenShapeKernel = (kernel_l % 2 == 0)
        
        if lex == 2:            
            dim2 = True
            p0 = x.shape[0]
            p1 = x.shape[1]
        else:
            
            # x parameter was the output of this method previously called
            if lex == lef:
                num_new_filters = int(num_filters / x.shape[-1])
            
                if (num_new_filters % 2 != 0) and (num_filters % 2 == 0):
                    num_new_filters = num_new_filters - 1
                    
                if (num_new_filters == 0):
                    num_new_filters = 1
            
            else: # It is the first convolutional layer of a color image
                num_new_filters = num_filters
                dim3 = True
            
            dim2 = False
            p0 = x.shape[0]
            p1 = x.shape[1]
        
        if mode == 'full':
            fs0 = int(filters.shape[0] / 2)
            fs1 = int(filters.shape[1] / 2)
            
            max0 = p0 + fs0
            max1 = p1 + fs1
            
            ini0 = -1 * fs0
            ini1 = -1 * fs1
            
        elif mode == 'same':
            max0 = p0
            max1 = p1
            
            ini0 = 0
            ini1 = 0
            
        elif mode == 'valid':
            fs0 = int(filters.shape[0] / 2)
            fs1 = int(filters.shape[1] / 2)
            
            max0 = p0 - fs0
            max1 = p1 - fs1
            
            ini0 = fs0
            ini1 = fs1
            
        else:
            print('Mode must be same, valid or full')
            return 0
        
        if evenShapeKernel and mode == 'valid':
            max0 = max0 + 1
            max1 = max1 + 1
        
        for i in range(ini0, max0, stride):
            
            o = []
            
            for j in range(ini1, max1, stride):
                
                i0 = i - c
                j0 = j - c
                
                i1 = i + c + 1
                j1 = j + c + 1
                                
                if evenShapeKernel:
                    i0 = i0 + 1
                    j0 = j0 + 1
      
                zero_padding_top = 0
                zero_padding_bottom = 0
                zero_padding_left = 0
                zero_padding_right = 0
                
                if i0 < 0:
                    zero_padding_top = abs(i0)
                    i0 = 0
                    
                if j0 < 0:
                    zero_padding_left = abs(j0)
                    j0 = 0
                    
                if i1 > p0:
                    zero_padding_bottom = i1 - p0
                    i1 = p0
                    
                if j1 > p1:
                    zero_padding_right = j1 - p1
                    j1 = p1
                    
                if dim2:
                    m = x[i0:i1, j0:j1]
                    
                    #print('mshape:', m.shape, kernel_size, zero_padding_top, zero_padding_left)
                    
                    # Zero padding:
                    m = np.pad(m, ((zero_padding_top,zero_padding_bottom),(zero_padding_left,zero_padding_right)), 'constant')
                    
                    if lef != 2:
                        m = np.expand_dims(m, axis=-1)
                        m = np.repeat(m, num_filters, axis=-1)
                    
                else:
                    xi = x[i0:i1, j0:j1, :]
                    
                    # Zero padding:
                    xi = np.pad(xi, ((zero_padding_top,zero_padding_bottom),(zero_padding_left,zero_padding_right),(0,0)), 'constant')
                    
                    if dim3:
                        xi = np.expand_dims(xi, axis=-1)
                    
                    m = np.repeat(xi, num_new_filters, axis=-1)
                    
                
                #print('M,F\n', m[:,:,0], filters[:,:,0])
                #print(m.shape, filters.shape)
                m = m * filters
                
                #print('m*f\n', m[:,:,0])
                
                m = np.sum(m, axis=0)
                m = np.sum(m, axis=0)
                
                if dim3:
                    m = np.sum(m, axis=0)                    
                
                o.append(m)
                
            output.append(o)
            
        output = np.asarray(output)
        
        if relu:
            output[output < 0] = 0
                
        return output
    
    def kernelInitializer(self, i, ksize, inSize, outSize):
        
        if 'xavier' in self.kernel_initializer:
            
            if self.kernel_initializer == 'xavier_normal':
            
                if len(ksize) == 4:
                    self.filtersValues[i] = np.random.randn(ksize[0],ksize[1],ksize[2],ksize[3]) * math.sqrt(2.0 / (inSize + outSize))
                else:
                    self.filtersValues[i] = np.random.randn(ksize[0],ksize[1],ksize[2]) * math.sqrt(2.0 / (inSize + outSize))
                
            elif self.kernel_initializer == 'xavier_uniform':
                
                highVal = math.sqrt(6.0 / (inSize + outSize))
                lowVal = -1 * highVal

                self.filtersValues[i] = np.random.uniform(low=lowVal, high=highVal, size=ksize)
                
        else:
            
            if self.kernel_initializer == 'he_normal':
                
                if len(ksize) == 4:
                    self.filtersValues[i] = np.random.randn(ksize[0],ksize[1],ksize[2],ksize[3]) * math.sqrt(2.0 / inSize)
                else:
                    self.filtersValues[i] = np.random.randn(ksize[0],ksize[1],ksize[2]) * math.sqrt(2.0 / inSize)
                
            elif self.kernel_initializer == 'he_uniform':
                
                highVal = math.sqrt(6.0 / inSize)
                lowVal = -1 * highVal
                
                self.filtersValues[i] = np.random.uniform(low=lowVal, high=highVal, size=ksize)
    
    def convLayersFeedForward(self, im):
        
        self.convInputs = []
        
        len_m = len(im.shape)
        #print('len_m:', len_m)
        
        for i, cl in enumerate(self.convFilters):
            
            self.convInputs.append(im)
            
            if (self.filtersValues[i] is None):
                
                if (type(self.convFilterSizes) == list):
                    ks = self.convFilterSizes[i]
                else:
                    ks = self.convFilterSizes
                
                
                inSize = np.prod(im.shape)
                
                if 'xavier' in self.kernel_initializer:
                    if self.batch_size == 1:
                        imshape = np.asarray([im.shape[0], im.shape[1]])
                    else:
                        imshape = np.asarray([im.shape[1], im.shape[2]])
                        
                    extraShape = int((ks % 2) == 0)
                    ks2 = int(ks / 2) * 2
                    outSize = np.prod((imshape - ks2 + extraShape)) * cl
                else:
                    outSize = 0

                
                if i == 0 and len_m == 3:
                    
                    if self.batch_size == 1:
                        self.kernelInitializer(i, (ks,ks,im.shape[2],cl), inSize, outSize)
                    else:
                        self.kernelInitializer(i, (ks,ks,cl), inSize, outSize)
                    
                else:
                    
                    self.kernelInitializer(i, (ks,ks,cl), inSize, outSize)
                                        
            
            k_filters = self.filtersValues[i]
            
            if (type(self.convStride) == list):
                stride_par = self.convStride[i]
            else:
                stride_par = self.convStride
            
            #print('Convolutional layer', i, '\n')
            #print('Layer input shape:', im.shape)
            #print('Layer filters array shape:', k_filters.shape)
            
            
            # Start of convolutions
            
            #im = self.conv_filters(im, k_filters, relu=True, stride=stride_par, mode='valid')
            
            filtersValues_shape01 = np.asarray([k_filters.shape[0], k_filters.shape[1]])
            filtersValues_shape_d2 = (filtersValues_shape01 / 2).astype(int)

            extraShape = (filtersValues_shape01 % 2) == 0
            eS0 = extraShape[0].astype(int)
            eS1 = extraShape[1].astype(int)
            posYf = eS0
            posXf = eS1

            filter_shape0 = k_filters.shape[0]
            filter_shape1 = k_filters.shape[1]

            if (len(k_filters.shape) >= 3):
                num_filters = k_filters.shape[-1]
            else:
                num_filters = 1

            if self.batch_size == 1:
                xshape = np.asarray([im.shape[0], im.shape[1]])
            else:
                xshape = np.asarray([im.shape[1], im.shape[2]])
                
            output_shape = xshape - filtersValues_shape_d2*2 + eS0


            if ((len(im.shape) < len(k_filters.shape)) or (len(im.shape) == 2 and num_filters == 1)):
                Xr = np.expand_dims(im, axis=-1)
                Xr = np.repeat(Xr, num_filters, axis=-1)

            else:

                if (len(im.shape) == len(k_filters.shape)):
                    
                    if self.batch_size == 1:
                        
                        new_filters = int(im.shape[-1] / num_filters)
                        Xr = np.repeat(im, new_filters, axis=-1)
                        
                    else:
                        
                        Xr = np.expand_dims(im, axis=-1)
                        Xr = np.repeat(Xr, num_filters, axis=-1)
                        

                else:
                    Xr = im

            if (len(Xr.shape) == 2):
                npad = ((0,eS0), (0,eS1))
                out_s = [output_shape[0], output_shape[1], 1]
            elif (len(Xr.shape) == 3):
                npad = ((0,eS0), (0,eS1), (0,0))
                out_s = [output_shape[0], output_shape[1], num_filters]
            elif (len(Xr.shape) == 4):
                
                if self.batch_size == 1:
                    npad = ((0,eS0), (0,eS1), (0,0), (0,0))
                    out_s = [output_shape[0], output_shape[1], im.shape[2], num_filters]
                else:
                    npad = ((0,0), (0,eS0), (0,eS1), (0,0))
                    out_s = [im.shape[0], output_shape[0], output_shape[1], num_filters]

            X_pad = np.pad(Xr, npad, 'constant')

            out_s[0 if self.batch_size == 1 else 1] = int(np.ceil(out_s[0 if self.batch_size == 1 else 1] / stride_par))
            out_s[1 if self.batch_size == 1 else 2] = int(np.ceil(out_s[1 if self.batch_size == 1 else 2] / stride_par))
            conv_output = np.zeros(out_s)

            
            
            if self.batch_size != 1:
                k_filters = np.expand_dims(k_filters, axis=0)
                k_filters = np.repeat(k_filters, im.shape[0], axis=0)
                
            #print(Xr.shape, X_pad.shape, k_filters.shape, conv_output.shape, output_shape)

            for posY in range(0, filter_shape0):
                for posX in range(0, filter_shape1):       

                    # valid convolution
                    if self.batch_size == 1:
                        conv_output += X_pad[posYf:posYf+output_shape[0]:stride_par, posXf:posXf+output_shape[1]:stride_par] * k_filters[posY, posX]
                    else:
                        conv_output += X_pad[:, posYf:posYf+output_shape[0]:stride_par, posXf:posXf+output_shape[1]:stride_par] * k_filters[:, posY, posX].reshape(k_filters.shape[0],1,1,k_filters.shape[3])

                    posXf = posXf + 1

                posYf = posYf + 1
                posXf = eS1
            
            # End of convolutions
            
            if self.pre_norm:
                
                ax_f = tuple(range(0,len(conv_output.shape)))
                
                if self.batch_size == 1:
                    ax_f = ax_f[0:-1]
                    conv_output = (conv_output - np.mean(conv_output, axis=ax_f)) / (np.std(conv_output, axis=ax_f) + 1e-7)
                else:
                    ax_f = ax_f[1:-1]
                    conv_output = (conv_output - np.mean(conv_output, axis=ax_f).reshape(conv_output.shape[0],1,1,conv_output.shape[3])) / (np.std(conv_output, axis=ax_f).reshape(conv_output.shape[0],1,1,conv_output.shape[3]) + 1e-7)
                               
                #conv_output = (conv_output - conv_output.mean()) / (conv_output.std() + 1e-7)
                
            im = self.ActivationFunction(conv_output, 'relu')
            
            #print('Layer output shape:', im.shape, '\n---------------------\n')            
            
            
        return im
     
        
    def convLayersBackpropagation(self, last_layer_output, prev_cost):
        
        i = len(self.filtersValues) - 1
                                
        last_shape = list(last_layer_output.shape)
        
        if self.batch_size != 1:
            batch_el = last_shape[0]
            last_shape = last_shape[1:] + [batch_el]
                
        error_by_x = np.reshape(prev_cost, last_shape)
        
        """
        if self.batch_size == 1:
            num_filters = last_layer_output.shape[2]
        else:
            num_filters = last_layer_output.shape[3]
        """
        
        self.log('Start of convLayersBackpropagation:', '\n')
        
        #self.log('prev_cost:', prev_cost.shape, prev_cost, '\n')
        #self.log('last_layer_output:', last_layer_output.shape, last_layer_output, '\n')
        #self.log('error_by_x:', error_by_x.shape, error_by_x, '\n')
        
        #if self.batch_size != 1:
            #error_by_x = np.mean(error_by_x, axis=0)
            
        
        for k_filters in self.filtersValues[::-1]:
            
            X = self.convInputs[i]
            
            if self.batch_size != 1:
                X_batchshape = list(X.shape)
                X_batch_elements = X_batchshape[0]
                X_batchshape = X_batchshape[1:] + [X_batch_elements]
                
                X = np.reshape(X, X_batchshape)
                                
                #X = np.mean(X, axis=0)
                        
            
            # to dilate gradient if needed because of stride
            if (type(self.convStride) == list):
                stride_par = self.convStride[i]
            else:
                stride_par = self.convStride
            
            if stride_par != 1:
                
                #erShape = error_by_x.shape[0] * stride_par
                erShape = (X.shape[0])
                
                if self.batch_size == 1:
                    error_by_output = np.zeros((erShape, erShape, self.convFilters[i]), dtype=float)
                else:
                    error_by_output = np.zeros((erShape, erShape, self.convFilters[i], batch_el), dtype=float)

                #print(error_by_output.shape, error_by_x.shape)
                
                posI = 0
                posJ = 0
                erx1 = (error_by_x.shape[0])
                erx2 = (error_by_x.shape[1])
                # Zero-interweave:
                for pe_i in range(0, erx1):
                    for pe_j in range(0, erx2):
                        
                        error_by_output[posI, posJ] = error_by_x[pe_i, pe_j]
                        
                        if (posJ + 2) < erShape:
                            posJ = posJ + 2
                        else:
                            posJ = posJ + 1
                    
                    if (posI + 2) < erShape:
                        posI = posI + 2
                    else:
                        posI = posI + 1
                        
                    posJ = 0
            
            else:
                
                # dE/dO
                error_by_output = error_by_x
        
                       
            f_rotated = np.flip(self.filtersValues[i], 0)
            f_rotated = np.flip(f_rotated, 1)
                        
            # dE/dF
            #error_by_filter = self.conv_filters(X, error_by_output, relu=False, stride=1, mode='valid')
            # dE/dX
            #error_by_x = self.conv_filters(f_rotated, error_by_output, relu=False, stride=1, mode='full')
            
                
            # Start of convolutions
            
            err_output_shape01 = np.asarray([error_by_output.shape[0], error_by_output.shape[1]])
            err_out_shape_d2 = (err_output_shape01 / 2).astype(int)

            
            xshape = np.asarray([X.shape[0], X.shape[1]])
            fshape = np.asarray([f_rotated.shape[0], f_rotated.shape[1]])

            extraShape = (err_output_shape01 % 2) == 0
            eS0 = extraShape[0].astype(int)
            eS1 = extraShape[1].astype(int)
            
            err_filt_shape = xshape - err_out_shape_d2*2 + eS0
            err_x_shape = fshape + err_out_shape_d2*2 + eS0
            
            num_filters = self.filtersValues[i].shape[-1]
            
            #print(error_by_output.shape, xshape, err_output_shape01, err_out_shape_d2*2, eS0, err_filt_shape)
            
            if self.batch_size == 1:
                error_by_filter = np.zeros((err_filt_shape[0], err_filt_shape[1], num_filters))
                error_by_x = np.zeros((err_x_shape[0], err_x_shape[1], num_filters))
            else:
                error_by_filter = np.zeros((err_filt_shape[0], err_filt_shape[1], num_filters, X_batch_elements))
                error_by_x = np.zeros((err_x_shape[0], err_x_shape[1], num_filters, X_batch_elements))
            
            
            err_out_shape0 = error_by_output.shape[0]
            err_out_shape1 = error_by_output.shape[1]

            fil_shape0 = error_by_filter.shape[0]
            fil_shape1 = error_by_filter.shape[1]

            ex_shape0 = self.filtersValues[i].shape[0]
            ex_shape1 = self.filtersValues[i].shape[1]

            posYf = eS0
            posXf = eS1
            
            if (len(X.shape) < 3):
                Xr = np.expand_dims(X, axis=-1)
                Xr = np.repeat(Xr, num_filters, axis=-1)
            else:
                Xr = X
            
            if (len(Xr.shape) == 3):
                X_pad = np.pad(Xr, ((0,eS0), (0,eS1), (0,0)), 'constant')
            elif (len(Xr.shape) == 4):
                X_pad = np.pad(Xr, ((0,eS0), (0,eS1), (0,0), (0,0)), 'constant') 
            else: # color image with batch
                X_pad = np.pad(Xr, ((0,0), (0,eS0), (0,eS1), (0,0), (0,0)), 'constant')

            
            layer_filters = self.filtersValues[i]
            
            if self.batch_size != 1:
                layer_filters = np.expand_dims(layer_filters, axis=-1)
                layer_filters = np.repeat(layer_filters, X_batch_elements, axis=-1)
            
            
            #print(X_pad.shape, error_by_output.shape, error_by_filter.shape, self.filtersValues[i].shape, error_by_output.shape, error_by_x.shape)
            for posY in range(0, err_out_shape0):
                for posX in range(0, err_out_shape1):       

                    # valid convolution (dE/dF)
                    error_by_filter += X_pad[posYf:posYf+fil_shape0, posXf:posXf+fil_shape1] * error_by_output[posY, posX]

                    # full convolution (dE/dX)
                    error_by_x[posYf:posYf+ex_shape0, posXf:posXf+ex_shape1] += layer_filters * error_by_output[posY, posX]

                    posXf = posXf + 1

                posYf = posYf + 1
                posXf = eS1

            error_by_x = np.flip(error_by_x, 0)
            error_by_x = np.flip(error_by_x, 1)
            
            
            # End of convolutions
                
                
            #print(X.shape, X_pad.shape, self.filtersValues[i].shape, error_by_filter.shape, error_by_x.shape, error_by_output.shape)
                    
            #self.log('error_by_filter:', error_by_filter[:,:,0], '\n\n')
            #self.log('prev filtersValues[i]:', self.filtersValues[i][:,:,0], '\n\n')
            #self.log('error_by_x:', error_by_x[:,:,0], '\n\n')
            
            if self.batch_size != 1:
                error_by_filter = np.mean(error_by_filter, axis=-1)
            
            #if self.pre_norm:
                #ax_f = tuple(range(0,len(error_by_filter[i].shape)))[0:-1]
                #error_by_filter = (error_by_filter - np.mean(error_by_filter, axis=ax_f)) / (np.std(error_by_filter, axis=ax_f) + 1e-7)
                
                #error_by_filter = (error_by_filter - error_by_filter.mean()) / (error_by_filter.std() + 1e-7)
            
            # Filters update
            self.filtersValues[i] = self.filtersValues[i] - self.learningRateConv * error_by_filter
            
            if self.pre_norm:
                ax_f = tuple(range(0,len(self.filtersValues[i].shape)))[0:-1]
                self.filtersValues[i] = (self.filtersValues[i] - np.mean(self.filtersValues[i], axis=ax_f)) / (np.std(self.filtersValues[i], axis=ax_f) + 1e-7)
            
            #self.log('filtersValues[i] updated:', self.filtersValues[i][:,:,0], '\n\n')            
            #self.log('\n-----------------------\n')
            
            i = i - 1
            
        self.log('End of convLayersBackpropagation')
        
    def draw(self, showWeights=False, textSize=9, customRadius=0, showLegend=True):
        fig = plt.figure(figsize=(10,8))

        ax = fig.subplots()
        
        ax.set_title("Layers and neurons of the fully-connected part of the ConvNet")
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
        
        num_FC_layers = len(self.hiddenL) + 1

        # For each layer
        for capa, h in enumerate([self.n_layer0] + self.hiddenL):
            space_per_neuron = ydim / h
            y0 = ymin
            y1 = ymin + space_per_neuron
            medio_intervalo_n = space_per_neuron / 2
            lista_lineas_xy_pre = []
            ne = (lasth * h) - 1
            neY = h - 1
            
            if showLegend:
                if capa == 0:
                    plot_label = "First FC layer"
                    neuron_color = 'r'
                elif capa + 1 == num_FC_layers:
                    plot_label = "Output layer"
                    neuron_color = 'b'
                else:
                    plot_label = "Hidden FC layer"
                    neuron_color = 'g'

                    if capa > 1:
                        plot_label = "_" + plot_label # Avoid displaying the same label in the legend for each hidden layer
            else:
                plot_label = ""
                neuron_color = 'r'
                    
            # For each neuron in this layer
            for j in range(0, h):
                plot_label = plot_label if j == 0 else ("_" + plot_label) # Avoid displaying the same label in the legend for each neuron in that layer
                
                ax.add_patch(plt.Circle(((medio_intervalo + x0), (medio_intervalo_n + y0)), radio, color=neuron_color, label=plot_label, zorder=1))
                
                neX = lasth - 1

                # For each input to this neuron
                for xy in lista_lineas_xy:
                    ax.plot([xy[0],(medio_intervalo + x0)],[xy[1], (medio_intervalo_n + y0)], zorder=0)

                    my = ((medio_intervalo_n + y0) - xy[1])
                    mx = ((medio_intervalo + x0) - xy[0])
                    pendiente = my / mx
                    ordenada_origen = xy[1] - pendiente * xy[0]
                    margen_ord = 0.015
                    if pendiente < 0:
                        margen_ord = -0.045 # compensate text rotation
                    ordenada_origen = ordenada_origen + margen_ord # add the text above the line
                        
                    # random between the x's of the line segment (minus a margin so it does not appear too close to the neuron)
                    mx2 = random.uniform(xy[0] + 0.04, (medio_intervalo + x0) - 0.04)
                    my2 = pendiente*mx2 + ordenada_origen

                    alfa = math.degrees(math.atan(pendiente))
                        
                    if showWeights:
                        ax.text(mx2, my2, round(self.hiddenWeights[capa-1][neX][neY], 3), rotation=alfa, fontsize=textSize, zorder=2)
                            
                    ne = ne - 1
                    neX = neX - 1 # Index of the neuron of the previous layer

                lista_lineas_xy_pre.append([(medio_intervalo + x0), (medio_intervalo_n + y0)])
                
                neY = neY - 1 # Index of the neuron of the current layer

                y0 = y0 + space_per_neuron
                y1 = y1 + space_per_neuron
                
            lasth = h

            x0 = x0 + space_per_layer
            x1 = x1 + space_per_layer

            lista_lineas_xy = lista_lineas_xy_pre

        if showLegend:
            plt.legend(loc='best')
            
        plt.show()
        
    def importModel(self, path='', filename='ConvNetAbel_model'):
        
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


        convConfig = np.load(path + filename + '_convConfig.npy', allow_pickle=True)

        self.convFilters = convConfig[0]
        self.convStride = convConfig[1]
        self.convFilterSizes = convConfig[2]
        self.kernel_initializer = str(convConfig[3])
        self.convEpochs = int(convConfig[4])
        self.learningRateConv = float(convConfig[5])

        self.filtersValues = np.load(path + filename + '_filtersValues.npy', allow_pickle=True)

                        
        if self.debugMode > 0:
            
            self.meanCostByEpoch = np.load(path + filename + '_meanCostByEpoch.npy', allow_pickle=True).tolist()
            
        if self.debugMode > 1:

            self.debugWeights = np.load(path + filename + '_debugWeights.npy', allow_pickle=True).tolist()
    
    def exportModel(self, path='', filename='ConvNetAbel_model'):
        
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


        convConfig = []
        convConfig.append(self.convFilters)
        convConfig.append(self.convStride)
        convConfig.append(self.convFilterSizes)
        convConfig.append(self.kernel_initializer)
        convConfig.append(self.convEpochs)
        convConfig.append(self.learningRateConv)

        convConfig = np.asarray(convConfig, dtype=object)
        
        np.save(path + filename + '_convConfig.npy', convConfig)

        np.save(path + filename + '_filtersValues.npy', np.asarray(self.filtersValues, dtype=np.float32))

            
            
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
        #print('ns: ', ns)
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
        n_layer0 = -1
        
        self.hiddenL = copy.deepcopy(self.hiddenL2)
                
        hiddenW = [None] * (len(self.hiddenL) + 1)
        
        self.lastLayerNeurons = y.shape[1]
                
        self.hiddenL.append(y.shape[1])
        
        self.convOutputs = []
        
        self.printVerbose('Training started with', x.shape[0], 'samples')
        
        if self.batch_size == 1:
            numIterations = x.shape[0]
        else:
            numIterations = math.ceil(x.shape[0] / self.batch_size)
            
        numIterations = int(numIterations * (1 - self.iterationDrop))
            
        
        for epochs in range(0, self.numEpochs):
            
            meanCostByEpochE = 0
            
            batch_pos = 0
            

            if epochs < self.convEpochs:
                xy_ind = np.arange(x.shape[0])
            else:
                xy_ind = np.arange(len(self.convOutputs))
            
            if self.shuffle:
                np.random.shuffle(xy_ind)
                        
            for x_pos in range(0, numIterations):
                
                if epochs < self.convEpochs:
                    
                    if self.batch_size == 1:
                        c_positions = xy_ind[x_pos]
                    else:
                        if (batch_pos + self.batch_size) < xy_ind.shape[0]:
                            c_positions = xy_ind[batch_pos:batch_pos+self.batch_size]
                        else:
                            c_positions = xy_ind[batch_pos:]

                    x_val = x[c_positions]

                    x_val_batch_s = x_val.shape[0]
                
                
                    
                    last_layer_output = self.convLayersFeedForward(x_val)
                    x_val = last_layer_output.flatten()
                    
                    if self.batch_size != 1:
                        x_val = x_val.reshape(x_val_batch_s, int(x_val.shape[0] / x_val_batch_s))
                    
                    if epochs == (self.convEpochs - 1):
                    
                        self.convOutputs.append([x_val, c_positions])
                        
                else:
                    
                    x_val, c_positions = self.convOutputs[xy_ind[x_pos]]
                
                #self.log('x_val:', x_val.shape, x_val)
                                
                #print(x_val.shape)
                
                if n_layer0 == -1:
                    
                    if self.batch_size == 1:
                        n_layer0 = x_val.shape[0]
                    else:
                        n_layer0 = x_val.shape[1]
                        
                    self.n_layer0 = n_layer0
            
                v_layer = x_val 
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
                     
                    
                    lastN = hiddenLayer
                    
                coste_anterior = None

                i = len(self.hiddenL) - 1
                
                #print(f_vlayer)
                
                """
                if(self.softmax):
                    f_vlayer = self.softmaxF(f_vlayer).reshape(-1)
                    self.log('f_vlayer (Softmax output):', f_vlayer)
                    #print(f_vlayer)
                """
                    
                #print(f_vlayer, '\n\n')
                
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
                        
                        #coste = coste.reshape(-1)
                        #coste = coste.reshape(coste.shape[0], 1)
                        
                        
                        #if self.batch_size != 1:
                            #coste = np.sum(coste, axis=0)
                            #derivf_coste = np.sum(derivf_coste, axis=0)
                            
                            
                             
                        if self.debugMode > 0:
                            meanCostByEpochE = meanCostByEpochE + (abs(coste) if self.batch_size == 1 else np.mean(np.absolute(coste), axis=0))
                            
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
                          
                        #print("coste_anterior.shape: ", coste_anterior.shape)
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
                    #print('Coste anterior shape: ', coste_anterior.shape)
            
                if epochs < self.convEpochs: # because of resources limitations
                    self.convLayersBackpropagation(last_layer_output, coste_anterior)
                
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
            n_layer0 = -1
            
            layerValues = np.zeros(shape=(x.shape[0],self.lastLayerNeurons))
        
        
            batch_pos = 0
            
            if self.batch_size == 1:
                numIterations = x.shape[0]
            else:
                numIterations = math.ceil(x.shape[0] / self.batch_size)
                        
            for x_pos in range(0, numIterations):
                
                if self.batch_size == 1:
                    x_val = x[x_pos]
                else:
                    if (batch_pos + self.batch_size) < x.shape[0]:
                        x_val = x[batch_pos:batch_pos+self.batch_size]
                    else:
                        x_val = x[batch_pos:]
                        
                    x_val_batch_s = x_val.shape[0]
                    
                    
            #for x_pos, x_val in enumerate(x):
                
                x_val = self.convLayersFeedForward(x_val).flatten()
                if self.batch_size != 1:
                    x_val = x_val.reshape(x_val_batch_s, int(x_val.shape[0] / x_val_batch_s))
                
                if n_layer0 == -1:
                    
                    n_layer0 = x_val.shape[0]
                    self.n_layer0 = n_layer0
            
                v_layer = x_val
                lastN = n_layer0
                
                f_vlayer = self.ActivationFunction(v_layer, 'identity')
                
                for i, hiddenLayer in enumerate(self.hiddenL):
                    entries = hiddenLayer * lastN

                    valuesForPerc = int(entries / hiddenLayer)

                    firstPos = 0
                    lastPos = valuesForPerc

                    v_layer = f_vlayer.dot(self.hiddenWeights[i])
                    
                    if self.pre_norm and (i < (len(self.hiddenL) - 1)):
                        v_layer = self.pre_norm_forward_FC(v_layer)
                    
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
                    
                    v_layer = f_vlayer
                                        
                    lastN = hiddenLayer
                    
                if self.batch_size == 1:
                    layerValues[x_pos] = f_vlayer
                else:
                    if (batch_pos + self.batch_size) < x.shape[0]:
                        layerValues[batch_pos:batch_pos+self.batch_size] = f_vlayer
                    else:
                        layerValues[batch_pos:] = f_vlayer
                        
                    batch_pos = batch_pos + self.batch_size
                    
                
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
            print('ConvNet debug mode must be level 1 or higher')

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
            print('ConvNet debug mode must be level 2 or higher')