'''
Author : Ihsaan Malek Fall 2020, Comp 432

Original Paper : https://arxiv.org/pdf/1803.01271.pdf
Original Github Repo: https://github.com/locuslab/TCN

TCN Model
'''

from torch import nn
import numpy as np
import torch.nn.utils 
import torch


class Residual_Block(nn.Module):
    def __init__(self, n_inputs, n_outputs,kernel_size=3,stride=(1,1),dilation=2,padding = 0, 
               dropout = 0.2, weight_normalization=True):
    
        super(Residual_Block, self).__init__()
        
        self.n_inputs = n_inputs #input channel
        self.n_outputs = n_outputs #output channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        
        self.conv1 = nn.Conv1d(in_channels = n_inputs,out_channels = n_outputs,
                               kernel_size = kernel_size,stride = stride, padding = padding, dilation =dilation)
        self.activation1 =nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels = n_outputs,out_channels = n_outputs,
                               kernel_size = kernel_size,stride = stride, padding = padding, dilation=dilation)
        self.activation2 =nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        if weight_normalization:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv2 = nn.utils.weight_norm(self.conv2)
            
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs,n_outputs,1)
            
        self.activation3 = nn.ReLU()
        
    #padding left of data since pytorch only padds bi-directionally
    def forward(self, x):
        residuals = x
        
        #padding for first convnet
        left_padding = ((self.dilation)*(self.kernel_size-1))
        pad = torch.from_numpy(np.zeros(left_padding*x.shape[0]*x.shape[1]).reshape(x.shape[0],self.n_inputs,left_padding).astype('float32'))
        x = torch.cat((pad,x),2) #concat along input width aka time
        
        output = self.conv1(x)
        output = self.activation1(output)
        output = self.dropout1(output)
     
        
        #print('conv 2')
        #print('original input shape: ', output.shape)
        #conv2
        left_padding2 = ((self.dilation)*(self.kernel_size-1))#(2**self.dilation*(self.kernel_size-1)) #<- original
        #need extra padding across out channels
        pad2 = torch.from_numpy(np.zeros(left_padding2*output.shape[0]*output.shape[1]).reshape(output.shape[0]
                                                                    ,self.n_outputs,left_padding2).astype('float32'))

        output = torch.cat((pad2,output),2) #concat along input width aka time     
        
        output = self.conv2(output)
        output = self.activation2(output)
        output = self.dropout2(output)
        
        #if input channels and output channels are of different size use a 1D conv with kernel 1    

        if self.n_inputs != self.n_outputs:
            residuals = self.downsample(residuals)

        return self.activation3(output + residuals)
    
    
#To use th model, you can this class
class TemporalConvolutionalNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, weight_normal= True, dilation_fixed =False, dilation =2):
        super(TemporalConvolutionalNet, self).__init__()
        layers = []
        num_levels= len(num_channels)
        for i in range(num_levels):
            if dilation_fixed == True:
                dilation_size = dilation
            else:
                dilation_size = 2**i
            if i==0:
                input_channels = num_inputs
            else:
                input_channels = num_channels[i-1]
            out_channels = num_channels[i]
            layers+=[Residual_Block(n_inputs = input_channels, n_outputs = out_channels,kernel_size = kernel_size,
                                    stride=1, dilation = dilation_size,padding = 0, 
                                    dropout = dropout,weight_normalization = weight_normal )]
        print('layers')
        for j in layers:
            print(j.n_inputs,j.n_outputs)
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)