# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:00:59 2020

@author: Ihsaan
Preshaing the data to the appropriate shape

A TCN needs data in the shape (samples, features, time_steps)

"""

import numpy as np
import torch



def mulitvariate_preprocessing(num_features,time_steps, df, num_predictions, sampling_method):
    '''
    Ex:
    num_predictions = 2
    input : t1,t2,t3,t4,t5
    target : t3,t4,t5,t6,t7
    predictions = t6,t7
    sampling method: either you push the sliding window along the data or you jump ie select sample [0,timestep],[timestep,2*timstep] etc..
    
    '''  
    
    sampling_index = 0
    end_sampling_index = time_steps

    #set final subsampling shape Temp array to append actual data in the right shape
    temp = np.arange(time_steps) 
    for i in range(num_features-1):
        temp = np.vstack((temp,np.arange(time_steps))) #end of for loop you get 4 depth,20  input length shape
    temp = np.vstack(([temp],[temp])) #shape (2,4,20) or (samples, channels/features, time length)


    train_Set = df.iloc[sampling_index:end_sampling_index, [0]].to_numpy().flatten() #first time_steps samples
    #print(train_Set)
        
    if sampling_method == 'sliding_window':
        
        
        target_start = num_predictions  #retrieve t+1
        target_end = target_start + time_steps
        y_train = df.iloc[target_start:target_end, [0]].to_numpy().flatten() #first target samples

        
        for i in range((df.shape[0]-num_predictions-time_steps)+num_predictions): #loop of data to generate samples
  
            for i in range(1,num_features): #loop to iterate over features for x
               
                train_Set = np.vstack((train_Set,
                                   df.iloc[sampling_index:end_sampling_index, [i]].to_numpy().flatten()))
                
            #print('x',train_Set[0:1,])
            
            sampling_index += num_predictions
            end_sampling_index += num_predictions
            
            target_start += num_predictions
            target_end += num_predictions
            
                
            temp = np.vstack((temp,[train_Set])) #shape n, feature,time length
            train_Set = df.iloc[sampling_index:end_sampling_index, [0]].to_numpy().flatten() #reset to get shape 20, -> array of sub-sampled prices
            
            if target_end > df.shape[0]:
                break #error checking condition
            #y-value
            y_train = np.vstack((y_train,
                                        df.iloc[target_start:target_end, [0]].to_numpy().flatten()))

        y_train = y_train.reshape(-1,1,time_steps).astype('float32') #reshape to sample, 1 ,timesteps
            
            
    else:
        #jumping by timesteps
        
        for i in range(int(df.shape[0])//time_steps):
            for i in range(1,num_features): #loop to iterate over features for x

                train_Set = np.vstack((train_Set,
                                       df.iloc[sampling_index:end_sampling_index, [i]].to_numpy().flatten()))
                
            #print('x',train_Set[0:1,])
            sampling_index += time_steps
            end_sampling_index += time_steps
            
            temp = np.vstack((temp,[train_Set])) #shape n, feature,time length
            train_Set = df.iloc[sampling_index:end_sampling_index, [0]].to_numpy().flatten() #reset to get shape 20, -> array of sub-sampled prices
    

        y_train = df['close'].tail(df.shape[0]-num_predictions) # last n - num_prediction values
        y_train = y_train.to_numpy().flatten().reshape(-1,1,time_steps).astype('float32')    
        
   
    
    print('Finish resampling')
    
    temp = temp[2:,:,:].astype('float32')
      
    #print('x',temp.shape)#nice shape is good
    #print(temp[:1,:,:])#check subsample for proper format
    #print('y',y_train.shape)
    
    X_train_torch = torch.from_numpy(temp)
    
    y__train_torch = torch.from_numpy(y_train) # n_Samples x 1 x temporal length
    
    return X_train_torch, y__train_torch