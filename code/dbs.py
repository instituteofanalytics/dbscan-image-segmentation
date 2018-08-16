#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:45:08 2018

@author: Jesrael
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dbslib import read_data,normalize_image,precompute_distances,MAP
from dbslib import mask2interior,interior2mask
import os, pickle

#Fixed Parameters
EPS = 1.0
MIN_PTS = 20
#############################
#Parameters to optimize...
ALPHA = 0.25
BETA = 2.0

def optimize_parameters(image_normalized,combined_mask):
    "Function to find values of epsilon, alpha and beta to maximize accuracy"
    d_r,d_m = precompute_distances(image_normalized)

    #Function to optimize...to predict actual interior
    def opt_func(prm):
        "Function to minimize"
        dbs.fit(prm[0]*d_r + prm[1]*d_m)
        clusters = dbs.labels_.reshape(image_normalized.shape[:2])
        
        #To calculate an optimization function...
        prediction_set = [clusters==c for c in range(np.max(clusters)+1)]
        
        #If no clusters are predicted...
        if not len(prediction_set):
            return 0

        #Map predicted_mask to combined_mask
        combined_prediction = np.stack(prediction_set).sum(0,'u1')
        
        #Error 1  - predicted mask, but actually background
        e1 = np.logical_and(1-combined_mask,combined_prediction).sum()
        #Error 2 - predicted background but actually mask
        e2 = np.logical_and(combined_mask,1-combined_prediction).sum()
        union = np.logical_or(combined_mask,combined_prediction).sum()
        
        error = (e1 + e2)/union
        return error
        
    ans = minimize(opt_func,x0 = (ALPHA,BETA),method = 'Nelder-Mead',
                   options={'disp':True})
    dbs.fit(ans.x[0]*d_r + ans.x[1]*d_m)
    clusters = dbs.labels_.reshape(image_normalized.shape[:2])
    return ans.x.tolist(),clusters

if __name__=='__main__':
    #Load the data and store it
    train_file = '../data/stage1_train_labels.csv'
    image_ids =image_ids = pd.read_csv(train_file,usecols = [0],
                                       squeeze=True).unique()
    output_folder = '../results/'
    if os.path.isfile(output_folder + 'parameters.p'):
        save_values = pickle.load(open(output_folder + 'parameters.p','rb'))
    else:
        save_values = {}
    processed_image_ids = os.listdir(output_folder)
    remaining_image_ids = [i for i in image_ids if i not in processed_image_ids]

    dbs = DBSCAN(EPS,MIN_PTS+1,'precomputed')
    for image_id in remaining_image_ids:
        print(image_id)
        image,masks = read_data(train_file,image_id)

        #From these, we can calculate the combined mask...
        combined_mask = np.stack(masks).sum(0,'u1')
        combined_interior = np.stack([mask2interior(m) for m in masks
                                      ]).sum(0,'u1')
        #Now, normalize the image
        image_normalized = normalize_image(image.astype('f8'))

        #Now, we use optimization techniques to find the right parameters
        ((ALPHA,BETA),clusters) = optimize_parameters(image_normalized,
                                       combined_interior)
        
        #plt.imsave() to save the png files...
        output_dir = output_folder + image_id
        os.mkdir(output_dir) if not os.path.isdir(output_dir) else None
        
        plt.imsave(output_dir + '/image.png',arr = image)
        plt.imsave(output_dir + '/image_normalized.png',
                   arr = image_normalized/np.max(image_normalized))
        plt.imsave(output_dir + '/combined_mask.png',
                   arr = (combined_mask+combined_interior)/2)
        plt.imsave(output_dir + '/predicted_interior.png',clusters)
        
        #To calculate MAP
        prediction_set = [interior2mask(clusters==c) for c in 
                          range(np.max(clusters)+1)]
        ground_truth_set = [m.astype(np.bool) for m in masks]
        
        accuracy = MAP(ground_truth_set,prediction_set)
        
        print(accuracy)
        save_values[image_id] = (ALPHA,BETA,accuracy)
    
        #The dictionary containing the parameters...
        pickle.dump(save_values,open(output_folder + 'parameters.p','wb'))
