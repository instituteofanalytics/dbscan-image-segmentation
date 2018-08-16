#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:41:25 2018

@author: Jesrael
"""

from imageio import imread
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.signal import convolve2d

MAX_PATH_LENGTH = 7

def read_data(csv_file,image_id):
    "To read the image data masks associated with a single image"
    labels = pd.read_csv(csv_file)
    image = imread(csv_file.replace('_labels.csv','/')+
                 image_id+'/images/'+image_id+'.png')[:,:,:3]
    
    #Image size...
    rows,cols = image.shape[:2]
    masks = []
    #Construct masks for each nucleus seperately...
    for rle in labels.EncodedPixels[image_id==labels.ImageId]:
        mask = np.zeros(rows*cols,dtype=np.uint8)
        pixels = np.array([int(x) for x in rle.split(' ')],
                           dtype=np.uint32).reshape(-1,2)
        for p,l in pixels:
            mask[(p-1):(p-1+l)] = 1 #Note, the given data is 1-indexing
        
        mask = mask.reshape(rows,cols,order = 'F')
        masks.append(mask)
    
    return image,masks
    
def normalize_image(image):
    "Set background to 0, variance to 1, invert bright background"
    
    #Flat array of image data in each channel
    R,G,B = np.transpose(image.reshape(-1,3))
    
    r_vls,r_frq = np.unique(R,return_counts = True)
    g_vls,g_frq = np.unique(G,return_counts = True)
    b_vls,b_frq = np.unique(B,return_counts = True)
        
    #Subtract the backgrounds
    R-=r_vls[np.argmax(r_frq)]
    G-=g_vls[np.argmax(g_frq)]
    B-=b_vls[np.argmax(b_frq)]

    #Invert if most values are negative
    if np.mean(R)<0:
        R = -R
    if np.mean(G)<0:
        G = -G
    if np.mean(B)<0:
        B = -B
    
    #Now, set values close to background to be zero...
    R[R<=-np.median(R[R<0])] = 0
    G[G<=-np.median(G[G<0])] = 0
    B[B<=-np.median(B[B<0])] = 0
    
    #Divide by mean of the non-zero values
    R /= R[R>0].mean()
    G /= G[G>0].mean()
    B /= B[B>0].mean()
    
    #Now, create a combined image, and use tanh to flatten outliers...
    image = np.stack([R,G,B],axis = -1).reshape(image.shape)
    return image

#E- norm of a RGB colour
en3 = lambda c : (c[0]**2 + c[1]**2 + c[2]**2)**0.5

#Precompute distances using a kdtree and return a sparse matrix...
def precompute_distances(image_normalized):
    "This function will precompute a sparse matrix of distances in CSR format"
    cds = lambda max_c,min_c : (max_c  - min_c,1/(0.1+min_c))
    
    rows,cols = image_normalized.shape[:2]    
    #The the set of nearest neighbours and distances
    nn_set = np.empty((rows,cols),object)
    
    def kd_tree(path,split_dir = 'b',max_c = None,min_c = None):
        "Calculate distances to all neighbours"
        #The current end point of the path under consideration
        s_x,s_y = path[0]
        i_x,i_y = path[-1]
        if not max_c:
            max_c = en3(image_normalized[i_x,i_y])
        if not min_c:
            min_c = en3(image_normalized[i_x,i_y])
        
        #We put a limit on the depth of recursion..
        if len(path)>MAX_PATH_LENGTH:
            return
        
        #Set of recursion directions...
        recur = []
        
        if split_dir!='x':  #We need to split in the y-direction
            if i_y!=0:
                nbr = [i_x,i_y-1] ; nbr_x,nbr_y = nbr
                if nbr not in path:
                    #Update max_c and min_c
                    ma = max(max_c,en3(image_normalized[nbr_x,nbr_y]))
                    mi = min(min_c,en3(image_normalized[nbr_x,nbr_y]))
                    d_r,d_min = cds(ma,mi)
                    key = str(nbr_x)+','+str(nbr_y)
                    #Update the pair of distances
                    if key in nn_set[s_x,s_y].keys():
                        nn_set[s_x,s_y][key]= (min(d_r,nn_set[s_x,s_y][key][0])
                        ,min(d_min,nn_set[s_x,s_y][key][1]))
                    else:
                        nn_set[s_x,s_y][key] = (d_r,d_min)
                    recur.append((nbr,'x',ma,mi))
            if i_y!= cols-1:
                nbr = [i_x,i_y+1]; nbr_x,nbr_y = nbr
                if nbr not in path:
                    #Update max_c and min_c
                    ma = max(max_c,en3(image_normalized[nbr_x,nbr_y]))
                    mi = min(min_c,en3(image_normalized[nbr_x,nbr_y]))
                    d_r,d_min = cds(ma,mi)
                    key = str(nbr_x)+','+str(nbr_y)
                    #Update the pair of distances
                    if key in nn_set[s_x,s_y].keys():
                        nn_set[s_x,s_y][key]= (min(d_r,nn_set[s_x,s_y][key][0])
                        ,min(d_min,nn_set[s_x,s_y][key][1]))
                    else:
                        nn_set[s_x,s_y][key] = (d_r,d_min)
                    recur.append((nbr,'x',ma,mi))
                        
        if split_dir!='y':  #We need to split in the y-direction
            if i_x!=0:
                nbr = [i_x-1,i_y]; nbr_x,nbr_y = nbr
                if nbr not in path:
                    #Update max_c and min_c
                    ma = max(max_c,en3(image_normalized[nbr_x,nbr_y]))
                    mi = min(min_c,en3(image_normalized[nbr_x,nbr_y]))
                    d_r,d_min = cds(ma,mi)
                    key = str(nbr_x)+','+str(nbr_y)
                    #Update the pair of distances
                    if key in nn_set[s_x,s_y].keys():
                        nn_set[s_x,s_y][key]= (min(d_r,nn_set[s_x,s_y][key][0])
                        ,min(d_min,nn_set[s_x,s_y][key][1]))
                    else:
                        nn_set[s_x,s_y][key] = (d_r,d_min)
                    recur.append((nbr,'y',ma,mi))
            if i_x!= rows-1:
                nbr = [i_x+1,i_y]; nbr_x,nbr_y = nbr
                if nbr not in path:
                    #Update max_c and min_c
                    ma = max(max_c,en3(image_normalized[nbr_x,nbr_y]))
                    mi = min(min_c,en3(image_normalized[nbr_x,nbr_y]))
                    d_r,d_min = cds(ma,mi)
                    key = str(nbr_x)+','+str(nbr_y)
                    #Update the pair of distances
                    if key in nn_set[s_x,s_y].keys():
                        nn_set[s_x,s_y][key]= (min(d_r,nn_set[s_x,s_y][key][0])
                        ,min(d_min,nn_set[s_x,s_y][key][1]))
                    else:
                        nn_set[s_x,s_y][key] = (d_r,d_min)
                    recur.append((nbr,'y',ma,mi))
        
        #Now, do the recursion to the next level of neighbours...
        for r,s_d,*m in recur:
            kd_tree(path = path+[r],split_dir = s_d,max_c = m[0],min_c = m[1])
    
    #Now, we run the kd_tree function for all pixels...multiprocessing later?
    for r in range(rows):
        for c in range(cols):
            #The distances from a point to itself...
            nn_set[r,c] = {str(r)+','+str(c):(0.0,
                  1/(1e-3 + en3(image_normalized[r,c])))}
            kd_tree([[r,c]])
    
    #Construct a CSR matrix for the first distance measure...
    data = []
    indices = []
    indptr = [0]
    for r in range(rows):
        for c in range(cols):
            #Non-zero column indices...
            non_0_cols =[int(i.split(',')[0])*cols + int(i.split(',')[1]) 
                        for i in nn_set[r,c].keys()]
            values = [v[0] for v in nn_set[r,c].values()]
            
            #Add this to the CSR format...
            indices.extend(non_0_cols)
            data.extend(values)
            indptr.append(indptr[-1]+len(values))
    
    #The precomputed distance...
    pcd_r = csr_matrix((data, indices, indptr), shape=(rows*cols, rows*cols))
    
    #Construct a CSR matrix for the second distance measure...
    data = []
    indices = []
    indptr = [0]
    for r in range(rows):
        for c in range(cols):
            #Non-zero column indices...
            non_0_cols =[int(i.split(',')[0])*cols + int(i.split(',')[1]) 
                        for i in nn_set[r,c].keys()]
            values = [v[1] for v in nn_set[r,c].values()]
            
            #Add this to the CSR format...
            indices.extend(non_0_cols)
            data.extend(values)
            indptr.append(indptr[-1]+len(values))
    
    #The precomputed distance...
    pcd_m= csr_matrix((data, indices, indptr), shape=(rows*cols, rows*cols))
    return pcd_r,pcd_m

def IoU(ground_truth_set,prediction_set):
    "Function to calculate IoU, given a predicted set & ground truth set"
    IoU = np.zeros((len(ground_truth_set),len(prediction_set)),'f8')
    for n,gts in enumerate(ground_truth_set):
        for c,ps in enumerate(prediction_set):
            intersection = np.logical_and(gts,ps)
            union = np.logical_or(gts,ps)
            IoU[n,c] = np.sum(intersection)/np.sum(union)
            
    return IoU

def MAP(ground_truth_set,prediction_set):
    "Function to calculate MAP"
    iou = IoU(ground_truth_set,prediction_set)
    APs = []
    for t in np.arange(0.5,0.99,0.05):
        tp = (iou > t).sum()
        fp = iou.shape[1] - tp
        fn = iou.shape[0] - tp
        APs.append(tp/(tp+fp+fn))
        
    return (sum(APs)/len(APs))

#Def to convert mask to interior
def mask2interior(mask):
    mask_padded = np.zeros((mask.shape[0]+2,mask.shape[1]+2),'i1')
    mask_padded[1:-1,1:-1] = mask
    mask_padded[[0,-1],1:-1] = mask[[0,-1],:]
    mask_padded[:,[0,-1]] = mask_padded[:,[1,-2]]
    
    #Edge filter
    edge_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],'i1')
    not_inner_edge = convolve2d(mask_padded,edge_filter,'valid') <= 0
    interior = mask * not_inner_edge.astype('u1')
    return interior

#Inverse of the above...
def interior2mask(interior):
    interior_padded = np.zeros((interior.shape[0]+2,interior.shape[1]+2),'i1')
    interior_padded[1:-1,1:-1] = interior
    interior_padded[[0,-1],1:-1] = interior[[0,-1],:]
    interior_padded[:,[0,-1]] = interior_padded[:,[1,-2]]

    #Edge filter
    edge_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],'i1')
    outer_edge = convolve2d(interior_padded,edge_filter,'valid') < 0
    mask = interior + outer_edge.astype('u1')
    return mask
