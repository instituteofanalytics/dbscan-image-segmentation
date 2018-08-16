# DBSCAN for Data Science Bowl 2018

The Data Science Bowl 2018 was a Kaggle competition that challenged participants to identify cellular nuceli in biomedical microscopic images.The images were of multiple sizes and resolutions, in various lighting conditions, and of various cell types. The goal was to predict the regions of the image where nuclei were located i.e. this is a problem of image segmentation each pixel needs to be labelled as nucleus, or background.

Most approaches to the problem used U-net neural networks or mask-RCNN, but given that training set is small, it is not clear that a deep learning model with thousands of parameters is the most efficient or accurate way to approach the problem. Here we approach this as a clustering problem, using Density-Based Spatial Clustering of Applications with Noise.

In short DBSCAN uses uses a distance measure between points, to define a neighbour as lying within less than a distance ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon). A core point is one with more than minPts neighbours, where  ![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon) and minPts are parameters. Plotting a graph of connections between neighbours, we get a clustering rule i.e. all core points that are linked to each other through a path of core points belong to the same cluster. Points which don't belong to any cluster are noise points.

# Sample images
A few sample images:

![Alt text](samples/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png?raw=true)
![Alt text](samples/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png?raw=true)
![Alt text](samples/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe.png?raw=true)

# Image pre-processing

The preprocessing involves several steps - first identify the colour of the background, this is most likely the pixel value  that is most frequent (i.e. the statistical mode) for each of the channels R,G and B, and then subtract this value from the data array in that channel:

![equation](http://latex.codecogs.com/gif.latex?RGB_%7Bij%7D%20%5Crightarrow%20%28R%20-%20R_%7Bmode%7D%29%28G%20-%20G_%7Bmode%7D%29%28B%20-%20B_%7Bmode%7D%29_%7Bij%7D)

If it was a brightfield image, invert it, and set all values close to the background to be zero. Divide by the mean in each channel, and we have a normalized image.

*Original*
![Alt text](samples/image.png?raw=true)
*Processed*
![Alt text](samples/image_normalized.png?raw=true)

# Custom Distance Function

The key in any clustering algorithm is to find a meaningful and useful distance function. We need a function that will put pixels close together if they belong to the same nucleus, and further apart, if they belong to  different nuclei. This is determined, by considering all the paths to connecting two pixels, and if there is a path connecting them with little change in colour along the path:

![equation](http://latex.codecogs.com/gif.latex?measure%5C_A_%7B%28ij%29%20%5Cleftrightarrow%20%28kl%29%7D%20%3D%20%5Cinf_%7B_%7Bp%20%5C%20%5Cepsilon%20%5C%20%28ij%29%5Cleftrightarrow%28kl%29%7D%7D%5C%7Bmax%28%7C%7CRGB%7C%7C_p%29%20-%20min%28%7C%7CRGB%7C%7C_p%29%20%5C%7D)

We also can consider a distance as low if there is a path connecting the two  points that  doesn't pass through the background:

![equation](http://latex.codecogs.com/gif.latex?measure%5C_B_%7B%28ij%29%20%5Cleftrightarrow%20%28kl%29%7D%20%3D%20%5Cinf_%7B_%7Bp%5Cepsilon%28ij%29%5Cleftrightarrow%28kl%29%7D%7D%20%5CBig%5C%7B%20%5Cfrac%7B1%7D%7B%5Cdelta%20&plus;%20%5Cmin%28%7C%7CRGB%7C%7C%29_p%7D%20%5CBig%5C%7D)

A small value ![equation](http://latex.codecogs.com/gif.latex?%5Cdelta) is there to prevent singularities. In practice a value of ![equation](http://latex.codecogs.com/gif.latex?%5Cdelta%3D0.1) seems to work well. 

We have used a custom distance function takes a linear combination of the above, thus incorporating both ideas:

![equation](http://latex.codecogs.com/gif.latex?custom%5C_distance%20%3D%20%5Calpha%20%5Ctimes%20measure%5C_A%20&plus;%20%5Cbeta%20%5Ctimes%20measure%5C_B)

The parameters ![equation](http://latex.codecogs.com/gif.latex?%5Calpha%20%5C%20%5C%26%20%5C%20%5Cbeta) are determined for the training data by searching for the set that will minimize the error. But to apply this method for test data, we need to develop an algorithm that will predict appropriate values for the parameters from the image itself.
