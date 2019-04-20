# Single-Image Crowd Counting using Convolutional Neural Networks
Our project aims to develop models that can accurately estimate the crowd count from an individual image with arbitrary crowd density and arbitrary perspective. In this project, we propose two models to estimate crowd count. They are as follows: <br />
   (a) Multi-Column Convolutional Neural Network (MCNN): The MCNN model utilizes filters with receptive fields of different sizes to map the input image to its crowd density map. The MCNN model allows the input image to be of arbitrary size or resolution. The features learnt by each column CNN are adaptive to variations in people/head size due to perspective effect or image resolution. This will be an implementation of the paper "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network" by Zhang et al. <br /> <br />
   (b) CNN based Cascaded Multi-task learning of High level prior (CMTL): The CMTL model uses a cascaded network of CNNs to jointly learn crowd count classification and density map estimation. Classifying crowd count can be seen as coarsely estimating the total count in the image which can serve as a high-level prior to estimate the density map of the image. This enables the layers in the network to learn globally relevant discriminative features which aid in estimating highly refined density maps with lower count errors. This will be an implementation of the paper "CNN-based Casscaded Multi-task Learning of High-level prior and Density Estimation for Crowd Counting" by Sindagi et al. <br /> <br />
   We have implemented our project on Shanghaitech Dataset - Part A and Part B. The code is written in Python using the PyTorch framework. <br />
   
 Some of the results are as follows:
 
![Hi](/writeup/output_IMG_10.png)
![Hi](/writeup/output_IMG_123.png)
![Hi](/writeup/output_IMG_5.png)
![Hi](/writeup/output_IMG_15.png)
![Hi](/writeup/output_IMG_2.png)
![Hi](/writeup/output_IMG_1.png)
![Hi](/writeup/output_IMG_306.png)
![Hi](/writeup/output_IMG_315.png)
![Hi](/writeup/output_IMG_316.png)
![Hi](/writeup/output_IMG_132.png)
![Hi](/writeup/output_IMG_310.png)

