from __future__ import division
import numpy as np
import scipy as sc
import cv2, sys, time, json, copy, subprocess, os
from skimage import transform
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import matplotlib.image
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d,reconstruct_from_patches_2d# for divided image into overlapping blocks
from sklearn.preprocessing import normalize # normalize the array
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
def g2r(im):#function to transfor grayscale back to rgb
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret
def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step: int = 1):
    return view_as_windows(patches, patch_size, step)
def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculat the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor
def img_read(path1,path2):
    frame = cv2.imread(path1,cv2.IMREAD_GRAYSCALE) # load raw data
    frame2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE) # load masked img_data
    frame2_binary = cv2.threshold(frame2, 254, 255, cv2.THRESH_BINARY)[1] # convert masked img to binary image
    frame2_binary = np.uint8(frame2_binary) # unit 8 binary Image
    frame2_rgb = g2r(frame2_binary) # generate unit 8 rgb image from binary image
    red = np.array([255,0,0],dtype=np.uint8) # define red rgb color
    black = np.array([0,0,0],dtype=np.uint8) # define black rgb color
    white = np.array([255,255,255],dtype=np.uint8) #define white rgb color
    frame2_rgb[np.where((frame2_rgb == black).all(axis=2))] = red # change marked color from black to red
    frame2_rgba = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2RGBA) # change marked color from rgb to rgba
    frame2_rgba[np.where((frame2_rgb == white).all(axis=2))] = np.array([255,255,255,0],dtype=np.uint8) # transparent the white color
    return frame,frame2,frame2_binary,frame2_rgb,frame2_rgba
def patches_size_find(frame):
    rows, columns = frame.shape
    patches_size = 1
    count = 1
    while count < 6:
        if rows % patches_size == 0 and columns % patches_size == 0:
            count = count +1
        patches_size = patches_size +1
    patches_size = patches_size - 1
    print('patches size',patches_size,'rows',rows,'columns',columns)
    return patches_size
def generate_training(frame2_binary,training_sample,patches_size,patches_step):
    training_result = frame2_binary/255 # training result: 0 means graphene; 1 means no graphene
    patches_sample = patchify(training_sample,(patches_size,patches_size),step=patches_step) # generate patches for sample
    patches_result = patchify(training_result,(patches_size,patches_size),step=patches_step) # generate patches for result
    result = np.sum(patches_result, axis = (2,3)) # digitize the patches_result
    n = np.int(np.round(patches_size**2/2))
    result = (result > n).astype(float) # get result from patches: 0 means graphene; 1 means no graphene
    return training_sample,patches_sample,result
def feature_generate(patches_sample,result):
    a,b,c,d = patches_sample.shape
    feature_training = np.zeros([a,b,127])
    print('a',a,'b',b) # calculate the row and columns number
# %% feature generate for each patches
    for i in range(a): # using histogram generate bin edge and value of bins
        for j in range(b):
            hist_info= np.histogram(patches_sample[i,j,:,:].flatten(),bins = np.linspace(0,255,num=128))
            feature_training[i,j,:] = hist_info[0]
    feature_training = feature_training.reshape(a*b,127)
    result = result.reshape(a*b,1)
    #mean_feature = np.mean(patches_sample,axis=(2,3)).flatten().reshape(a*b,1)
    #scaler = StandardScaler()
    #mean_feature = scaler.fit(mean_feature.tolist()).transform(mean_feature.tolist())
    #mean_feature = mean_feature/max(mean_feature)
    return feature_training,result,a,b,c,d#,mean_feature
def PCA_analysis(n,feature_training):
    pca = PCA(n_components = n)
    scaler = StandardScaler()
    feature_training = scaler.fit(feature_training.tolist()).transform(feature_training.tolist())
    PCs = pca.fit(feature_training).transform(feature_training)
    return PCs
def KNN_analysis(result,n,PCs,L):
    result = result.reshape(L,)
    nbrs = KNeighborsClassifier(n_neighbors = n)
    nbrs.fit(PCs,result)
    return result,nbrs
def QDA_analysis(result,PCs,L):
    clf = QuadraticDiscriminantAnalysis()
    result = result.reshape(L,)
    clf.fit(PCs, result)
    return result,clf
def LDA_analysis(result,PCs,L):
    clf = LinearDiscriminantAnalysis()
    result = result.reshape(L,)
    clf.fit(PCs, result)
    return result,clf
def SVC_analysis(result,PCs,L):
    clf = SVC(gamma='auto')
    result = result.reshape(L,)
    clf.fit(PCs, result)
    return result,clf
def reconstruct_image_prediction(a,b,patches_size,y_predict,frame):
    y_new = np.zeros([a*b,patches_size,patches_size])
    for i in range(a*b):
        if y_predict[i] == 1:
            y_new[i,:,:] = np.ones([patches_size,patches_size])
        else:
            y_new[i,:,:] = np.zeros([patches_size,patches_size])
    y_new = y_new.reshape(a,b,patches_size,patches_size)
    y_new = unpatchify(y_new,frame.shape)
    m,n = y_new.shape
    predict_plot = np.empty([m,n,4],dtype = np.uint8)
    for i in range(m):
        for j in range(n):
            if y_new[i,j]  > 0:
                predict_plot[i,j,:] = np.array([255,255,255,0],dtype=np.uint8)
            else:
                predict_plot[i,j,:] = np.array([0,128,0,255],dtype=np.uint8)
    return predict_plot
def final_plot(frame,frame2,frame2_rgba,predict_plot):
    plt.subplot(2,2,1)# show original figure
    plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
    plt.subplot(2,2,2)# show mask from software
    plt.imshow(frame2, cmap = 'gray', interpolation = 'bicubic')
    plt.subplot(2,2,3)# show mask overlay with original figure
    plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
    plt.imshow(frame2_rgba,alpha = 0.3)
    plt.subplot(2,2,4)
    plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
    plt.imshow(predict_plot,alpha = 0.3)
