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
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# %% All functions
import function_user as uf
import importlib
from sklearn.preprocessing import StandardScaler
importlib.reload(uf)

#first image training

def get_training_data(path1,path2):
    frame1,frame1_2,frame1_2_binary,frame1_2_rgb,frame1_2_rgba = uf.img_read(path1,path2)
    training_sample1 = frame1.astype(float)
    patches_size = 8
    patches_step = np.int(patches_size/2)
    training_sample1,patches_sample1,result1 = uf.generate_training(frame1_2_binary,training_sample1,patches_size,patches_step)
    feature_training1,result1,a,b,c,d = uf.feature_generate(patches_sample1,result1)
    PCs1 = uf.PCA_analysis(2,feature_training1)
    return PCs1,result1,frame1,frame1_2,frame1_2_binary,frame1_2_rgb,frame1_2_rgba
path1 = 'training_data/training_set2/20171026_raw.tif' # define path1 for raw img_data
path2 = 'training_data/training_set2/20171026.tif' # define path2 for masked img_data
PCs1,result1,frame1,frame1_2,frame1_2_binary,frame1_2_rgb,frame1_2_rgba= get_training_data(path1,path2)
path3 = 'training_data/training_set2/2017111513_raw.tif' # define path1 for raw img_data
path4 = 'training_data/training_set2/2017111513.tif' # define path2 for masked img_data
PCs2,result2,frame2,frame2_2,frame2_2_binary,frame2_2_rgb,frame2_2_rgba = get_training_data(path3,path4)
path5 = 'training_data/training_set2/20170830.tif' # define path1 for raw img_data
path6 = 'training_data/training_set2/20170830_masked.tif' # define path2 for masked img_data
PCs3,result3,frame3,frame3_2,frame3_2_binary,frame3_2_rgb,frame3_2_rgba = get_training_data(path5,path6)
path7 = 'training_data/training_set2/20170829.tif' # define path1 for raw img_data
path8 = 'training_data/training_set2/20170829_masked.tif'
PCs4,result4,frame4,frame4_2,frame4_2_binary,frame4_2_rgb,frame4_2_rgba = get_training_data(path7,path8)
path9 = 'training_data/training_set2/20170829_3.tif' # define path1 for raw img_data
path10 = 'training_data/training_set2/20170829_3_masked.tif'
PCs5,result5,frame5,frame5_2,frame5_2_binary,frame5_2_rgb,frame5_2_rgba = get_training_data(path9,path10)

PC= np.concatenate((PCs1,PCs2,PCs3,PCs4,PCs5),axis=0)
result = np.concatenate((result1,result2,result3,result4,result5),axis=0)
#result,brain = uf.KNN_analysis(result,10,PC,len(result))
#histogram
plt.figure(figsize=(6,3))
hist1 = plt.hist(frame1.flatten(),bins = np.linspace(0,255,num=255))
plt.savefig('train1_hist.tif')
plt.figure(figsize=(6,3))
hist1 = plt.hist(frame2.flatten(),bins = np.linspace(0,255,num=255))
plt.savefig('train2_hist.tif')
plt.figure(figsize=(6,3))
hist1 = plt.hist(frame3.flatten(),bins = np.linspace(0,255,num=255))
plt.savefig('train3_hist.tif')
plt.figure(figsize=(6,3))
hist1 = plt.hist(frame4.flatten(),bins = np.linspace(0,255,num=255))
plt.savefig('train4_hist.tif')
plt.figure(figsize=(6,3))
hist1 = plt.hist(frame5.flatten(),bins = np.linspace(0,255,num=255))
plt.savefig('train5_hist.tif')






# plotting decision regions
X = PC
y = result.reshape(len(result),)
X.shape

qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)
f, axarr = plt.subplots(sharex='col', sharey='row', figsize=(16, 6))
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
cm = plt.cm.RdYlBu
cm_discrete = ListedColormap(['#c41f27','#3d5aa7'])
Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axarr.contourf(xx, yy, Z, cmap=cm, alpha=0.4)
idx = np.random.randint(len(y), size=2000)
axarr.scatter(X[idx, 0], X[idx, 1], c=y[idx], cmap=cm_discrete, s=40, edgecolor='k')

result,brain = uf.QDA_analysis(result,PC,len(result))
plt.savefig('QDACLASS.tif')


y_predict1 = brain.predict(PCs1)
y_predict2 = brain.predict(PCs2)
y_predict3 = brain.predict(PCs3)
y_predict4 = brain.predict(PCs4)
y_predict5 = brain.predict(PCs5)
confusion_matrix(result1,y_predict1)
confusion_matrix(result2,y_predict2)
confusion_matrix(result3,y_predict3)
confusion_matrix(result4,y_predict4)
confusion_matrix(result5,y_predict5)
patches_size = 8
patches_step = np.int(patches_size/2)
predict_plot1 = uf.reconstruct_image_prediction(223,319,patches_size,y_predict1,frame1)
predict_plot2 = uf.reconstruct_image_prediction(239,319,patches_size,y_predict2,frame2)
predict_plot3 = uf.reconstruct_image_prediction(253,383,patches_size,y_predict3,frame3)
predict_plot4 = uf.reconstruct_image_prediction(255,383,patches_size,y_predict4,frame4)
predict_plot5 = uf.reconstruct_image_prediction(255,383,patches_size,y_predict5,frame5)
plt.figure(1,figsize=(12,12))
uf.final_plot(frame1,frame1_2,frame1_2_rgba,predict_plot1)
plt.savefig('train1.tif')
plt.figure(2,figsize=(12,12))
uf.final_plot(frame2,frame2_2,frame2_2_rgba,predict_plot2)
plt.savefig('train2.tif')
plt.figure(3,figsize=(12,12))
uf.final_plot(frame3,frame3_2,frame3_2_rgba,predict_plot3)
plt.savefig('train3.tif')
plt.figure(4,figsize=(12,12))
uf.final_plot(frame4,frame4_2,frame4_2_rgba,predict_plot4)
plt.savefig('train4.tif')
plt.figure(5,figsize=(12,12))
uf.final_plot(frame5,frame5_2,frame5_2_rgba,predict_plot5)
plt.savefig('train5.tif')
#test 1
path11 = 'training_data/training_set2/CroppedImage.png' # define path1 for raw img_data
path12 = 'training_data/training_set2/FilteredImage.png' # define path2 for masked img_data
PCs6,result6,frame6,frame6_2,frame6_2_binary,frame6_2_rgb,frame6_2_rgba = get_training_data(path11,path12)
y_predict6 = brain.predict(PCs6)
confusion_matrix(result6,y_predict6)
predict_plot6 = uf.reconstruct_image_prediction(223,319,patches_size,y_predict6,frame6)
plt.figure(6,figsize=(12,12))
uf.final_plot(frame6,frame6_2,frame6_2_rgba,predict_plot6)
plt.savefig('test.tif')
# test 2
path13 = 'training_data/training_set2/test2.tif' # define path1 for raw img_data
path14 = 'training_data/training_set2/test2_masked.tif'
PCs7,result7,frame7,frame7_2,frame7_2_binary,frame7_2_rgb,frame7_2_rgba = get_training_data(path13,path14)
y_predict7 = brain.predict(PCs7)
confusion_matrix(result7,y_predict7)
predict_plot7 = uf.reconstruct_image_prediction(223,319,patches_size,y_predict7,frame7)
plt.figure(7,figsize=(12,12))
uf.final_plot(frame7,frame7_2,frame7_2_rgba,predict_plot7)
plt.savefig('test2.tif')
