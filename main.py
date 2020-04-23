# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 00:48:40 2019

@author: TEJAS
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from Functions import imgLibs,plotLibs,computeLibs
from skimage.filters import rank
from skimage.morphology import watershed, disk
from scipy import ndimage as ndi
from sklearn.decomposition import PCA



#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgNum = 1             

if imgNum==1:
    imga = np.load('Dataset/0270MR0011860360203259E01_DRLX.npy')
    imga[np.isnan(imga)] = 0
    plotLibs().dispImg(imga)
    imga = cv2.resize(imga,(512,512))
    slic0_params = {'colorSpace_s':'lab','n_segments':250,'max_iter':1000,'hcThresh':0.1,'ncuts':{'tresh':0.09,'num_cuts':100}}
    decorImg = computeLibs().decorrstretch(imga)
    plotLibs().dispImg(decorImg)
    stretch_img = (imga-np.min(imga))/(np.max(imga)-np.min(imga))
    plotLibs().dispImg(stretch_img)
    labels = computeLibs().slic0(slic0_params,cv2.cvtColor(stretch_img,cv2.COLOR_BGR2Lab))    
    new_labels = computeLibs().graphMergeHierarchical(decorImg,labels,thresh=0.2)
    plotLibs().plotBoundaries(imga,new_labels)

if imgNum==2:
    imga = np.load('Dataset/0153MR0008490000201265E01_DRLX.npy')
    imga[np.isnan(imga)] = 0
    plotLibs().dispImg(imga)
    imga = cv2.resize(imga,(512,512))
    stretch_img = (imga-np.min(imga))/(np.max(imga)-np.min(imga))
    plotLibs().dispImg(stretch_img)
    pca = PCA(n_components=1)
    pca_transformed = pca.fit_transform(stretch_img.reshape((512*512,-1)))
    new_featureset = pca_transformed.reshape(512,512)
    plotLibs().dispGrayImg(new_featureset)     
    labels=computeLibs().watershed(pca_transformed.reshape(512,512),smoothen = 5,extrasmoothen=12,con_grad=15)    
    plotLibs().plotBoundaries(imga,labels,'gray')
    new_labels = computeLibs().graphMergeHierarchical(imga,labels,thresh=0.14)

if imgNum==3:
    imga = np.load('Dataset/0172ML0009240000104879E01_DRLX.npy')
    imga[np.isnan(imga)] = 0
    plotLibs().dispImg(imga)
    imga = cv2.resize(imga,(512,512))
    stretch_img = (imga-np.min(imga))/(np.max(imga)-np.min(imga))
    plotLibs().dispImg(stretch_img) 
    felzParam = {'colorSpace_s':'rgb','scale':300,'sigma':0.4,'min_size':250,'hcThresh':0.08,'ncuts':{'tresh':0.5,'num_cuts':200}}
    decorImg = computeLibs().decorrstretch(imga)
    plotLibs().dispImg(decorImg)
    stretch_img = (imga-np.min(imga))/(np.max(imga)-np.min(imga))
    plotLibs().dispImg(stretch_img)
    labels = computeLibs().felzenszwalb(felzParam,stretch_img)    
    new_labels = computeLibs().graphMergeHierarchical(decorImg,labels,thresh=0.16)
    plotLibs().plotBoundaries(imga,new_labels)
    
if imgNum==4:
    imga = np.array(Image.open(('Dataset/0053MR0002430000102993E01_DRCL.png')))
    plotLibs().dispImg(imga)
    imga = cv2.resize(imga,(512,512))
    stretch_img = (imga-np.min(imga))/(np.max(imga)-np.min(imga))
    plotLibs().dispImg(stretch_img) 
    felzParam = {'colorSpace_s':'rgb','scale':100,'sigma':0.55,'min_size':250,'hcThresh':0.08,'ncuts':{'tresh':0.5,'num_cuts':200}}
    decorImg = computeLibs().decorrstretch(imga)
    plotLibs().dispImg(decorImg)
    stretch_img = (imga-np.min(imga))/(np.max(imga)-np.min(imga))
    plotLibs().dispImg(stretch_img)
    labels = computeLibs().felzenszwalb(felzParam,stretch_img)    
    new_labels = computeLibs().graphMergeHierarchical(decorImg,labels,thresh=0.05)
    plotLibs().plotBoundaries(imga,new_labels)






