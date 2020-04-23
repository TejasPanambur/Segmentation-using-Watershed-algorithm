# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:50:08 2019

@author: TEJAS
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation as seg
import cv2
from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage.measure import regionprops
from skimage import draw
import PIL.Image
from scipy.ndimage.filters import generic_filter
from itertools import product, chain
import scipy.stats as st
from sklearn.feature_extraction import image
from skimage.feature import greycomatrix,greycoprops
import multiprocessing
from joblib import Parallel, delayed
from sklearn.cluster import KMeans,SpectralClustering
from skimage.filters import rank
from skimage.morphology import watershed, disk
from scipy import ndimage as ndi
from plot_rag_merge import merge_mean_color,_weight_mean_color
from functools import reduce

class computeLibs:
    
   def __sharpen(self,img):
        # Create our shapening kernel, it must equal to one eventually
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
              
        return sharpened
   def preprocess(self,img,sharpen=False,cgray=True,resiz=True,plot=True):
       original_img = img
       if sharpen:
           img =  self.__sharpen(img)
       
       if resiz:
           img = cv2.resize(img,(512,512))
           color_img = cv2.resize(original_img,(512,512))

       if plot:
           plt.figure(figsize=(10,10))
           plt.title('Pre-Processed Image')
           plt.imshow(img,cmap='gray')
       return img,color_img
   
   
   def __MSE(self,Im1, Im2):
    	# computes error
    	Diff_Im = Im2-Im1
    	Diff_Im = np.power(Diff_Im, 2)
    	Diff_Im = np.sum(Diff_Im, axis=2)
    	Diff_Im = np.sqrt(Diff_Im)
    	sum_diff = np.sum(np.sum(Diff_Im))
    	avg_error = sum_diff / float(Im1.shape[0]*Im2.shape[1])
    	return avg_error
   
   def __KmeansHelper(self,img,no_of_clusters):       
        
        Kmean = KMeans(n_clusters=no_of_clusters)
        Kmean.fit(img)
        kmean_clusters = np.asarray(Kmean.cluster_centers_,dtype=np.float32)
        reconstructedImg = kmean_clusters[Kmean.labels_,:].reshape((512,512,-1))
        loss = self.__MSE(img.reshape((512,512,-1)),reconstructedImg)
        labels = Kmean.labels_.reshape((512,512,-1))
        return labels,loss,reconstructedImg
    
   def kmeans(self,img,no_of_clusters=6,bruteforceRange=(0,12),bruteforce=False):
        labels = None        
        if bruteforce:
            l,h = bruteforceRange
            loss = []
            reconstructedImg = []
            for i in range(l,h):
                print('Starting Clustering with'+str(i)+'centers')
                _,l,reconstImg = self.__KmeansHelper(img,i)                               
                loss.append(l)
                reconstructedImg.append(reconstImg) 
        else:
            labels,loss,reconstructedImg = self.__KmeansHelper(img,no_of_clusters)   
        return labels.reshape(512,512),no_of_clusters,loss,reconstructedImg  
   
   def watershed(self,img,smoothen = 5,extrasmoothen=2,con_grad=10,plot=True):
        smoothImg = rank.median(img, disk(smoothen))
        markers = rank.gradient(smoothImg, disk(smoothen)) < con_grad
        markers = ndi.label(markers)[0]
        gradient = rank.gradient(img, disk(extrasmoothen))
        
        labels = watershed(gradient, markers)
        if plot==True:
        
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                                     sharex=True, sharey=True)
            ax = axes.ravel()
            
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[0].set_title("Original")
            
            ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
            ax[1].set_title("Local Gradient")
            
            ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
            ax[2].set_title("Markers")
            
            ax[3].imshow(img, cmap=plt.cm.gray)
            ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.7)
            ax[3].set_title("Segmented")
            
            for a in ax:
                a.axis('off')
            
            fig.tight_layout()
            plt.show()
        return labels
    
   def slic(self,slicParam,img,plot=True,save=0):
        colorspace = slicParam['colorSpace_s']
        slic_img = seg.slic(img, n_segments=slicParam['n_segments'], compactness=slicParam['compactness'], max_iter=slicParam['max_iter'])
        if plot:
            if colorspace=='lab':
                img = cv2.cvtColor(img,cv2.COLOR_Lab2BGR)
            plotLibs().plotBoundaries(img,slic_img,save=save,title='Slic Segmented Image')
        return slic_img
   def felzenszwalb(self,felzParam,img,plot=True,save=0):
        colorspace = felzParam['colorSpace_s']
        felzenszwalb_img = seg.felzenszwalb(img,scale=felzParam['scale'],sigma=felzParam['sigma'],min_size=felzParam['min_size'])
        if plot:
            if colorspace=='lab':
                img = cv2.cvtColor(img,cv2.COLOR_Lab2BGR)
            plotLibs().plotBoundaries(img,felzenszwalb_img,save=save,title='F-H segmented Image')
        return felzenszwalb_img
    
   def labels2img(self,img,labels,plot=True,save=0):
        labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
        label_rgb = color.label2rgb(labels, img, kind='avg')
        if plot:
            plotLibs().plotBoundaries(label_rgb,labels,save=save,title='Label rgb Image')
        return label_rgb
    
   def graphMergeHierarchical(self,img, labels,thresh=0.15,plot=True,save=0):
        g = graph.rag_mean_color(img,labels)
        new_labels = graph.merge_hierarchical(labels, g, thresh, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)
        if plot:        
            plotLibs().dispImg(color.label2rgb(new_labels, img, kind='avg'),save=save,title='Merge Hierarchical Label rgb Image')
            plotLibs().plotBoundaries(img,new_labels,save=save,title='Merge Hierarchical')
        return new_labels
    
   def graphNormalizedCuts(self,img, labels,thresh=0.5,num_cuts=100,plot=True,save=0):
        g = graph.rag_mean_color(img, labels)
        new_labels = graph.cut_normalized(labels, g,thresh,num_cuts)
        if plot:
            plotLibs().dispImg(color.label2rgb(new_labels, img, kind='avg'),save=save,title='Ncut Label rgb Image')
            plotLibs().plotBoundaries(img,new_labels,save=save,title='Ncut Boundary Images')
        return new_labels
    
   def decorrstretch(self,A, tol=None):
        """
        Apply decorrelation stretch to image
        Arguments:
        A   -- image in cv2/numpy.array format
        tol -- upper and lower limit of contrast stretching
        """
    
        # save the original shape
        orig_shape = A.shape
        # reshape the image
        #         B G R
        # pixel 1 .
        # pixel 2   .
        #  . . .      .
        A = A.reshape((-1,3)).astype(np.float)
        # covariance matrix of A
        cov = np.cov(A.T)
        # source and target sigma
        sigma = np.diag(np.sqrt(cov.diagonal()))
        # eigen decomposition of covariance matrix
        eigval, V = np.linalg.eig(cov)
        # stretch matrix
        S = np.diag(1/np.sqrt(eigval))
        # compute mean of each color
        mean = np.mean(A, axis=0)
        # substract the mean from image
        A -= mean
        # compute the transformation matrix
        T = reduce(np.dot, [sigma, V, S, V.T])
        # compute offset 
        offset = mean - np.dot(mean, T)
        # transform the image
        A = np.dot(A, T)
        # add the mean and offset
        A += mean + offset
        # restore original shape
        B = A.reshape(orig_shape)
        # for each color...
        for b in range(3):
            # apply contrast stretching if requested
            if tol:
                # find lower and upper limit for contrast stretching
                low, high = np.percentile(B[:,:,b], 100*tol), np.percentile(B[:,:,b], 100-100*tol)
                B[B<low] = low
                B[B>high] = high
            # ...rescale the color values to 0..255
            B[:,:,b] = 1 * (B[:,:,b] - B[:,:,b].min())/(B[:,:,b].max() - B[:,:,b].min())
        # return it as uint8 (byte) image
        return np.asarray(B,dtype='float32')
           
   def slic0(self,slic0Param,img,plot=True,save=0):
        colorspace = slic0Param['colorSpace_s']
        slic0_img = seg.slic(img,n_segments=slic0Param['n_segments'],max_iter=slic0Param['max_iter'],slic_zero=True)
        if plot:
            if colorspace=='lab':
                img = cv2.cvtColor(img,cv2.COLOR_Lab2BGR)
            plotLibs().plotBoundaries(img,slic0_img,save=save,title='Slic0 segmented Image')
        return slic0_img
            
        

        
        

class plotLibs:
    
    def dispImg(self,img,save=0,title='image'):
        plt.figure(figsize=(10,10))
        plt.title(title)        
        plt.imshow(img)
        plt.show()
        
    def dispGrayImg(self,img,save=0,title='image'):
        plt.figure(figsize=(10,10))
        plt.title(title)        
        plt.imshow(img,cmap='gray')
        plt.show()
    
    def plot_3d(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img)
        r,g,b =  r.flatten(), g.flatten(), b.flatten()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(r, g, b)
        plt.show()
        
    def __segments(self,img,labels,center):
        labels = np.array(labels,dtype='float32')
        labels[labels!=center]= np.nan
        labels[labels==center]=1
        labels[labels==np.nan] = 0
        return img*labels
        
    def dispSegment(self,img,labels,number_of_clusters):
        M,N = number_of_clusters//3,3
        fig, axs = plt.subplots(M,N, figsize=(60, 60), facecolor='w', edgecolor='k',squeeze=True)
        fig.subplots_adjust(hspace = 0.1, wspace=.01)
        axs = axs.ravel()
        for i in range(number_of_clusters):
            segment = self.__segments(img,labels,i)
            axs[i].imshow(segment)
            axs[i].set_title('segment'+str(i))  
        
    def dispKmeansBruteImg(self,reconstructedImg,l):
        M,N = len(reconstructedImg)//2,2
        fig, axs = plt.subplots(M,N, figsize=(60, 60), facecolor='w', edgecolor='k',squeeze=True)
        fig.subplots_adjust(hspace = 0.1, wspace=.01)
        axs = axs.ravel()
        for i in range(len(reconstructedImg)):
            axs[i].imshow(reconstructedImg[i].reshape((512,512,-1)))
            axs[i].set_title('K_'+str(i+l))  
        plt.show()
        
    def plotResponse(self,response):
        fig2, ax2 = plt.subplots(3, 3)
        for axes, res in zip(ax2.ravel(), response):
            axes.imshow(res, cmap=plt.cm.gray)
            axes.set_xticks(())
            axes.set_yticks(())
        ax2[-1, -1].set_visible(False)
        plt.show()
        
    def plotLoss(self,Loss):
        plt.plot(Loss)
        plt.show() 
        
        
        
   
    
    def __display_edges(self,image, g, threshold):
        """Draw edges of a RAG on its image
     
        Returns a modified image with the edges drawn.Edges are drawn in green
        and nodes are drawn in yellow.
     
        Parameters
        ----------
        image : ndarray
            The image to be drawn on.
        g : RAG
            The Region Adjacency Graph.
        threshold : float
            Only edges in `g` below `threshold` are drawn.
     
        Returns:
        out: ndarray
            Image with the edges drawn.
        """
        image = image.copy()
        for edge in g.edges:
            n1, n2 = edge
     
            r1, c1 = map(int, g.node[n1]['centroid'])
            r2, c2 = map(int, g.node[n2]['centroid'])
     
            line  = draw.line(r1, c1, r2, c2)
            circle = draw.circle(r1,c1,2)
     
            if g[n1][n2]['weight'] < threshold :
                image[line] = 0,1,0
            image[circle] = 1,1,0
     
        return image
    
    def plotRAG(self,img,rag,save=0,title='RAG Image'):
        edges_drawn_all = self.__display_edges(img, rag, np.inf)
        self.dispImg(edges_drawn_all,save,title)   
        
    def plotRagwithColorMaps(self,img,labels):
        g = graph.rag_mean_color(img, labels)
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

        ax[0].set_title('RAG drawn with default settings')
        lc = graph.show_rag(labels, g, img, ax=ax[0])
        # specify the fraction of the plot area that will be used to draw the colorbar
        fig.colorbar(lc, fraction=0.03, ax=ax[0])
        
        ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
        lc = graph.show_rag(labels, g, img,
                            img_cmap='gray', edge_cmap='viridis', ax=ax[1])
        fig.colorbar(lc, fraction=0.03, ax=ax[1])
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def plotBoundaries(self,img,labels,save=0,title='Boundaries Image'):
        Boundary_Img = seg.mark_boundaries(img, labels)
        self.dispImg(Boundary_Img,save,title)            
                    
    
class imgLibs:
    
    def __init__(self,imgName=None,clrSpace='rgb'):
        if imgName is not None:
            self.img = np.load(imgName)
            self.imgShape = self.img.shape
        self.clrSpace = clrSpace
    
    def loadImg(self):
        self.img[np.isnan(self.img)] = 0
        if self.clrSpace =='rgb':
            return self.img
        elif self.clrSpace == 'hsv':
            return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        elif self.clrSpace == 'gray':
            return cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        elif self.clrSpace == 'lab':
            return cv2.cvtColor(self.img,cv2.COLOR_BGR2Lab)
        
    def loadImgHelper(self,fpath,colorSpace_s,plot=True):

        img = np.asarray(PIL.Image.open(fpath))
        if colorSpace_s=='lab':
            colorSpace_img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        if colorSpace_s=='hsv':
            colorSpace_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        if colorSpace_s == 'rgb':
            colorSpace_img = img 
        if plot:
            plotLibs().dispImg(img,save=0,title='Input Image')
        return img,colorSpace_img
    
    def plotBoundaries(self,img,labels,save=0,title='Boundaries Image'):
        Boundary_Img = seg.mark_boundaries(img, labels)
        self.dispImg(Boundary_Img,save,title)      
        