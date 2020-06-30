#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:32:27 2020

@author: gnodj
"""

import numpy as np
from tkinter import Tk,filedialog
import os
import glob
import ntpath
import time
import cv2 as cv
import shutil
import pandas as pd


def image_list(path, ext):
    #ext = '*'+imagetype+'.png'
    os.chdir(path)
    imgs = [i for i in glob.glob(ext, recursive=True)]
    imgs.sort()
    return(imgs)

def list_comb(path, imagetypes):
    images = []
    for items in imagetypes:
        imagetype = items
        images.append(image_list(path, imagetype))
    images.sort()
    return(images)

def compute_ssim(or_img, pred_img):
    from skimage.metrics import structural_similarity as ssim
    computed_ssim = ssim(or_img, pred_img)
    return(computed_ssim)

root = Tk()
root.withdraw()
path = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the folder with the original and predicted png:")
print('Working folder:', path)


images_ori = image_list(path, '*original.png')
images_pred = (image_list(path, '*predicted.png'))

ssims = []
for i in range(len(images_ori)):
    # print(images_ori[i])
    # print(images_pred[i])
    img = cv.imread(images_ori[i])[:,:,2]
    img_tozero = np.where((img ==255), 0, img)    
    ssim = compute_ssim(img_tozero, (cv.imread(images_pred[i]))[:,:,2])
    # print('SSIM for ', images_ori[i], ' is: ', ssim)
    ssims.append(ssim)
    
ssim_results=list(zip(image_list(path, '*predicted.png'),ssims))

df = pd.DataFrame(ssim_results, columns=('Spectral index', 'ssim'))
savename = path+'/ssim_results.csv'
df.to_csv(savename)
