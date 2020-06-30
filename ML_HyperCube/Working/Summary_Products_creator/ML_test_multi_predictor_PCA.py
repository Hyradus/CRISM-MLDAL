# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:45:15 2020

@author: HyraN
"""
import pandas as pd
from tkinter import Tk,filedialog
import os
import ntpath
import joblib
import shutil
import time
from spectral import envi
import numpy as np
import cv2 as cv
import csv

def read_conf():
# reading config file for minerals
    cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Spectral configuration file selected:', cfg)
    data=[]
    with open(cfg, newline='') as f:
        for row in csv.reader(f):
            data.append(row[0])
    return(data)

def path_leaf(tgtname):
    head, tail = ntpath.split(tgtname)
    return tail or ntpath.basename(head)


def load_model():
    pkl = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select pkl file to load:", filetypes= (('pkl files', '*.pkl'), ('all files', '*.*)))')))
    print('png file selected:', pkl)
    model = joblib.load(pkl)
    modname = path_leaf(pkl)
    return(model, modname)

def load_original_img():
    png = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select png file to load:", filetypes= (('png files', '*.png'), ('all files', '*.*)))')))
    print('png file selected:', png)
    img = cv.imread(png)
    img_slice = img[:,:,1].astype('float64')
    img_slice_fill = np.where(img_slice ==255, 0, img_slice)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(img_slice_fill)
    img_scaled = scaler.transform(img_slice_fill)
    img_scaled_mask = np.ma.masked_where(img_scaled == 0, img_scaled)
    return(img_scaled, img_scaled_mask, img_slice, png)

def hdf2df():
    hdf = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select hdf file to load:", filetypes= (('hdf files', '*.hdf'), ('all files', '*.*)))')))
    print('hdf file selected:', hdf)
    df = pd.read_hdf(hdf)
    return(df) 

def make_folder(PATH, modname):
    savepath = PATH+'/'+ str(modname)+'_predictions'
    if os.path.exists(savepath):
           print('Folder exist, removing.')
           shutil.rmtree(savepath)
           time.sleep(1)
           os.mkdir(savepath)
           print('New model folder created')
    else:
           print('Folder not exist, creating.')
           os.mkdir(savepath)
           print('Created new predictions folder')
    return(savepath)



def img_dim():
    hdrfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Pleaaase select hdr file:", filetypes= (('hdr files', '*.hdr'), ('all files', '*.*)))')))
    print('hdr file selected: ', hdrfile)  
    header_hdr = envi.read_envi_header(hdrfile)
    width = int(header_hdr['lines'])
    height = int(header_hdr['samples'])
    return(width, height)

# def pred2img(pred_array):
#     width, height = img_dim()
#     img = np.reshape(pred_array, (width, height))
#     return(img)

def pred2img(pred_array):
    width, height = img_dim()
    array = np.empty((width, height,len(pred_array[1])))
    for i in range(pred_array.shape[1]):
        array[:,:,i] = np.reshape(pred_array[:,i], (width, height))
    return(array)

def img_scale_mask(img):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    #replace prediction outlayer with 0
    img_tozero = np.where((img <0.05), 0, img)
    scaler.fit(img_tozero)
    img_scaled = scaler.transform(img_tozero)
    img_scaled_mask = np.ma.masked_where(img_scaled == 0, img_scaled)
    return(img_tozero, img_scaled, img_scaled_mask)

def compute_ssim(or_img, pred_img):
    from skimage.metrics import structural_similarity as ssim
    computed_ssim = ssim(or_img, pred_img)
    return(computed_ssim)

def tgt_PCA(df):
    from sklearn.decomposition import PCA
    # Make an instance of the Model
    pca = PCA(n_components=5)
    pca.fit(df)
    df_pca= pca.transform(df)
    return(df_pca)

def predictor(model, tgt_pred, SPECTRAL_INDEX):
       
    pred = model.predict(tgt_pred)
    #conversion of the prediction back to 2d image
    pred_imgs = pred2img(pred).astype('float64')   
    return(pred, pred_imgs)

def make_images(pred, pred_imgs, modname):
    
    images = []
    savepath = make_folder(PATH, modname) 
    os.chdir(savepath)
    for i in range(len(pred[1])):
                 images.append(pred_imgs[:,:,i]*255)
                 filename = SPECTRAL_INDEX[i] + '_predicted.png'
                 print('Saving image: ', i)
                 cv.imwrite(filename,pred_imgs[:,:,i])
    return(images)    
    
def main(model, modname, tgt_pred, SPECTRAL_INDEX):
    if CHOICE_PCA =='yes':
        tgt_pred = tgt_PCA(tgt_pred)
        
    pred, pred_imgs = predictor(model, tgt_pred, SPECTRAL_INDEX)
    images = make_images(pred, pred_imgs, modname)
    return(pred, images)
    


if __name__ == "__main__":
    
    root = Tk()
    root.withdraw()
    
    #select workdir
    PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
    print('Working folder:', PATH)
    #load summary products original png to compare
    #Olindex3_scaled, Olindex3_scaled_mask, Olindex3_slice, Original_png = load_original_img()
    
    #load the mtrdr hdf file to predict
    print('Select MTRDR to predict')
    tgt_pred = (hdf2df().fillna(0))
    
    CHOICE_PCA = 'yes'
    
    #read spectral index list
    print('Select spectral indexes csv')
    SPECTRAL_INDEX = read_conf()
    
    
    model, modname = load_model()
    
    pred, pred_imgs = main(model, modname, tgt_pred, SPECTRAL_INDEX)
    

    

    
