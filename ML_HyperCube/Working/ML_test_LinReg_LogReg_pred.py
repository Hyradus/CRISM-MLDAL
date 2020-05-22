# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:45:15 2020

@author: HyraN
"""
import pandas as pd
from tkinter import Tk,filedialog
import os
# from sklearn.preprocessing import normalize
# from sklearn.model_selection import train_test_split
import joblib
# import shutil
# from timeit import default_timer as timer
# import time
from spectral import envi
import numpy as np
import cv2 as cv



#Model 067
#LinR_model = joblib.load('D:/Mars_CRISM_DATA/Datasets/MTRDR/2008_067/combined_1_1_dataframe/OLINDEX3_models/OLINDEX3_linReg.pkl')
#Model 360
#LinR_model = joblib.load('D:/Mars_CRISM_DATA/Datasets/MTRDR/2007_360/combined_1_1_dataframe_classes/OLINDEX3_models/OLINDEX3_linReg.pkl')
#combined 
#LinR_model = joblib.load('D:/Mars_CRISM_DATA/Datasets/MTRDR/Combined_test/combined_6_1_dataframe_nan/OLINDEX3_models/OLINDEX3_linReg.pkl')
def load_model():
    pkl = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select pkl file to load:", filetypes= (('pkl files', '*.pkl'), ('all files', '*.*)))')))
    print('png file selected:', pkl)
    model = joblib.load(pkl)
    return(model)

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

def img_dim():
    hdrfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Pleaaase select hdr file:", filetypes= (('hdr files', '*.hdr'), ('all files', '*.*)))')))
    print('hdr file selected: ', hdrfile)  
    header_hdr = envi.read_envi_header(hdrfile)
    width = int(header_hdr['lines'])
    height = int(header_hdr['samples'])
    return(width, height)

def pred2img(pred_array):
    width, height = img_dim()
    img = np.reshape(pred_array, (width, height))
    return(img)

def predict(model, target):
    pred = model.predict(target)
    return(pred)

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

root = Tk()
root.withdraw()

#load summary products original png to compare
Olindex3_scaled, Olindex3_scaled_mask, Olindex3_slice, Original_png = load_original_img()

#load the mtrdr hdf file to predict
print('Select MTRDR to predict')
tgt_pred = (hdf2df().fillna(0))

#
load linear regression model
LinR_model = load_model()

###### LINEAR REGRESSION prediction ######

LinReg_pred = predict(LinR_model, tgt_pred)
#conversion of the prediction back to 2d image
LinReg_pred_img = pred2img(LinReg_pred).astype('float64')
#scale the image in range 0-1
#LinReg_scaled_mask = img_scale_mask(LinReg_tgt_img).astype('float64')

# create all steps variable for debugging
pred_lin_tozero, LinReg_scaled, LinReg_scaled_mask = img_scale_mask(LinReg_pred_img)

#compute structural similarity index between original summary products and predictions. 
LinReg_ssim = compute_ssim(Olindex3_scaled, LinReg_scaled)

# #save the images
# cv.imwrite('Olindex3_original.png', Olindex3_scaled_mask*255)
# cv.imwrite('Olindex3_LinReg_prediction_mask.png', LinReg_scaled_mask*255)

##### LOGISTIC REGRESSION prediction ######

# LogR_model = load_model()

# LogReg_pred = predict(LogR_model, tgt_pred)

# LogReg_pred_img = pred2img(LogReg_pred).astype('float64')

# pred_log_tozero, LogReg_scaled, LogReg_scaled_mask = img_scale_mask(LogReg_pred_img)

# cv.imwrite('Olindex3_LogReg_prediction_mask.png', LogReg_scaled_mask*255)

