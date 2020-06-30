#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:13:37 2020

Isolation forests tests

@author: gnodj
"""


import h2o
import pandas as pd
from tkinter import Tk,filedialog
import os
from spectral import envi
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def hdf2df():
    hdf = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),
                                     title="Please select select hdf file to load:",
                                     filetypes= (('hdf files', '*.hdf'),
                                                 ('all files', '*.*)))')))
    print('hdf file selected:', hdf)
    df = pd.read_hdf(hdf)
    return(df, hdf)

def img_dim():
    hdrfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select hdr file:", filetypes= (('hdr files', '*.hdr'), ('all files', '*.*)))')))
    print('hdr file selected: ', hdrfile)  
    header_hdr = envi.read_envi_header(hdrfile)
    width = int(header_hdr['lines'])
    height = int(header_hdr['samples'])
    return(width, height)


def pred2img(pred_array):
    width, height = img_dim()
    array = np.empty((width, height,0))
    for i in range(pred_array.shape[1]):
        array[:,:,i] = np.reshape(pred_array[:,i], (width, height))
    return(array)

def prepData():
    df_ori, hdf_name= hdf2df()
    df_nonan = df_ori.dropna(how='all')
    df_clean = df_nonan.fillna(df_ori.mean())
    
    hdf_name=hdf_name.split('.')[0]
    csv_name = hdf_name +'.csv'
   # df_clean.to_csv(csv_name)
    return(df_ori, df_clean, csv_name)


def isoForest(dataframe):
    df_h2o = h2o.import_file(dataframe)
    
    isoforest = h2o.estimators.H2OIsolationForestEstimator(
        ntrees=ntrees, seed=seed)
    isoforest.train(x=df_h2o.col_names, training_frame=df_h2o)
    predictions = isoforest.predict(df_h2o)
    #converting to pandas dataframe    
    df_predictions = predictions.as_data_frame()
    #visualizion mean_length and anomaly score
    df_predictions.hist()
    #grouping mean_length and anomaly score
    mean_groups= df_predictions.round(1).groupby('mean_length')['predict'].sum()
    pred_groups = df_predictions.round(1).groupby('predict').sum()
    
    return(df_predictions, mean_groups, pred_groups)

def anomalies(df_merged, mean_length) :
    df_anomalies = df_merged.copy()
    #remove values above the threshold
    df_anomalies[df_anomalies['mean_length'] > float(mean_length)] = np.nan
    df_anomalies= df_anomalies.dropna()
    return(df_anomalies)

def img_anomalies(df_ori, df_anomalies):
    
    width, height = img_dim()
    df_or_anoma = pd.concat([df_ori, df_anomalies['mean_length']], axis=1)
    #df_or_anoma.drop([0], axis=0, inplace=True)
    anomalies_img = (np.reshape(df_or_anoma['mean_length'].values, (width, height))*255/10)
    cv.imwrite(PATH+'/anomalies.png', anomalies_img)
    return(anomalies_img)

def plot_anomalies(df_ori, df_anomalies):
    index = df_anomalies.index
    # df.columns=df.columns.values.astype('float64')
    figure = plt.figure(figsize=(12,5))
    columns = df_ori.columns.values.astype('float64')



    ############################## plots
    xticks = np.arange(0, columns.max(), 25)
    plt.xticks(xticks, rotation=45)
    plt.tight_layout()
    for i in range(len(index)):
        row=df_ori.iloc[index[i]]
        plt.plot(row,color='red')
        plt.tight_layout()   
    
def syntetic_anomalies(df):
##### TEST SYNTETIC ANOMALIES 
    import random
    rows = random.sample(range(0,100000),10000)
    for rows in rows:
        anomaly=np.random.uniform(size=489)
        # print(anomaly)
        df.iloc[rows]=anomaly    
    
def main(seed, ntrees):
    #get dataframes
    df_ori, df_clean, csv_name = prepData()
    
    #isolation forest
    h2o_predictions, mean_groups, pred_groups = isoForest(csv_name)
    df_merged = pd.concat([df_ori, h2o_predictions], axis=1)
    print(mean_groups)
    #ask for mean_length threshold for defining anomalies limit
    mean_length = float(input("Please input mean_length# df_anomalies[df_anomalies['mean_length'] <] = np.nan threshold: "))
    try:
        mean = float(mean_length)
        print("Mean_length selected is: ", mean_length)
    except ValueError:
        print("This is not a valid number or its outside the range of mean_length values")
    
    df_anomalies=anomalies(df_merged, mean_length)    
   
    anomalies_img=img_anomalies(df_ori, df_anomalies)
    plot_anomalies(df_ori, df_anomalies)
    return(df_ori, df_clean, h2o_predictions, df_anomalies)
    

    


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    
    #select workdir
    PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),
                               title="Please select the working folder:")
    print('Working folder:', PATH)
    os.chdir(PATH)
    
    #initialize h2o cluster
    h2o.init()
    # h2o.cluster().shutdown(prompt=True) 

    
    
    # define ntrees
    seed = 12345
    ntrees = 100
    
    df_ori, df_clean, h2o_predictions, df_anomalies  = main(seed, ntrees)
    



