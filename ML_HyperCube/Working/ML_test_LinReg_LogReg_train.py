# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:29:19 2020

@author: HyraN
"""


import pandas as pd
from tkinter import Tk,filedialog
import os
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import joblib
import shutil
from timeit import default_timer as timer
import time
from spectral import envi
import numpy as np
import psutil

def hdf2df():
    hdf = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select hdf file to load:", filetypes= (('hdf files', '*.hdf'), ('all files', '*.*)))')))
    print('hdf file selected:', hdf)
    df = pd.read_hdf(hdf)
    return(df) 

def img_dim():
    hdrfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select hdr file:", filetypes= (('hdr files', '*.hdr'), ('all files', '*.*)))')))
    print('hdr file selected: ', hdrfile)  
    header_hdr = envi.read_envi_header(hdrfile)
    width = int(header_hdr['lines'])
    height = int(header_hdr['samples'])
    return(width, height)

def pred2img(pred_array, width, height):
    #width, height = img_dim()
    img = np.reshape(pred_array, (width, height))
    return(img)

#       LINEAR REGRESSION

def LinReg_train(X_train, Y_train, X_test, Y_test):
    from sklearn.linear_model import LinearRegression
    linR = LinearRegression(n_jobs=jobs)
    start = timer()
    linR_model = linR.fit(X_train, Y_train)
    end = timer()
    print('Trained in: ', end-start, 's')
    pred = linR.predict(X_test)
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    mse = mean_squared_error(Y_test, pred)
    r2 = r2_score(Y_test, pred)
    print('mse: ', mse, '\nr2: ', r2)
    return(linR_model, mse, r2)

def LinReg_pred(linR_model, target):
    linR_pred = linR_model.predict(target)
    return(linR_pred)

def LinReg(X_train, Y_train, self_mtrdr, width, height):
    print('Training Linear Regression')
    linR_model, mse, r2 = LinReg_train(X_train, Y_train, X_test, Y_test)
    linR_prediction = LinReg_pred(linR_model, self_mtrdr)
    linR_self_img = pred2img(linR_prediction, width, height)
    return(linR_model, mse, r2, linR_self_img)    

#       LOGISITC REGRESSION

def LogReg_train(X_train, Y_train, X_test, Y_test, maxiter):
    #Model training
    from sklearn.linear_model import LogisticRegression
    logR = LogisticRegression(max_iter=maxiter)
    start = timer()
    logR_model = logR.fit(X_train, Y_train)
    end = timer()
    print('Trained in: ', end-start, 's')
    #accuracy and probability of training
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import log_loss 
    pred = logR.predict(X_test)
    prob = logR.predict_proba(X_test)
    accuracy = accuracy_score(Y_test, pred)
    logloss=log_loss(Y_test, prob)
    print("ACCURACY: "+str(accuracy))
    print("LOG LOSS: "+str(logloss))
    return(logR_model, accuracy, logloss)

def LogReg_pred(logR_model, target):
    pred = logR_model.predict(target)
    prob = logR_model.predict_proba(target)
    return(pred, prob)

def LogReg(X_train, Y_train, self_mtrdr, width, height):
    maxiter=500
    print('Training Logistic Regression')
    logR_model, logR_accuracy, logR_logloss = LogReg_train(X_train, Y_train, X_test, Y_test, maxiter)
    logR_prediction, logR_pred_proba = LogReg_pred(logR_model, self_mtrdr)
    logR_self_img = pred2img(logR_prediction, width, height)
    return(logR_model, logR_prediction, logR_pred_proba, logR_self_img)


def model_valid(model, X_train, Y_train):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, Y_train, cv=10, n_jobs=jobs, verbose=1)
    print("\nValidating training\n")
    start = timer()
    for fold,score in enumerate(scores):
        print("Fold %d score=%.4f" % (fold+1,score))
    end = timer()
    print("\nValidation Accuracy = %.2f" % scores.mean())    
    print('Validated in in: ', end-start, 's')
    return(scores)

def save_models(model, name, savepath):
    os.chdir(savepath)
    joblib.dump(model, name)

def make_folder(PATH):
    savepath = PATH+'/OLINDEX3_LinReg-LonReg_models'
    if os.path.exists(savepath):
           print('Folder exist, removing.')
           shutil.rmtree(savepath)
           time.sleep(1)
           os.mkdir(savepath)
           print('New model folder created')
    else:
           print('Folder not exist, creating.')
           os.mkdir(savepath)
           print('Created new Trained_Models Folder')
    return(savepath)

root = Tk()
root.withdraw()

#select workdir
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

savepath = make_folder(PATH)

print('\nSelect combined hdf file\n')
complete_df = hdf2df()
complete_df_nanzero = complete_df.fillna(0)    
features = complete_df_nanzero.drop('OLINDEX3', axis=1)
#features_nonan = features.fillna(0)
target = complete_df_nanzero['OLINDEX3']

X = features.values
Y = target.values

#get cpu threads
jobs=psutil.cpu_count(logical=False)-4
#jobs = 
    
#create train and test by splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#select the mtrdr that is used for training to evaluate the training
print('Select original mtrdr dataset: ')
self_mtrdr = (hdf2df()).fillna(0)

width, height = img_dim()

#Linear Regression
linR_model, mse, r2, linR_self_img = LinReg(X_train, Y_train, self_mtrdr, width, height)

#save model
linname = 'LinReg.pkl'
save_models(linR_model, linname, savepath)

#validation
lin_scores = model_valid(linR_model, X_train, Y_train)

#Logistic Regression
logR_model, logR_prediction, logR_pred_proba, logR_self_img = LogReg(X_train, Y_train, self_mtrdr, width, height)
logname = 'LogReg.pkl'
save_models(logR_model, logname, savepath)
# log_scores = model_valid(logR_model, X_train, Y_train)

