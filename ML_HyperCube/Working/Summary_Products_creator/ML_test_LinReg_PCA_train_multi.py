#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:34:04 2020

@author: gnodj
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

###### OPTIONS EDIT BEFORE RUN ######
    
#get cpu threads WARNING use psutil only on high end systems with lot of ram
JOBS = psutil.cpu_count(logical=True)
# JOBS = 8

#specify if use PCA and number of components
USE_PCA = 'yes'
N_PCA = 5
# specify spectral INDEXES to train - leave empty to use all INDEXES
INDEXES = [] #e.g. ['OLINDEX3', 'BDI1000VIS', 'HCPINDEX2', 'LCPINDEX2']

###### END OPTIONS EDIT BEFORE RUN ######

####### GLOBAL FUNCTIONS #######

def hdf2df():
    hdf = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),
                                     title="Please select select hdf file to load:",
                                     filetypes= (('hdf files', '*.hdf'),
                                                 ('all files', '*.*)))')))
    print('hdf file selected:', hdf)
    df = pd.read_hdf(hdf)
    return(df) 

def make_folder(PATH):
    savepath = PATH+'/multi_LinRegr_models'
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



def get_data():
    print('\nSelect combined hdf file to be trained\n')
    combined_df = hdf2df()
    
    
    # if CHOICE_DF == 'full':    
    train_df = combined_df.drop((combined_df.columns[489:527]), axis=1)
    
    if not INDEXES:
        tgt_df = combined_df.drop((combined_df.columns[0:489]), axis=1)
    else:
        
        tgt_df = combined_df[INDEXES]   
    
    print('\n Dropping combined_df')        
    combined_df = None
 
    
    X = train_df.values
    Y = tgt_df.values       

    #create train and test by splitting the dataset
    print('\n Splitting dataset')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=1)
    
    if USE_PCA == 'yes':
        print('\n Computing PCA')
        X_train, X_test, pca_fit = feat_PCA(X_train, X_test)
    else:
        pca_fit = np.nan
    
    print('\n Creating Xs and Ys')
    Xs = (X_train, X_test)
    Ys = (Y_train, Y_test)
    
    return(X_train, X_test, Y_train, Y_test, Xs, Ys, pca_fit)

def feat_PCA(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_PCA)
    pca_fit = pca.fit(X_train)
    X_train_PCA = pca_fit.transform(X_train)
    X_test_PCA = pca_fit.transform(X_test)
    return(X_train_PCA, X_test_PCA, pca_fit)

def save_models(model, model_name, savepath):
    save_name = str(os.path.split(os.path.dirname(PATH))[1]) + '_'+model_name +'.pkl'
    print('\nSaving model\n')
    os.chdir(savepath)
    joblib.dump(model, save_name)#, compress=3)
    
    
def overfit_eval(model, savename, Xs, Ys):
    from sklearn.metrics import mean_squared_error, r2_score
    Y_pred_train = model.predict(Xs[0])
    Y_pred_test = model.predict(Xs[1])
    mse_train = mean_squared_error(Ys[0], Y_pred_train)
    mse_test = mean_squared_error(Ys[1], Y_pred_test)
    r2_train = r2_score(Ys[0], Y_pred_train)
    r2_test = r2_score(Ys[1], Y_pred_test)    
    print("Train set:  MSE="+str(mse_train)+" R2="+str(r2_train))
    print("Test set:  MSE="+str(mse_test)+" R2="+str(r2_test))
        # overfitting evaluation
             
    results = [['mse train', mse_train],
               ['mse test', mse_test],
               ['r2 train', r2_train],
               ['r2 test', r2_test]]
    
    results_df = pd.DataFrame(results, columns=['Variable','Score'])
    results_df.to_csv(savename, header=True)
    return(results)



def cross_val(model, savename, X_train, Y_train, valjobs):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, Y_train,
                             cv=10, n_jobs=valjobs,
                             verbose=1)
    print("\nValidating training\n")
    
    start = timer()
    for fold,score in enumerate(scores):
        print("Fold %d score=%.4f" % (fold+1,score))
    end = timer()
    print("\nValidation Accuracy = %.2f" % scores.mean())    
    print('Validated in in: ', end-start, 's')
    return (scores)

def evaluate(model, model_name, Xs, Ys, valjobs):
    pathname = str(os.path.split(os.path.dirname(PATH))[1])
    savename = savepath + '/'+pathname+'_'+model_name+"_Results.csv"
    print('\nComputing MSE and r2 for train and test sets')
    results = overfit_eval(model, savename, Xs, Ys)
    print('\nComputing cross validation')
    # scores = cross_val(model, savename, Xs[0], Ys[0], valjobs)
    scores= 0
    return(results, scores)

def predict(model, target):
    pred = model.predict(target)
    # prob = tree_model.predict_proba(target)
    return(pred)


####### END GLOBAL FUNCTIONS #######

####### LINEAR REGRESSION #######

def LinReg_train(X_train, Y_train):
    from sklearn.linear_model import LinearRegression
    linR = LinearRegression(n_jobs=JOBS)
    start = timer()
    linR_model = linR.fit(X_train, Y_train)
    end = timer()
    print('Trained in: ', end-start, 's')
    return(linR_model)

def LinReg(X_train, X_test, Y_train, Y_test, Xs, Ys):
    LinReg_model = LinReg_train(X_train, Y_train)
    model_name = 'LinRegestReg'
    save_models(LinReg_model, model_name, savepath)
    results, scores  = evaluate(LinReg_model, model_name, Xs, Ys, valjobs=1)

####### END OF LINEAR REGRESSION #######



def main(X_train, X_test, Y_train, Y_test, Xs, Ys):    
      
    LinReg(X_train, X_test, Y_train, Y_test, Xs, Ys)


if __name__ == "__main__":
    
    root = Tk()
    root.withdraw()
    
    #select workdir
    PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),
                                   title="Please select the working folder:")
    print('Working folder:', PATH)
    
    savepath = make_folder(PATH)
    
    X_train, X_test, Y_train, Y_test, Xs, Ys, pca_fit = get_data()
    
    main(X_train, X_test, Y_train, Y_test, Xs, Ys)
