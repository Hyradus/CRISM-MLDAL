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

def DecTree_train(X_train, Y_train):
    #Model training
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=6)
    start = timer()
    tree_model = tree.fit(X_train, Y_train)
    end = timer()
    print('Trained in: ', end-start, 's')
    #accuracy and probability of training
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import log_loss 
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    accuracy = accuracy_score(Y_test, pred)
    logloss=log_loss(Y_test, prob)
    print("ACCURACY: "+str(accuracy))
    print("LOG LOSS: "+str(logloss))
    return(tree_model, accuracy, logloss)

def DecTree_pred(tree_model, target):
    pred = tree_model.predict(target)
    prob = tree_model.predict_proba(target)
    return(pred, prob)

def DecTree(X_train, Y_train, self_mtrdr, width, height):
    print('Training Decision Tree')
    tree_model, tree_accuracy, tree_logloss = DecTree_train(X_train, Y_train)
    tree_prediction, tree_pred_proba = DecTree_pred(tree_model, self_mtrdr)
    tree_self_img = pred2img(tree_prediction, width, height)
    return (tree_model, tree_prediction, tree_pred_proba, tree_self_img)


def RanForest_train(X_train, Y_train):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=False, n_jobs=jobs)
    start = timer()
    forest_model = forest.fit(X_train, Y_train)
    end = timer()
    print('Trained in: ', end-start, 's')
    #accuracy and probability of training
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import log_loss 
    pred = forest.predict(X_test)
    prob = forest.predict_proba(X_test)
    accuracy = accuracy_score(Y_test, pred)
    logloss=log_loss(Y_test, prob)
    print("ACCURACY: "+str(accuracy))
    print("LOG LOSS: "+str(logloss))
    return(forest_model, accuracy, logloss)

def RanForest_pred(forest_model, target):
    pred = forest_model.predict(target)
    prob = forest_model.predict_proba(target)
    return(pred, prob)

def RanForest(X_train, Y_train, self_mtrdr, width, height):
    print('Training Decision forest')
    forest_model, forest_accuracy, forest_logloss = RanForest_train(X_train, Y_train)
    forest_prediction, forest_pred_proba = RanForest_pred(forest_model, self_mtrdr)
    forest_self_img = pred2img(forest_prediction, width, height)
    return (forest_model, forest_prediction, forest_pred_proba, forest_self_img)


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
    savepath = PATH+'/OLINDEX3_DecTree_RanForest_models'
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
    
features = complete_df.drop('OLINDEX3', axis=1)
features_nonan = features.fillna(0)
target = complete_df['OLINDEX3']

X = features_nonan.values
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

## Decision Tree

tree_model, tree_prediction, tree_pred_proba, tree_self_img = DecTree(X_train, Y_train, self_mtrdr, width, height)
treename = 'Tree.pkl'
save_models(tree_model, treename, savepath)
tree_scores = model_valid(tree_model, X_train, Y_train)

# Random Forest

forest_model, forest_prediction, forest_pred_proba, forest_self_img = RanForest(X_train, Y_train, self_mtrdr, width, height)
forestname = 'Forest.pkl'
save_models(forest_model, forestname, savepath)
forest_scores = model_valid(forest_model, X_train, Y_train)