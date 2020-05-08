# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:29:19 2020

@author: HyraN
"""


import pandas as pd
from tkinter import Tk,filedialog
import os
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import joblib
import shutil
from timeit import default_timer as timer
import time


def hdf2df():
    hdf = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select hdf file to load:", filetypes= (('hdf files', '*.hdf'), ('all files', '*.*)))')))
    print('hdf file selected:', hdf)
    df = pd.read_hdf(hdf)
    return(df) 


def KKN(X_train, Y_train):
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(X_train, Y_train)
    return(kn)

# def svm(X_train, Y_train):
    

root = Tk()
root.withdraw()

#select workdir
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

print('\nSelect combined hdf file\n')
complete_df = hdf2df()
    
features = complete_df.drop('OLINDEX3', axis=1)
features_nonan = features.fillna(0)
target = complete_df['OLINDEX3']

X = features_nonan.values
Y = target.values
    
    #normalize train dataset
    #create train and test by splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# knmodel = KKN(X_train, Y_train)   

# y_pred_1 = knmodel.predict(X_test) 
    

from sklearn.svm import SVC

svc = SVC(kernel="rbf",probability=True) #equivale alla classe LinearSVC
svcmodel = svc.fit(X_train, Y_train)
print("ACCURACY: Train=%.4f Test=%.4f" % (svc.score(X_train, Y_train), svc.score(X_test,Y_test)))
plot_bounds((X_train, X_test),(Y_train, Y_test),svc)