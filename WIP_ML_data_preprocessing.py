
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:31:22 2020

@author: gnodj
"""


import pandas as pd
import numpy as np
from tkinter import Tk,filedialog
import os
import csv
from sklearn.preprocessing import MinMaxScaler



root = Tk()
root.withdraw()
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

def read_conf():
# reading config file for minerals
    cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Spectral configuration file selected:', cfg)
    with open(cfg, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return(data)

def read_features():
    
    feats = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select index_features file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Features file selected:', feats)
    with open(feats, newline='') as f:
        features = pd.read_csv(feats)
    return(features)

def read_mineral():
    
    miner = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select mineral file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Features file selected:', miner)
    with open(features, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return(data)

def data_norm():
    features = read_features()
    features_norm = features.copy()
    to_norm = features[SPECTRAL_INDEX]
    features_norm = (to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
    print(features_norm)
    return (features_norm)


SPECTRAL_INDEX, MINERAL_LIST = map(list,zip(*read_conf()))


