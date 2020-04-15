"""
Created on Fri Apr 10 11:37:47 2020

@author: HyraN
"""

import pandas as pd
import numpy as np
from tkinter import Tk,filedialog
import os
import csv
import joblib

root = Tk()
root.withdraw()

    
def read_pred_target():
    tgt = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be tested:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Submitted file selected:', tgt)
    with open(tgt, newline='') as f:
        #reader = csv.reader(f)
        pred_target = pd.read_csv(tgt)
    return(pred_target)    

def read_model():
    modelfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select model to load:", filetypes= (('pkl files', '*.pkl'), ('all files', '*.*)))')))
    print('Submitted file selected:', modelfile)
    model = joblib.load(modelfile)
    return(model)
    
pred_target = read_pred_target()
tree_model = read_model()
forest_model = read_model()


pred_tree = tree_model.predict(pred_target)
pred_forest = forest_model.predict(pred_target)

