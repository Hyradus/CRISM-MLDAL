"""
Created on Tue Mar 10 11:31:22 2020

@author: Hyradus
"""
import pandas as pd
import numpy as np
from tkinter import Tk,filedialog
import os
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODEL = 'tree'

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
    with open(miner, newline='') as f:
        reader = csv.reader(f)
        mineral = pd.read_csv(miner)
    return(mineral)
    #     mineral = list(reader)
    # return(mineral)

def data_norm():
    features = read_features()
    features_norm = features.copy()
    to_norm = features[SPECTRAL_INDEX]
    features_norm = (to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
    print(features_norm)
    return (features_norm)

def normalizer(X_train, X_test):
    norm = Normalizer()
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.fit_transform(X_test)
    return (X_train_norm, X_test_norm)

def encoderY(Y_train, Y_test):
    le = LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    Y_test_enc = le.transform(Y_test)
    return (Y_train_enc, Y_test_enc)

def logReg(X_train_norm, Y_train_enc):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train_norm, Y_train_enc)
    return (lr)

def decTrees(X_train, Y_train):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, Y_train)
    return(tree)

def ranForest(X_train, Y_train):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=False)
    forest.fit(X_train, Y_train)
    return(forest)

def pred(X_train, Y_train, MODEL):
    Y_pred_train = MODEL.predict(X_train)
    Y_pred_test = MODEL.predict(X_test)
    return(Y_pred_train, Y_pred_test)

def check(Y_train, Y_pred_train, Y_test, Y_pred_test):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import log_loss
    accuracy_train = accuracy_score(Y_train, Y_pred_train)
    accuracy_test = accuracy_score(Y_test, Y_pred_test)
    return(accuracy_train, accuracy_test)


SPECTRAL_INDEX, MINERAL_LIST = map(list,zip(*read_conf()))

features = read_features()
mineral = read_mineral()

featnonan = features.fillna(0)

X = featnonan.values

Y = mineral.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

#Normalizing X
X_train_norm, Xftst_norm = normalizer(X_train, X_test)
#Encoding Y
Y_train_enc, Y_test_enc = encoderY(Y_train, Y_test)


if MODEL == 'tree':
    mod = decTrees(X_train, Y_train)
elif MODEL == 'forest':
    mod = ranForest(X_train, Y_train)

Y_pred_train = mod.predict(X_train)
Y_pred_test = mod.predict(X_test)
scores = check(Y_train, Y_pred_train, Y_test, Y_pred_test)
print("ACCURACY FOR MODEL", MODEL,": TRAIN=%.8f TEST=%.8f" % (scores[0], scores[1]))

feature_imp = pd.Series(mod.feature_importances_,index=features.columns).sort_values(ascending=False)
feature_imp

# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# # Creating a bar plot
# sns.barplot(x=feature_imp, y=feature_imp.index)
# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()



def read_pred_target():
    tgt = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be tested:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Submitted file selected:', tgt)
    with open(tgt, newline='') as f:
        #reader = csv.reader(f)
        pred_target = pd.read_csv(tgt)
    return(pred_target)    

pred_target = read_pred_target()

pred_tree = mod.predict(pred_target)
