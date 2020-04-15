"""
Created on Tue Mar 10 11:31:22 2020

@author: Hyradus
"""
import pandas as pd
import numpy as np
from tkinter import Tk,filedialog
import os
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import shutil
from timeit import default_timer as timer
from datetime import datetime
import time

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

def read_target():
    miner = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be trained:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Features file selected:', miner)
    with open(miner, newline='') as f:
        #reader = csv.reader(f)
        mineral = pd.read_csv(miner)
    return(mineral)

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


def DecTree(X_train, Y_train):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=6)
    tree.fit(X_train, Y_train)
    return(tree)

def RanForest(X_train, Y_train):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=False)
    forest.fit(X_train, Y_train.ravel())
    return(forest)

def train(X_train, Y_train):
    start = timer()
    trees = DecTree(X_train, Y_train)
    end = timer()
    print('\nDecisional Trees Model is trained in: ', end-start, 'seconds')
    start = timer()
    forest = RanForest(X_train, Y_train)  
    end = timer()
    print('\nRandom Forest Model is trained in: ', end-start, 'seconds')
    return (trees, forest)
    
def pred(model, X_train, Y_train):
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    return(Y_pred_train, Y_pred_test)

def evaluation(trained_results, X_test, Y_test):
    # training evaluation
    tree = trained_results[0]
    forest = trained_results[1]  
    # Calculate the accuracy and predictions
    score_tree = tree.score(X_test, Y_test)
    score_forest = forest.score(X_test, Y_test)
    print("Test score of Decisional Trees model is: {0:.2f} %".format(100 * score_tree))    
    print("Test score of Random Forest model is: {0:.2f} %".format(100 * score_forest))
    featImpDf = pd.DataFrame()
    featImpDf['Decisional Trees'] = pd.Series(tree.feature_importances_,index=featnonan.columns).sort_values(ascending=False)
    featImpDf["Random Forest"] = pd.Series(forest.feature_importances_,index=featnonan.columns).sort_values(ascending=False)    
    print("\nFeatures importances for all models are:\n", featImpDf)
    return(featImpDf, score_tree, score_forest)

def main(featnonan, target, tgt):    
    Y = target[target.columns[tgt]].values
    X = featnonan.values
    #create train and test by splitting the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    
    #Trainig results
    trained_results = train(X_train, Y_train)
    
    #Evaluate feature score
    eval_results = evaluation(trained_results, X_test, Y_test)
    
    #Saving models    
    SAVENAME = [(target.columns[tgt]+'_Decisional_Trees.pkl'), target.columns[tgt]+'_Random_Forest.pkl']    
    ## removing older models and writing new ones
    savedir = target.columns[tgt]+'_models'
    # if os.path.exists(savedir):
    #     print('Folder exists, removing...')
    #             #os.chdir(PATH)
    #     shutil.rmtree(savedir)
    #     #os.mkdir(savedir+'/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #     print("Removing old Trained_Models folder...")
    #     time.sleep(0.000001)
    #     os.mkdir(PATH+'/'+savedir)
    #     print('Created new Trained_Models Folder')
    # else:
    #     #os.makedirs(PATH+'/'+savedir+'/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #     os.mkdir(PATH+'/'+savedir)
    #     print('Created new Trained_Models Folder')
    if os.path.exists(PATH+'/'+savedir):
        print('Folder exist, removing.')
        shutil.rmtree(PATH+'/'+savedir)
        time.sleep(1)
        os.mkdir(PATH+'/'+savedir)
        print('New model folder created')
    else:
        print('Folder not exist, creating.')
        os.mkdir(PATH+'/'+savedir)
        print('Created new Trained_Models Folder')
    
    
    # saving models to files  
    joblib.dump(trained_results[0], PATH+'/'+savedir+'/'+SAVENAME[0])
    print('\nSaved ', SAVENAME[0], "in ", savedir)
    joblib.dump(trained_results[1], PATH+'/'+savedir+'/'+SAVENAME[1])
    print('\nSaved ', SAVENAME[1], "in ", savedir)
    
    
    #Plot feature evaluation
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(18, 9))
    ax = plt.subplot(121)
    ax1 = sns.barplot(x=eval_results[0]['Decisional Trees'].values, y=eval_results[0].index, order=eval_results[0].index)
    plt.xlabel('Feature score')
    plt.ylabel('Features', multialignment='center')
    plt.title("Decisional Trees Model")
    plt.grid(True)
    
    plt.subplot(122, sharey=ax1, sharex=ax1)
    ax2 = sns.barplot(x=eval_results[0]['Random Forest'].values, y=eval_results[0].index, order=eval_results[0].index)
    plt.xlabel('Feature score')
    plt.ylabel('Features', multialignment='center')
    plt.grid(True)
    plt.title("Random Forest Model")
    plt.suptitle(target.columns[tgt], fontsize=16)
    #plt.subplots_adjust(bottom=0.5, top=0.5)
    figname =PATH+ '/'+savedir+'/'+target.columns[tgt]+'_Feature_importance_graphs.png'
    plt.savefig(figname)
    plt.show()
    
    # confusion matrix and classification report
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import classification_report
    #DECISIONAL TREES
    y_pred_tree=trained_results[0].predict(X_test)
    cm_tree = confusion_matrix(Y_test, y_pred_tree)
    print('Decisional tree confusion Matrix\n', cm_tree)
    cr_tree = classification_report(Y_test, y_pred_tree)
    print('\nDecisional tree classification report\n', cr_tree)
    
    #RANDOM FOREST
    y_pred_forest=trained_results[1].predict(X_test)
    cm_forest = confusion_matrix(Y_test, y_pred_forest)
    print('Random Forest confusion Matrix\n', cm_forest)
    cr_forest = classification_report(Y_test, y_pred_forest)
    print('\nRandom Forest classification report\n', cr_forest)
    
    #export graphics for decision tree
    from sklearn.tree import export_graphviz
    os.chdir(PATH+'/'+savedir+'/')
    dotfile = open("tree.dot", 'w')
    export_graphviz(trained_results[0], out_file = dotfile)
    dotfile.close()
    os.system('dot -Tpng tree.dot -o tree.png')

SPECTRAL_INDEX, MINERAL_LIST = map(list,zip(*read_conf()))

#assigning dataset
features = read_features()
featnonan = features.fillna(0)

target = read_target()
#target = target.fillna(0)

#tgt= 4 #to select which mineral phase to train

if __name__ == "__main__":
    tgt = 0
    start = timer()
    for tgt in range(len(features.columns)):
        print('Training ', features.columns[tgt], ' models..')
        main(featnonan, target, tgt)
        
    end = timer()
    print('All models trained in ', end-start, ' seconds')