# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:53:42 2020

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

def DecTree(X_train, Y_train):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=6)
    tree.fit(X_train, Y_train)
    return(tree)

def RanForest(X_train, Y_train):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=False)
    forest.fit(X_train, Y_train)
    return(forest)



def train(X_train, Y_train):
    start = timer()
    trees = DecTree(X_train, Y_train)
    end = timer()
    print('Training Decision Tree model...')
    print('\nDecision Trees Model trained in: ', end-start, 'seconds')
    start = timer()
    print('Training Random Forest model...')
    forest = RanForest(X_train, Y_train)  
    end = timer()
    print('\nRandom Forest Model trained in: ', end-start, 'seconds')
    return (trees, forest)

def evaluation(trained_results, X_test, Y_test):
    # training evaluation
    tree = trained_results[0]
    forest = trained_results[1]  
    # Calculate the accuracy and predictions
    score_tree = tree.score(X_test, Y_test)
    score_forest = forest.score(X_test, Y_test)
    print("Test score of Decision Trees model is: {0:.2f} %".format(100 * score_tree))    
    print("Test score of Random Forest model is: {0:.2f} %".format(100 * score_forest))
    featImpDf_tree = pd.DataFrame()
    featImpDf_forest = pd.DataFrame()
    featImpDf_tree['Decision Trees'] = pd.Series(tree.feature_importances_,index=features.columns).sort_values(ascending=False)
    featImpDf_forest["Random Forest"] = pd.Series(forest.feature_importances_,index=features.columns).sort_values(ascending=False)    
    print("\nFeatures importances for tree model are:\n", featImpDf_tree)
    print("\nFeatures importances for forest model are:\n", featImpDf_forest)
    return(featImpDf_tree, featImpDf_forest, score_tree, score_forest)

def eval_metrics(trained_results, X_test, Y_test):#,PATH,savedir,target):
    # confusion matrix and classification report
    from sklearn.metrics import classification_report, confusion_matrix
    
    #Decision TREES
    y_pred_tree=trained_results[0].predict(X_test)
    cm_tree = confusion_matrix(Y_test, y_pred_tree)
    print('Decision tree confusion Matrix\n', cm_tree)
    cr_tree = classification_report(Y_test, y_pred_tree, output_dict=True)
    cr_tree_df = pd.DataFrame(cr_tree).transpose()
    print('\nDecision tree classification report\n', cr_tree)
    
    #RANDOM FOREST
    y_pred_forest=trained_results[1].predict(X_test)
    cm_forest = confusion_matrix(Y_test, y_pred_forest)
    print('Random Forest confusion Matrix\n', cm_forest)
    cr_forest = classification_report(Y_test, y_pred_forest, output_dict=True)
    cr_forest_df = pd.DataFrame(cr_forest).transpose()
    print('\nRandom Forest classification report\n', cr_forest)
    cr_tree= '_Decision_tree_Classification_report.csv'
    cr_forest= '_Random_Forest_Classification_report.csv'
        
    cr_tree_file = PATH+ '/'+savedir+'/OLINDEX3'+cr_tree
    cr_forest_file = PATH+ '/'+savedir+'/OLINDEX3'+cr_forest
    cr_tree_df.to_csv(cr_tree_file)
    cr_forest_df.to_csv(cr_forest_file)
    return(cm_tree, cr_tree, cm_forest, cr_forest)

def plot(eval_results):#, PATH, savedir):
    #Plot feature evaluation
    # if answer == True:
    #     treetitle = 'Combined Decision Tree Model feature importance'
    #     foresttitle = 'Combined Random Forest Model feature importance'
    # else:
    #     treetitle = 'Decision Tree Model feature importance'
    #     foresttitle = 'Random Forest Model feature importance'
    
    treetitle = 'Decision Tree Model feature importance'
    foresttitle = 'Random Forest Model feature importance'
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(18, 9))
    ax1 = plt.subplot(121)
    sns.barplot(x=eval_results[0].values, y=eval_results[0].index)
    plt.xlabel('Feature score')
    plt.ylabel('Features', multialignment='center')
    plt.title(treetitle)
    plt.grid(True)
    plt.subplot(122, sharey=ax1, sharex=ax1)
    sns.barplot(x=eval_results[1].values, y=eval_results[1].index, order=eval_results[1].index)
    plt.xlabel('Feature score')
    plt.ylabel('Features', multialignment='center')
    plt.grid(True)
    plt.title(foresttitle)
    # plt.suptitle(target.columns[tgt], fontsize=16)
    #plt.subplots_adjust(bottom=0.5, top=0.5)
    # if answer == True:
    #     graphname= '_Combined_Feature_importance_graphs.png'
    # else:
    #     graphname='_Feature_importance_graphs.png'
        
    # figname =PATH+ '/'+savedir+'/'+target.columns[tgt]+graphname
    # plt.savefig(figname)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def save_models(PATH, savedir, trained_results):
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
    tree_path = PATH+'/'+savedir+'/'
    forest_path = PATH+'/'+savedir+'/'
    treename = SAVENAME[0]
    forestname = SAVENAME[1]

    joblib.dump(trained_results[0], tree_path+treename)
    print('\nSaved ', treename, "in ", savedir)
    joblib.dump(trained_results[1],forest_path+forestname)
    print('\nSaved ', forestname, "in ", savedir)


def main(features, target):
    # Y = target[target.columns[tgt]].values
    X = features.values
    Y = target.values
    
    #normalize train dataset
    #create train and test by splitting the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    
    #Trainig results
    print('Training...')
    trained_results = train(X_train, Y_train)
    print('\nEvaluating results...')
    eval_results = evaluation(trained_results, X_test, Y_test)
    save_models(PATH, savedir, trained_results)
    cm_tree, cr_tree, cm_forest, cr_forest = eval_metrics(trained_results, X_test, Y_test)

    #plot(eval_results)
    return (trained_results, eval_results, cm_tree, cr_tree, cm_forest, cr_forest)    


if __name__ == "__main__":
    
    root = Tk()
    root.withdraw()
    
    #select workdir
    PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
    print('Working folder:', PATH)
    
    print('\nSelect combined hdf file\n')
    complete_df = hdf2df()
    
    
    #creating savename and savedir
    SAVENAME = [('OLINDEX3_Decision_Trees.pkl'), 'OLINDEX3_Random_Forest.pkl']    
    savedir = 'OLINDEX3_models'
    
    #create features and target
    features = complete_df.drop('OLINDEX3', axis=1)
    features_nonan = features.fillna(0)
    target = complete_df['OLINDEX3']
    
    #training and evaluation
    trained_results, eval_results, cm_tree, cr_tree, cm_forest, cr_forest = main(features_nonan, target)
 


    