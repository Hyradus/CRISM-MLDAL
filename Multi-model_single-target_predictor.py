
import pandas as pd
from tkinter import Tk,filedialog
import os
import csv
import joblib
import glob
import ntpath
from timeit import default_timer as timer
from argparse import ArgumentParser


def read_conf():
# reading config file for minerals
    cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Spectral configuration file selected:', cfg)
    data=[]
    with open(cfg, newline='') as f:
        for row in csv.reader(f):
            data.append(row[0])
    return(data)

def read_pred_target(tgt):
    #tgt = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be tested:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Submitted file selected:', tgt)
    with open(tgt, newline=''):
        #reader = csv.reader(f)
        pred_target = pd.read_csv(tgt)
    return(pred_target, tgt)    

def read_model():
    modelfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select model to load:", filetypes= (('pkl files', '*.pkl'), ('all files', '*.*')))
    print('Submitted file selected:', modelfile)
    model = joblib.load(modelfile)
    modname = os.path.splitext(modelfile)[0]
    return(model, modname)

def path_leaf(tgtname):
    head, tail = ntpath.split(tgtname)
    return tail or ntpath.basename(head)

def pred_df(all_trees, all_forest, target): ##### Saving predictions as dataframes 
    tree_df = pd.DataFrame()
    forest_df = pd.DataFrame()
    for i in range(len(all_trees)):
        tree_model = joblib.load(all_trees[i])
        #treename = os.path.splitext(all_trees[i])[0]
#        print(all_trees[i])
        forest_model = joblib.load(all_forest[i])
        #forestname = os.path.splitext(all_forest[i])[0]
#        print(all_forest[i])        
        pred_tree = tree_model.predict(target)
        ser = pd.Series(pred_tree, name=all_trees[i])
        tree_df[ser.name] = ser
        pred_forest = forest_model.predict(target)
        ser = pd.Series(pred_forest, name=all_forest[i])
        forest_df[ser.name] = ser
    return(tree_df, forest_df)#, treename, forestname)


def make_lists(indpath, modpath):
    os.chdir(indpath)
    extension = 'csv'
    all_index = [i for i in glob.glob('*.{}'.format(extension))]
    os.chdir(modpath)
    extension = 'pkl'
    all_trees = [i for i in glob.glob('*Trees.{}'.format(extension))]
    all_forest = [i for i in glob.glob('*Forest.{}'.format(extension))]  
    return(all_index, all_trees, all_forest)

def plot(tree_df, all_trees, forest_df, all_forest, modpath, tgtname):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set(style='whitegrid', font_scale=2 )
    sns.set_style
    
    
    if len(all_trees) > 1:
        tree_name = tgtname + ' Predictions using multiple Decision tree models'
        fig, ax = plt.subplots(1, len(all_trees), figsize=(40, 10), sharey=False)
        for i, subplot in zip(tree_df.columns.values, ax.flatten()):
            variable = tree_df[i].values
            sns.countplot(variable, ax=subplot, palette='deep')
            for l in range(len(all_trees)):
                title = 'Model: ' + all_trees[l]
                ax[l].set_title(title, fontsize=20)
                #    for label in subplot.get_xticklabels():
    else:
        tree_name = tgtname + ' Predictions using single Decision tree models'
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=False)
        variable = tree_df.values.flatten()
        sns.countplot(variable, palette='deep')
        title = 'Model: ' + all_trees[0]
        ax.set_title(title, fontsize=20)
    plt.suptitle(tree_name, fontsize=20)
    #fig.text(0.5, 0.04, tree_name, ha='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname =modpath+'/'+tree_name +'_Predictions'
    plt.savefig(figname)
    plt.show()
      
    sns.set(style='whitegrid', font_scale=2 )
    sns.set_style
    
    if len(all_forest) > 1:
        forest_name = tgtname +' Predictions using multiple Random Forest models'
        fig, ax = plt.subplots(1, len(all_forest), figsize=(40, 10), sharey=False)
        for i, subplot in zip(forest_df.columns.values, ax.flatten()):
            variable = forest_df[i].values
            sns.countplot(variable, ax=subplot, palette='deep')
            for l in range(len(all_forest)):
                title = 'Model: ' + all_forest[l]
                ax[l].set_title(title, fontsize=20)
                #    for label in subplot.get_xticklabels():
    else:
        forest_name = tgtname +' Predictions using single Random Forest models'
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=False)
        variable = forest_df.values.flatten()
        sns.countplot(variable, palette='deep')
        title = 'Model: ' + all_forest[0]
        ax.set_title(title, fontsize=20)
    plt.suptitle(forest_name, fontsize=20)
    #fig.text(0.5, 0.04, forest_name, ha='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname =modpath+'/'+forest_name +'_Predictions'
    plt.savefig(figname)
    plt.show()
    
def main(indpath, modpath, target, tgtname):
    all_index, all_trees, all_forest, = make_lists(indpath, modpath)
    tree_df, forest_df= pred_df(all_trees, all_forest, target)
    tgtname = os.path.splitext(path_leaf(tgtname))[0]
    plot(tree_df, all_trees, forest_df, all_forest, modpath, tgtname)
    
    
if __name__ == "__main__":
    
    root = Tk()
    root.withdraw()
    modpath = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the models folder:")
    print('Working folder:', modpath)
    
    indpath = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the indexes folder:")
    print('Working folder:', indpath)
    start = timer()
    all_index, all_trees, all_forest, = make_lists(indpath, modpath)
    for i in range(len(all_index)):
        target, tgt = read_pred_target(indpath+'/'+all_index[i])
        main(indpath, modpath, target, tgt)
        print(all_index[i])
    end = timer()
    print('All models and index predicted in ', end-start, ' seconds')

