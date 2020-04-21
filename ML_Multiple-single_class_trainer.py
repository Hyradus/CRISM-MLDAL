"""
Created on Tue Mar 10 11:31:22 2020

@author: Hyradus
"""
import pandas as pd
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
import time
from argparse import ArgumentParser
import argparse


def read_conf():
# reading config file for minerals
    cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Spectral configuration file selected:', cfg)
    data=[]
    with open(cfg, newline='') as f:
        for row in csv.reader(f):
            data.append(row[0])
    return(data)

# def read_conf():
# # reading config file for minerals
#     cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
#     print('Spectral configuration file selected:', cfg)
#     with open(cfg, newline='') as f:
#         reader = csv.reader(f)
#         data = list(reader)
#     return(data)

def read_features():
    feats = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select index_features file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Features file selected:', feats)
    with open(feats, newline='') as f:
        features = pd.read_csv(f)
    return(features)

def read_target():
    miner = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be trained:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Features file selected:', miner)
    with open(miner, newline='') as f:
        #reader = csv.reader(f)
        mineral = pd.read_csv(f)
    return(mineral)


def ask():
    check = str(input("Are you using combined data? ? (Y/N): ")).lower().strip()
    yes = ['y', 'Y', 'yes']
    no = ['n', 'N', 'no']
    try:
        if check[0] in yes:
            return True
        elif check[0] in no:
            return False
        else:
            print('Invalid Input')
            return ask()
    except Exception as error:
        print("Valid inputs are: ", yes, ' or ', no)
        print(error)
        return ask()

# def data_norm():
#     features = read_features()
#     features_norm = features.copy()
#     to_norm = features[SPECTRAL_INDEX]
#     features_norm = (to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
#     print(features_norm)
#     return (features_norm)

# def normalizer(X_train, X_test):
#     norm = Normalizer()
#     X_train_norm = norm.fit_transform(X_train)
#     X_test_norm = norm.fit_transform(X_test)
#     return (X_train_norm, X_test_norm)

# def encoderY(Y_train, Y_test):
#     le = LabelEncoder()
#     Y_train_enc = le.fit_transform(Y_train)
#     Y_test_enc = le.transform(Y_test)
#     return (Y_train_enc, Y_test_enc)


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
    featImpDf = pd.DataFrame()
    featImpDf['Decision Trees'] = pd.Series(tree.feature_importances_,index=featnonan.columns).sort_values(ascending=False)
    featImpDf["Random Forest"] = pd.Series(forest.feature_importances_,index=featnonan.columns).sort_values(ascending=False)    
    print("\nFeatures importances for all models are:\n", featImpDf)
    
    return(featImpDf, score_tree, score_forest)

def eval_metrics(trained_results, X_test, Y_test, answer,PATH,savedir,target):
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
    
    if answer == True:
        cr_tree= '_Combinded_Decision_tree_Classification_report.csv'
        cr_forest= '_Combined_Decision_tree_Classification_report.csv'
    else:
        cr_tree= '_Decision_tree_Classification_report.csv'
        cr_forest= '_Random_Forest_Classification_report.csv'
        
    cr_tree_file = PATH+ '/'+savedir+'/'+target.columns[tgt]+cr_tree
    cr_forest_file = PATH+ '/'+savedir+'/'+target.columns[tgt]+cr_forest
    cr_tree_df.to_csv(cr_tree_file)
    cr_forest_df.to_csv(cr_forest_file)
    return(cm_tree, cr_tree, cm_forest, cr_forest)

def save_models(PATH, savedir, answer, SAVENAME, trained_results):
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
    if answer == True:
        treename= 'Combined_'+SAVENAME[0]
        forestname = 'Combined_'+SAVENAME[1]
    else:
        treename = SAVENAME[0]
        forestname = SAVENAME[1]

    joblib.dump(trained_results[0], tree_path+treename)
    print('\nSaved ', treename, "in ", savedir)
    joblib.dump(trained_results[1],forest_path+forestname)
    print('\nSaved ', forestname, "in ", savedir)

def plot(answer, eval_results, target, PATH, savedir):
    #Plot feature evaluation
    if answer == True:
        treetitle = 'Combined Decision Tree Model feature importance'
        foresttitle = 'Combined Random Forest Model feature importance'
    else:
        treetitle = 'Decision Tree Model feature importance'
        foresttitle = 'Random Forest Model feature importance'
        
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(18, 9))
    ax1 = plt.subplot(121)
    sns.barplot(x=eval_results[0]['Decision Trees'].values, y=eval_results[0].index, order=eval_results[0].index)
    plt.xlabel('Feature score')
    plt.ylabel('Features', multialignment='center')
    plt.title(treetitle)
    plt.grid(True)
    plt.subplot(122, sharey=ax1, sharex=ax1)
    sns.barplot(x=eval_results[0]['Random Forest'].values, y=eval_results[0].index, order=eval_results[0].index)
    plt.xlabel('Feature score')
    plt.ylabel('Features', multialignment='center')
    plt.grid(True)
    plt.title(foresttitle)
    plt.suptitle(target.columns[tgt], fontsize=16)
    #plt.subplots_adjust(bottom=0.5, top=0.5)
    if answer == True:
        graphname= '_Combined_Feature_importance_graphs.png'
    else:
        graphname='_Feature_importance_graphs.png'
        
    figname =PATH+ '/'+savedir+'/'+target.columns[tgt]+graphname
    plt.savefig(figname)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def export_graph(trained_results, answer, PATH, savedir, SPECTRAL_INDEX):
    #export graphics for decision tree
    from sklearn.tree import export_graphviz
    from sklearn import tree
    import collections
    import pydotplus
    clf = trained_results[0]
    if answer == True:
        output= 'Combined_Tree.png'
    else:
        output='Tree.png'
    os.chdir(PATH+'/'+savedir+'/')
    dot_data = tree.export_graphviz(clf, feature_names=SPECTRAL_INDEX,out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    colors = ('green', 'orange')
    edges = collections.defaultdict(list)
    
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    graph.write_png(output)

    #saving tree in dot file
    os.chdir(PATH+'/'+savedir+'/')
    dotfile = open("Decision_trees_graph.dot", 'w')
    export_graphviz(trained_results[0], out_file = dotfile)
    dotfile.close()
    # os.system('dot -Tpng tree.dot -o tree.png')

def main(featnonan, target, tgt, answer):    
    Y = target[target.columns[tgt]].values
    X = featnonan.values
    #create train and test by splitting the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    
    #Trainig results
    trained_results = train(X_train, Y_train)
    
    #Evaluate feature score
    eval_results = evaluation(trained_results, X_test, Y_test)
    
    #Saving models    
    SAVENAME = [(target.columns[tgt]+'_Decision_Trees.pkl'), target.columns[tgt]+'_Random_Forest.pkl']    
    ## removing older models and writing new ones
    savedir = target.columns[tgt]+'_models'

    save_models(PATH, savedir, answer, SAVENAME, trained_results)
    plot(answer, eval_results, target, PATH, savedir) 
    cm_tree, cr_tree, cm_forest, cr_forest = eval_metrics(trained_results, X_test, Y_test, answer,PATH,savedir,target)
    export_graph(trained_results, answer, PATH, savedir, SPECTRAL_INDEX)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--wdir',
                        help='Dir where data (source) is.')
    parser.add_argument('--cfg', help='Input csv file')
    parser.add_argument('--fea', help='Input features csv file')
    parser.add_argument('--cla', help='Input classes csv file')
    
    args = parser.parse_args()
    modpath = args.wdir
    cfg = args.cfg
    root = Tk()
    
    if args.wdir is None:
        root.withdraw()
        PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
        print('Working folder:', PATH)
    else:
        PATH = args.wdir
    
    if args.cfg is None:
        SPECTRAL_INDEX = read_conf()
    else:
        
        SPECTRAL_INDEX=[]
        with open(cfg, newline='') as f:
            for row in csv.reader(f):
                SPECTRAL_INDEX.append(row[0])   
    if args.fea is None:
        features = read_features()
    else:
        with open(args.fea, newline='') as f:
                #reader = csv.reader(f)
                features = pd.read_csv(f)  
      
    if args.cla is None:
        target = read_target()   
    else:
        with open(args.cla, newline='') as f:
                #reader = csv.reader(f)
                target = pd.read_csv(f)  
 
    
    featnonan = features.fillna(0)

    answer = ask()
    start = timer()
    for tgt in range(len(features.columns)):
        print('Training ', features.columns[tgt], ' models..')
        main(featnonan, target, tgt, answer)
        
    end = timer()
    print('All models trained in ', end-start, ' seconds')