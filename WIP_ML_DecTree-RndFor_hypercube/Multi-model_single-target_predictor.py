
import pandas as pd
from tkinter import Tk,filedialog
import os
# import csv
import joblib
import glob
import ntpath
from timeit import default_timer as timer
from spectral import envi
import numpy as np

def csv2tgt():
    target = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be trained:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Features file selected:', target)
    with open(target, newline='') as f:
        #reader = csv.reader(f)
        mineral = pd.read_csv(f)
    return(mineral)

def hdf2tgt():
    target = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select Spectral index file to be trained:", filetypes= (('hdf files', '*.hdf'), ('all files', '*.*)))')))
    print('Features file selected:', target)
    target_df = pd.read_hdf(target)
    return(target_df)  
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
    target=target.fillna(0)
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


def model_lists(modpath):
    os.chdir(modpath)
    extension = 'pkl'
    all_trees = [i for i in glob.glob('*Trees.{}'.format(extension))]
    all_forest = [i for i in glob.glob('*Forest.{}'.format(extension))]  
    return(all_trees, all_forest)

def pred2img(tree_df, forest_df, hdrfile):
    header_hdr = envi.read_envi_header(hdrfile)
    width = int(header_hdr['lines'])
    height = int(header_hdr['samples'])
    tree_np = tree_df.iloc[:,0].to_numpy()
    forest_np = forest_df.iloc[:,0].to_numpy()    
    tree_img = np.reshape(tree_np, (width, height))    
    forest_img = np.reshape(forest_np, (width, height))
    return(tree_img, forest_img)
    
def main(all_trees, all_forest, target):
    tree_df, forest_df= pred_df(all_trees, all_forest, target)
    
    return(tree_df, forest_df)
    
    
    
if __name__ == "__main__":
    
    root = Tk()
    root.withdraw()
    modpath = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the models folder:")
    print('Working folder:', modpath)
    hdrfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select hdr file:", filetypes= (('hdr files', '*.hdr'), ('all files', '*.*)))')))
    print('hdr file selected: ', hdrfile)  
    
    start = timer()
    all_trees, all_forest, = model_lists(modpath)
    target = hdf2tgt()
    tree_df, forest_df = main(all_trees, all_forest, target)
    tree_img, forest_img = pred2img(tree_df, forest_df, hdrfile)
    end = timer()
    
    print('All models and index predicted in ', end-start, ' seconds')

    
    import cv2 as cv
    
    Olindex3_thr_bol_npy = np.load('D:/Mars_CRISM_DATA/Datasets/Summary_Products/2007_360/frt00008fc1/Extracted/OLINDEX3_Thresholded_bool.npy')
    Olindex3_or_png = cv.imread('D:/Mars_CRISM_DATA/Datasets/Summary_Products/2007_360/frt00008fc1/Extracted/OLINDEX3_original.png')
    
    
    
    forest_img_masked = np.ma.masked_where(forest_img == 0, forest_img, copy=True)
    tree_img_masked = np.ma.masked_where(tree_img == 0, tree_img, copy=True)
    Olindex3_thr_bool_npy_masked = np.ma.masked_where(Olindex3_thr_bol_npy == 0, Olindex3_thr_bol_npy, copy=True)

    
    cv.imwrite('Olindex3_forest_masked.png', forest_img_masked*255)
    cv.imwrite('Olindex3_tree_masked.png', tree_img_masked*255)
    cv.imwrite('Olindex3_thresholded_bool_masked.png', Olindex3_thr_bool_npy_masked*255)
    cv.imwrite('Olindex3_original.png', Olindex3_or_png)
