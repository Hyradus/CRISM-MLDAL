# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:52:53 2020

@author: HyraN
"""

# Before run, make sure that both mtrdr and classes hdr files for all dataset are present in the selected folder


import shutil
import os
import pandas as pd
import glob
from tkinter import Tk,filedialog

def combine(f):
    df = pd.read_hdf(f)
    name = f.split('_')
    filename=name[0]+'_'+datatype + '.hdf'
    cla = pd.read_hdf(filename)
    #remove nan rows or not
    if answer == True:        
        complete_df = pd.concat([df, cla[index]], axis=1, sort=False)
        complete_df_nonan = complete_df.dropna()
        return_df = complete_df_nonan
    elif answer == False:
        complete_df = pd.concat([df, cla[index]], axis=1, sort=False)
        return_df = complete_df
    return(return_df)

   
def combinehdf(PATH, fraction):
    os.chdir(PATH)
    extension = 'hdr.hdf'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]    
    #combine all files in the list
    CombData = pd.DataFrame()
    for f in all_filenames:
        df = combine(f)
        subset_df = df.sample(frac=fraction).sort_index(axis=0)
        CombData = CombData.append(subset_df, ignore_index=True)
        
    #export to hdf
    ln = len(all_filenames)
    savefolder = str(ln)+'_combined_DF_'+str(fraction*100)+ '%_' + str(datatype)
    if os.path.exists(savefolder):
        shutil.rmtree(savefolder)
        os.mkdir(savefolder)
    else:
        os.chdir(PATH)
        os.mkdir(savefolder)
    savename = savefolder+'/'+str(ln)+'_combined_DF_'+str(fraction*100)+ '%_' + str(datatype)+'.hdf'
    CombData.to_hdf(savename, 'Combined')
    return(CombData)

root = Tk()
root.withdraw()
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the folder where are hdf mtrdr and hdf classes files:")
print('Working folder:', PATH)


#select class index to combine e.g OLINDEX3
origin_type = 'original' #choose original or thresholded
data_type = 'features' #choose classes or features
datatype = origin_type +'_'+data_type
index = 'OLINDEX3' 
#select fraction
fraction = 1
#drop nan rows?
answer = False

comb_feat = combinehdf(PATH, fraction)

