# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:52:53 2020

@author: HyraN
"""

# Before run, make sure that both mtrdr and classes hdr files for all dataset are present in the selected folder

from timeit import default_timer as timer
import shutil
import os
import pandas as pd
import glob
from tkinter import Tk,filedialog


#select class INDEX to combine e.g OLINDEX3
ORIGIN_TYPE = 'original' #choose original or thresholded
DATA_TYPE = 'classes'
FILE_TYPE = ORIGIN_TYPE +'_'+DATA_TYPE
INDEX = 'OLINDEX3' # full or INDEX e.g OLINDEX3 
#select FRACTION
FRACTION = 1
COMBINE = 'single'
EXTENSION = '*?3.hdf'
# is data preprocessed?
PROC = 'no'

def make_folder(name):
    folder = PATH+'/'+name
    if os.path.exists(folder):
           print(name, 'Folder exist, removing.')
           shutil.rmtree(folder)
           os.mkdir(folder)
           print(name, 'Folder created')
    else:
           print(name, ' Folder not exist, creating.')
           os.mkdir(folder)
           print('Created new ', name,' Folder')
    return(folder)

def save_combname(all_filenames):
        ln = len(all_filenames)
        name = INDEX+'_'+str(ln)+'_combined_DF_'+str(FRACTION*100)+ '%_' + str(FILE_TYPE)
        savedir = make_folder(name)
        savename = savedir +'/'+name+'.hdf'
        return(savename)

def save_hdf(comb_df, folder, name):
    print('\nSaving combined hdf\n')
    start = timer()
    if COMBINE == 'single':
        # savename=save_combname(name)
        comb_df.to_hdf(name, 'Combined')
    else:        
        savename = folder+'/'+name
        comb_df.to_hdf(savename, 'single')
    end = timer()
    print('\nSaved in: ', end-start, 's')



def single_feat_class(f):
    print(f)
    df = pd.read_hdf(f)
    name = f.split('_')
    filename=name[0]+'_'+FILE_TYPE + '.hdf'
    classes = pd.read_hdf(filename)
    #remove nan rows or not
    if INDEX != 'full':
        comb_df = pd.concat([df, classes[INDEX]], axis=1, sort=False)
        # return_df = comb_df
    else:
        comb_df = pd.concat([df, classes], axis=1, sort=False)
        # return_df = comb_df
    return(comb_df)

  
def multi_hdf2single(PATH):
    all_filenames = [i for i in glob.glob(EXTENSION)]
    #combine all files in the list
    comb_df = pd.DataFrame()
    start = timer()
    
    for f in all_filenames:
        if PROC != 'yes':
            df = single_feat_class(f)
        else:
            df = pd.read_hdf(f)
        subset_df = df.sample(frac=FRACTION).sort_index(axis=0)
        comb_df = comb_df.append(subset_df, ignore_index=True)
    end = timer()
    print('\nCombined in: ', end-start, 's')
    #export to hdf
    print('\nExporting in hdf')
    
    savename = save_combname(all_filenames)
    save_hdf(comb_df, all_filenames, savename)
    return(comb_df)

def multi_feat_class(f, folder):
    print(f)q
    comb_df = single_feat_class(f)
    name = 'Combined_'+f
    save_hdf(comb_df, folder, name)

def multi_hdf2multi(PATH):
    from joblib import Parallel, delayed
    import psutil
    JOBS=psutil.cpu_count(logical=False)
    # JOBS = 10


    all_filenames = [i for i in glob.glob(EXTENSION)]
    print('Combining singles hdf')
    folder = make_folder(str(len(all_filenames)) + '_' + 'single_combined')
    start = timer()
    Parallel (n_jobs=JOBS)(delayed(multi_feat_class)(all_filenames[i],folder)
                            for i in range(len(all_filenames)))

    end = timer()
    print('\n Parallel combine in :', end-start, 's')




def main(PATH):
    os.chdir(PATH)
    if COMBINE == 'single':
        comb_df = multi_hdf2single(PATH)
    else:
        multi_hdf2multi(PATH)
        comb_df=None
    return(comb_df)

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the folder where are hdf mtrdr and hdf classes files:")
    print('Working folder:', PATH)
    
    
    # comb_feat = combinehdf(PATH, FRACTION)
    main(PATH)

