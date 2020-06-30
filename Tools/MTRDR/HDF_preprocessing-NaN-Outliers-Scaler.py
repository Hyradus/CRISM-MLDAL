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
import time
import numpy as np

###### OPTIONS EDIT BEFORE RUN ######

# choice if if drop rows with full nan and replace remaining nan with 0
NAN = 'yes'
#choice if remove outliers
OUTLIERS = 'no'
# choice standardization (std), normalization (nrm), or none (None)
SCALE = 'None' 
# choice single or multi
CHOICE = 'single'

EXTENSION = '*?3.hdf'
###### END OPTIONS EDIT BEFORE RUN ######
def hdf2df():
    hdf = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select select hdf file to load:", filetypes= (('hdf files', '*.hdf'), ('all files', '*.*)))')))
    print('hdf file selected:', hdf)
    df = pd.read_hdf(hdf)
    return(df, hdf) 

def make_folder(PATH, name):
    folder = PATH+'/'+name
    if os.path.exists(folder):
           print(name, 'Folder exist, removing.')
           shutil.rmtree(folder)
           time.sleep(1)
           os.mkdir(folder)
           print(name, 'Folder created')
    else:
           print(name, ' Folder not exist, creating.')
           os.mkdir(folder)
           print('Created new ', name,' Folder')
    return(folder)

def outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    q1 = Q1 -1.5*IQR
    q3 = Q3 + 1.5 * IQR
    for col in df.columns:
        mean = df[col].mean()
        # print(mean)
        df[col] = np.where((df[col] < q1[col]) | (df[col] > q3[col]), mean, df[col])

    return(df)

def MinMax_Scaler(array):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(array)
    scaled_array = scaler.transform(array)
    return(scaled_array)

def Std_Scaler(array):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(array)
    scaled_array = scaler.transform(array)
    return(scaled_array)

def preprocessing_single(df, name):
    
    columns = df.columns
    if NAN == 'yes':
        print('\nComputing mean')
        # mean = df.mean()
        print('\nDropping NaNs')
        # df = df.dropna(how='all')
        # df = df.fillna(0)
        df.reset_index(drop=True,inplace=True)
        # mean = df.mean()
        print('\nReplacing residual NaN with mean values')
        # df = df.fillna(mean)
        # mean = None
    
    if OUTLIERS == 'yes' :
        print('\nReplacing outliers with mean values')
        df = outliers(df)
        
    if SCALE =='std':
        print('\nApply standardization')
        df = Std_Scaler(df)
    elif SCALE == 'nrm':
        print('\nApply Normalization')
        df = Std_Scaler(df)
    else:
        print('\nSkip Scaling')
    
    df = pd.DataFrame(df, columns=columns)
    savename = TEMPDIR+'/NaN_'+NAN+'_Outliers_'+OUTLIERS+'_Scaler_'+SCALE+'_'+name
    df.to_hdf(savename, 'Combined')   
    # return(df)
    
def preprocessing_multi(all_filenames, i):
    print('\nSelect combined hdf file to be processed')
    df = pd.read_hdf(all_filenames[i])
    df = preprocessing_single(df, str(all_filenames[i]))
    return (df)
    


def parallel_processing():
    from joblib import Parallel, delayed
    import psutil
    #JOBS=psutil.cpu_count(logical=False)
    JOBS = 8

    all_filenames = [i for i in glob.glob(EXTENSION)]
    print('Processing separate hdf')
    start = timer()
    Parallel (n_jobs=JOBS)(delayed(preprocessing_multi)(all_filenames,i)
                           for i in range(len(all_filenames)))
    end = timer()
    print('\n Parallel processing executed in :', end-start, 's')


def main():
    os.chdir(PATH)
    if CHOICE =='single':
        import pathlib
        df, hdf = hdf2df()
        name = pathlib.PurePath(hdf).name
        preprocessing_single(df, name)        
    else:
        parallel_processing()
    print('done')
 

if __name__ == "__main__":
    
    root = Tk()
    root.withdraw()
    PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the folder where are hdf mtrdr and hdf classes files:")
    print('Working folder:', PATH)
    os.chdir(PATH)
    all_filenames = [i for i in glob.glob(EXTENSION)]
    TEMPDIR = make_folder(PATH, str(len(all_filenames)) + '_' + 'preprocessed')

    main()

