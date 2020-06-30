import pandas
import numpy as np
import cv2 as cv
from tkinter import Tk,filedialog
import os
import csv
import shutil
import time
from argparse import ArgumentParser
import pickle

#global data_type
global file_type
global data_type
global imgs

def make_folder(PATH, fname):
    savepath = PATH+'/'+fname
    if os.path.exists(savepath):
           print('Folder exist, removing.')
           shutil.rmtree(savepath)
           time.sleep(1)
           os.mkdir(savepath)
           print('New model folder created')
    else:
           print('Folder not exist, creating.')
           os.mkdir(savepath)
           print('Created new Trained_Models Folder')
    return(savepath)

def read_conf():
# reading config file for minerals
    cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Spectral configuration file selected:', cfg)
    data=[]
    with open(cfg, newline='') as f:
        for row in csv.reader(f):
            data.append(row[0])
    return(data)

# read the 9 index-images (either original or the thresholde)
def read_single_image(SPECTRAL_INDEX, data_type, file_type):
    if file_type == 'dat':
        print('\nOpening dat file: ', SPECTRAL_INDEX)
        img = np.load(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type, allow_pickle=True)
    elif file_type == 'npy':
        print('\nOpening npy file: ', SPECTRAL_INDEX)
        img = np.load(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type, allow_pickle=False)
    elif file_type == 'png':
        print('\nOpening png file: ', SPECTRAL_INDEX)
        img = cv.imread(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type)
    #plt.imshow(img)
    return (img)

def array_to_series(img, i):
    vec = img.flatten()
    ser = pandas.Series(data=vec, name=SPECTRAL_INDEX[i])
    return(ser)

def merge_series_to_df(df,ser):
    assert df is not None
    df[ser.name] = ser
    return (df)

def thr_df(PATH):
    df = pandas.DataFrame()
    for i in range(len(SPECTRAL_INDEX)):
        img = read_single_image(SPECTRAL_INDEX[i], data_type, file_type)
        ser = array_to_series(img, i)
        #assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)
    return(df)

def bool_df(PATH):
    df = pandas.DataFrame()
    for i in range(len(SPECTRAL_INDEX)):
        img = read_single_image(SPECTRAL_INDEX[i], data_type, file_type)
        ser = array_to_series(img, i)
        #assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)
    return(df)

def main(PATH, SPECTRAL_INDEX, savedir):
    
    os.chdir(PATH+"/..")
    path = os.getcwd()
    prefix = os.path.basename(path)
    # savepath = PATH +'/'+ savedir + '/'
    THR_DF = thr_df(PATH)  
    BOL_DF = bool_df(PATH)
    class_name = os.path.join(savedir +'/' + prefix + '_' + data_type + '_classes.hdf')
    class_bol_name = os.path.join(savedir +'/'+ prefix + '_' + data_type +'_classes_bol.hdf')
    THR_DF.to_hdf(class_name, '_classes')
    BOL_DF.to_hdf(class_bol_name, '_bool_classes')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--wdir',
                        help='Dir where data (source) is.')
    parser.add_argument('--cfg', help='Input csv file')
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
        
    # savedir = 'Processed'
    # if os.path.exists(PATH+'/'+savedir):
    #     print('Folder exist, removing.')
    #     shutil.rmtree(PATH+'/'+savedir)
    #     time.sleep(1)
    #     os.mkdir(PATH+'/'+savedir)
    #     print('New Features folder created')
    # else:
    #     print('Folder not exist, creating.')
    #     os.mkdir(PATH+'/'+savedir)
    #     print('Created new Features Folder')
        
    savedir=make_folder(PATH, 'Processed')
    
    if args.cfg is None:
        SPECTRAL_INDEX = read_conf()
    else:
        
        SPECTRAL_INDEX=[]
        with open(cfg, newline='') as f:
            for row in csv.reader(f):
                SPECTRAL_INDEX.append(row[0])   
                
    data_type = 'original'
    file_type = 'dat'
    main(PATH, SPECTRAL_INDEX, savedir)

    
