import pandas
import numpy as np
import cv2 as cv
from tkinter import Tk,filedialog
import os
import csv
import shutil
import time
from argparse import ArgumentParser


#global data_type
global file_type
global imgs
#data_type = 'Thresholded'
file_type = 'npy'



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
def read_single_image(SPECTRAL_INDEX, file_type,data_type):
    if file_type == 'dump':
        print('dump')
    elif file_type == 'npy':
        img = np.load(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type, allow_pickle=False)
    elif file_type == 'png':
        img = cv.imread(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type)
    #plt.imshow(img)
    return (img)

def array_to_series(img, i):
    vec = img.flatten()
    ser = pandas.Series(data=vec, name=SPECTRAL_INDEX[i]).astype(int)
    return(ser)

def merge_series_to_df(df,ser):
    assert df is not None
    df[ser.name] = ser
    return (df)

def thr_df(PATH):
    df = pandas.DataFrame()
    for i in range(len(SPECTRAL_INDEX)):
        img = read_single_image(SPECTRAL_INDEX[i], file_type, data_type='Thresholded')
        ser = array_to_series(img, i)
        #assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)
    return(df)

def bool_df(PATH):
    df = pandas.DataFrame()
    for i in range(len(SPECTRAL_INDEX)):
        img = read_single_image(SPECTRAL_INDEX[i], file_type, data_type='Thresholded_bool')
        ser = array_to_series(img, i)
        #assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)
    return(df)

def main(PATH, SPECTRAL_INDEX, savedir):
    
    os.chdir(PATH+"/..")
    path = os.getcwd()
    prefix = os.path.basename(path)
    savepath = PATH +'/'+ savedir + '/'
    THR_DF = thr_df(PATH)  
    BOL_DF = bool_df(PATH)
    feat_name = os.path.join(savepath + prefix + '_features.hdf')
    class_name = os.path.join(savepath + prefix + '_classes.hdf')
    THR_DF.to_hdf(feat_name, 'Thresholded_classes')
    BOL_DF.to_hdf(class_name, 'Boolean_classes')

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
        
    savedir = 'Processed'
    if os.path.exists(PATH+'/'+savedir):
        print('Folder exist, removing.')
        shutil.rmtree(PATH+'/'+savedir)
        time.sleep(1)
        os.mkdir(PATH+'/'+savedir)
        print('New Features folder created')
    else:
        print('Folder not exist, creating.')
        os.mkdir(PATH+'/'+savedir)
        print('Created new Features Folder')
    if args.cfg is None:
        SPECTRAL_INDEX = read_conf()
    else:
        
        SPECTRAL_INDEX=[]
        with open(cfg, newline='') as f:
            for row in csv.reader(f):
                SPECTRAL_INDEX.append(row[0])   
        
        

    main(PATH, SPECTRAL_INDEX, savedir)
    
    # def Y(mineral_name, index_filename):
    #     index_array = read_single_image(SPECTRAL_INDEX=index_filename, file_type=file_type, data_type='Thresholded_bool')
    #     #arr = define_class_array(index_array)
    #     ser = array_to_series(img=ind ex_array, i=SPECTRAL_INDEX.index(index_filename))
    #     return (ser)
    # DfY = pandas.DataFrame()
    # DfTarget = pandas.DataFrame()
    # for i in range(len(MINERAL_LIST)):
    #     dfY = Y(mineral_name=SPECTRAL_INDEX[i], index_filename=SPECTRAL_INDEX[i],)
    #     #assert len(dfY) == len(dfX)
    #     mineral=SPECTRAL_INDEX[i] + '.csv'
    #     dfY.to_csv(os.path.join(savepath, mineral.strip( )), index=False)
    #     ser = pandas.Series(data=dfY, name=SPECTRAL_INDEX[i])
    #     DfTarget = merge_series_to_df(DfTarget,ser)

    
