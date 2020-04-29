# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:47:01 2020

@author: HyraN
"""

import numpy as np
from tkinter import Tk,filedialog
import os
import glob
import ntpath
import time
import rasterio as rio
from spectral import envi
import cv2 as cv
import shutil
import pandas as pd
from argparse import ArgumentParser


def path_leaf(tgtname):
    head, tail = ntpath.split(tgtname)
    return tail or ntpath.basename(head)

def hdr(IMAGE, filepath, filename, extracted_index):   
    FHDR = modpath+'/'+filename+'.hdr'
    HDR = envi.read_envi_header(FHDR)
    IDX = [(HDR['band names'][i], HDR['default bands'][i]) for i in range(0, len(HDR['band names']))]   
    for i in range(len(HDR['default bands'])):
        if any(IDX[i][0] in ele for ele in extracted_index):
            print(IDX[0][0],' already extracted')
            continue
        else:
            Or_Band = ori_img(IMAGE, i, IDX,filename, savepath)
            thr_img(IMAGE, i, IDX, Or_Band, THR, filename, savepath) 

def band_index_creator(savepath):
    ext = 'npy'
    os.chdir(savepath)
    extracted_bands = [i for i in glob.glob('**/**/*d.{}'.format(ext), recursive=True)]
    band_index = []
    items_to_remove = ['Extracted', '\\', '_Thresholded.npy']
    for i in range(len(extracted_bands)):
        result = extracted_bands[i]
        for x in items_to_remove:
            result = result.replace(x,'')        
        band_index.append(result)
    return(band_index)
       
def ori_img(IMAGE, i, IDX, filename, savepath):
    im = IMAGE.read(i+1, masked=True).astype('float64')
    savename = IDX[i][0] +'_original'
    savefile = os.path.join(savepath, savename)
    cv.imwrite(savefile+'.png', im.astype('uint8'))
    im.dump(savefile)
    return(im)

def thr_img(IMAGE, i, IDX, BANDor, THR, filename, savepath):
    BANDth = np.ma.filled(BANDor, np.nan)
    Bpc = np.nanpercentile(BANDth, THR)
    print(THR, 'th percentile for index', IDX[i][0], 'is: ', Bpc, '\n')
    BANDth_nonan = np.nan_to_num(BANDth)
    BANDth_nonan[BANDth_nonan < Bpc] = 0
    bol = np.array(BANDth_nonan, dtype=bool)
    savename = IDX[i][0] +'_Thresholded'
    filename = os.path.join(savepath, savename)
    cv.imwrite(filename+'.png', BANDth_nonan.astype('uint8'))
    np.save(filename, BANDth_nonan)
    #cv.imwrite(filename+'_bool'+'.png', BANDth_nonan.astype('uint8'))
    np.save(filename+'_bool', bol)
    #bol = np.isfinite(BANDth)
    return(BANDth_nonan, bol)

def main(modpath, savepath, THR):
    os.chdir(modpath)
    extension = 'img'
    all_index = [i for i in glob.glob('*.{}'.format(extension))]
   
    ext = 'img'
    extracted_index = [i for i in glob.glob('**/**/*d.{}'.format(ext), recursive=True)]
    
    for i in range(len(all_index)):
        tgtname=all_index[i]
        IMAGE = rio.open(tgtname)
        filename = os.path.splitext(path_leaf(tgtname))[0]
        hdr(IMAGE, modpath, filename, extracted_index)
        
    band_index = band_index_creator(savepath)
        
    df = pd.DataFrame(band_index)
    df.to_csv(savepath+"./Band_Index.csv", sep=',',index=False, header=False)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--wdir',
                        help='Dir where data (source) is.')
    parser.add_argument('--thr', default=20, type=float,
                        help='Threshold to apply')
    args = parser.parse_args()
    modpath = args.wdir
    THR = args.thr
    if args.wdir is None:
        root = Tk()
        root.withdraw()
        modpath = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the folder with the original *.img and *.hdr:")
        print('Working folder:', modpath)
    else:
        modpath = args.wdir
        
        
    savedir = 'Extracted'
    savepath = modpath +'/'+savedir+'/'
    
    if os.path.exists(modpath+'/'+savedir):
        print('Folder exist, removing.')
        shutil.rmtree(modpath+'/'+savedir)
        time.sleep(1)
        os.mkdir(modpath+'/'+savedir)
        print('New folder created')
    else:
        print('Folder not exist, creating.')
        os.mkdir(modpath+'/'+savedir)
        print('Created new Folder')
    main(modpath, savepath, THR)


  