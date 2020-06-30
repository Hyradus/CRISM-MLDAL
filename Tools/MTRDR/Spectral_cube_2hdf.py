# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:27:32 2020

@author: Hyradus
"""

from spectral import envi
import numpy as np
from tkinter import Tk,filedialog
from argparse import ArgumentParser
import os
import time
import shutil
import pandas as pd
import ntpath
from timeit import default_timer as timer

###### OPTIONS EDIT BEFORE RUN ######5
#choice if convert CRISM no-data values (65536) to NaN)
CHOICE_NAN = True

###### END OPTIONS EDIT BEFORE RUN ######

def path_leaf(tgtname):
    head, tail = ntpath.split(tgtname)
    return tail or ntpath.basename(head)

def img2np(hdrfile):
    imgfile = envi.open(hdrfile)
    #read hdr files
    header_hdr = envi.read_envi_header(hdrfile)
    #create cube
    cube = imgfile[:,:,:]
    #replace "nodata" values with np.nan
    #cube_np_nan = np.where(cube < cube.max(), cube, np.nan)
    #create index for wavelengths
    names = header_hdr['wavelength']
    if CHOICE_NAN == True:
        print('removing nan')
        cube = np.where(cube <65535, cube, np.nan)
        # cube = np.ma.masked_where(cube == 65535, cube, copy=True)
    else:
        cube = cube
    return(cube, names)

def CUBE_np2df(np_cube, names):
    cube_df = pd.DataFrame()
    for i in range(len(names)):
        img_slice = np_cube[:,:,i]
        series = pd.Series(data=img_slice.flatten(), name=names[i])
        cube_df[series.name] = series        
    return(cube_df)

def CUBE_df2hdf(cube_df, hdrfile):
    savepath = os.path.dirname(hdrfile)
    import pathlib
    name = pathlib.PurePath(hdrfile).name.split('.')[0] + '.hdf'
    savename = savepath + '/'+name
    start = timer()
    cube_df.to_hdf(savename, 'df')
    end = timer()
    print(savename, ' hdf exported in: ', savepath, 'in: ', end-start, 's')


def main(hdrfile):
    cube, names = img2np(hdrfile)
    cube_df = CUBE_np2df(cube, names)
    print('Saving to hdf')
    #export cube to hdf
    CUBE_df2hdf(cube_df, hdrfile)
    return(cube, cube_df)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hdrf',
                        help='Select*.hdr file.')
    args = parser.parse_args()
    hdrfile = args.hdrf
    
    if args.hdrf is None:
        root = Tk()
        root.withdraw()
        hdrfile = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select hdr file:", filetypes= (('hdr files', '*.hdr'), ('all files', '*.*)))')))
        print('hdr file selected: ', hdrfile)        
    else:
        modpath = args.wdir

cube, cube_df = main(hdrfile)


