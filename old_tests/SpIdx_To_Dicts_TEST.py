
"""
CRISM MTRDR Spectral Index Extractor

@Hyradus

"""
import os
import rasterio as rio
from spectral import envi
import numpy as np
import cv2 as cv
import shutil
from tkinter import filedialog
from tkinter import Tk
import csv
from argparse import ArgumentParser

def ori_img(IMAGE, i, IDX, RESWID, RESHEI):
    im = IMAGE.read(i+1, masked=True).astype('float64')
    y=int((im.shape[0]-RESWID)/2)
    x=int((im.shape[1]-RESHEI)/2)

# Crop, convert back from numpy to PIL Image and and save
    cropped=im[x:x+RESWID,y:y+RESHEI]
    savename = IDX[i][0] + '_original'
    filename = os.path.join(SAVEPATH, savename)
    cv.imwrite(filename+'.png', cropped.astype('uint8'))
    cropped.dump(filename)

    return(im)

def thr_img(IMAGE, i, IDX, BANDor, THR, RESWID, RESHEI):
    BANDth = np.ma.masked_where(BANDor == 255, BANDor)
    BANDth = np.ma.filled(BANDth, np.nan)
    Bpc = np.nanpercentile(BANDth, THR)
    print(THR, 'th percentile for index', IDX[i][0], 'is: ', Bpc, '\n')
    BANDth = np.ma.masked_inside(BANDth, 1, Bpc)
    BANDth = np.ma.filled(BANDth, np.nan)
    y=int((BANDth.shape[0]-RESWID)/2)
    x=int((BANDth.shape[1]-RESHEI)/2)
    cropped = BANDth[x:x+RESWID,y:y+RESHEI]
    savename = IDX[i][0] + '_Thresholded'
    _filename = os.path.join(SAVEPATH, savename)
    cv.imwrite(_filename+'.png', cropped.astype('uint8'))
    np.save(_filename, cropped)

    return(BANDth)

def bol_img(IMAGE, i, IDX, BANDth, RESWID, RESHEI):
    bol = np.isfinite(BANDth)
    y=int((bol.shape[0]-RESWID)/2)
    x=int((bol.shape[1]-RESHEI)/2)
    cropped = bol[x:x+RESWID,y:y+RESHEI]
    savename = IDX[i][0] + '_Thresholded_bool'
    _filename = os.path.join(SAVEPATH, savename)
    np.save(_filename, cropped)

    return(bol)

def img(filepath, subdir, files):
    IMG = filepath
    IMAGE = rio.open(IMG)
    for fname in files:
        filepath = subdir + os.sep + fname
        if fname.endswith(".hdr"):
            hdr(IMAGE, filepath)
    return(IMAGE, filepath)

def hdr(IMAGE, filepath):
    FHDR = filepath
    HDR = envi.read_envi_header(FHDR)
    IDX = [(HDR['band names'][i], HDR['default bands'][i]) for i in range(0, len(HDR['band names']))]
    for i in range(len(HDR['default bands'])):
        if IDX[i][0] in SPIDX and SPIDX.get(IDX[i][0]) == [255, 255]:
            print('present', IDX[i][0], 'value: ', SPIDX[IDX[i][0]])
            BANDor = ori_img(IMAGE, i, IDX, RESWID, RESHEI)
            SPIDXOR[IDX[i][0]] = ori_img(IMAGE, i, IDX, RESWID, RESHEI)
            BANDth = thr_img(IMAGE, i, IDX, BANDor, THR, RESWID, RESHEI)
            SPIDX[IDX[i][0]] = thr_img(IMAGE, i, IDX, BANDor, THR, RESWID, RESHEI)
            SPIDXBOL[IDX[i][0]] = bol_img(IMAGE, i, IDX, BANDth, RESWID, RESHEI)
        elif IDX[i][0] in SPIDX and SPIDX.get(IDX[i][0]) != [255, 255]:
            print('Present but already computed', IDX[i][0])
    return(SPIDXOR,SPIDX,SPIDXBOL)

######### MAIN PROGRAM #########

def main(workdir, thresh, dirout):
    WORKDIR = workdir
    for subdir, dirs, files in os.walk(WORKDIR, followlinks=True): # reading filepaths
        for fname in files:
            filepath = subdir + os.sep + fname
            if filepath.endswith(".img"): # opening *.img files
                img(filepath, subdir, files)

if __name__ == "__main__":
# Parsing arguments from cli
    parser = ArgumentParser()
    parser.add_argument('--wdir', default='data',
                        help='Dir where data (source) is.')
    parser.add_argument('--odir', default='output',
                        help='Dir where (result) data is going to.')
    parser.add_argument('--thresh', default=10, type=float,
                        help='Threshold () to apply')
    args = parser.parse_args()
# or asking for directories
    WORKDIR = args.wdir
    if not os.path.exists(WORKDIR):
        #print("{} does not exist. Ciao.".format(WORKDIR))
        root = Tk()
        root.withdraw()
        WORKDIR = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
        print('Working folder:', WORKDIR)

    SAVEPATH = args.odir
    if not os.path.exists(SAVEPATH):
        SAVEPATH = WORKDIR +'/processed/'
        if not os.path.exists(SAVEPATH):
            #shutil.rmtree(SAVEPATH)
            os.mkdir(SAVEPATH)
            PROCPATH = WORKDIR + '/processed/features/'
            os.mkdir(PROCPATH)
            print("Directory ", SAVEPATH, " Created ")
        else:
            shutil.rmtree(SAVEPATH)
            os.mkdir(SAVEPATH)
            PROCPATH = WORKDIR + '/processed/features/'
            os.mkdir(PROCPATH)
            print("Directory ", SAVEPATH, " Created ")
    else:
        print("Directory ", SAVEPATH, " already exists")

# reading config file for minerals
    cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
    print('Spectral configuration file selected:', cfg)
    with open(cfg, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

# creating dictionaries
    SPIDX = { data[i][0] : [255, 255] for i in range(0, len(data) ) }
    SPIDXOR = { data[i][0] : [255, 255] for i in range(0, len(data) ) }
    SPIDXBOL = { data[i][0] : [255, 255] for i in range(0, len(data) ) }

    THR = args.thresh
    RESWID = 550
    RESHEI = 550
    main(WORKDIR, THR, SAVEPATH)