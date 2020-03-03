
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



######### Create dictionary with Spectral Indexes and NaN values (SpIndx) ####

SPIDX = {'OLINDEX3' : [255, 255],
         'LCPINDEX2' : [255, 255],
         'HCPINDEX2' : [255, 255],
         'BD2100_2' : [255, 255],
         'BD1900_2' : [255, 255],
         'BDI1000VIS' : [255, 255],
         'D2300' : [255, 255],
         'SINDEX2' : [255, 255],
         'R770' : [255, 255],
         }

SPIDXOR = {'OLINDEX3' : [255, 255],
           'LCPINDEX2' : [255, 255],
           'HCPINDEX2' : [255, 255],
           'BD2100_2' : [255, 255],
           'BD1900_2' : [255, 255],
           'BDI1000VIS' : [255, 255],
           'D2300' : [255, 255],
           'SINDEX2' : [255, 255],
           'R770' : [255, 255],
           }

SPIDXBOL = {'OLINDEX3' : [255, 255],
            'LCPINDEX2' : [255, 255],
            'HCPINDEX2' : [255, 255],
            'BD2100_2' : [255, 255],
            'BD1900_2' : [255, 255],
            'BDI1000VIS' : [255, 255],
            'D2300' : [255, 255],
            'SINDEX2' : [255, 255],
            'R770' : [255, 255],
            }

######### MAIN PROGRAM #########

def main(workdir, thresh, dirout):
    WORKDIR = workdir
    THR = thresh
    SAVEPATH = dirout
    for subdir, dirs, files in os.walk(WORKDIR, followlinks=True): # reading filepaths
        for fname in files:
            filepath = subdir + os.sep + fname
            if filepath.endswith(".img"): # opening *.img files
                #print(filepath)
                IMG = filepath
                IMAGE = rio.open(IMG)
                for fname in files:
                    filepath = subdir + os.sep + fname
                    if fname.endswith(".hdr"):
                        #print(filepath)
                        FHDR = filepath
                        HDR = envi.read_envi_header(FHDR)
                        IDX = [(HDR['band names'][i], HDR['default bands'][i]) for i in range(0, len(HDR['band names']))]
                        for i in range(len(HDR['default bands'])):
                            if IDX[i][0] in SPIDX and SPIDX.get(IDX[i][0]) == [255, 255]:
                                print('present', IDX[i][0], 'value: ', SPIDX[IDX[i][0]])
                                BANDor = (IMAGE.read(i+1, masked=True)).astype('float64')
                                savename = IDX[i][0] + '_original'
                                _filename = os.path.join(SAVEPATH, savename)
                                cv.imwrite(_filename+'.png', BANDor.astype('uint8'))
                                BANDor.dump(_filename)
                                SPIDXOR[IDX[i][0]] = IMAGE.read(i+1, masked=True)
                                BANDth = np.ma.masked_where(BANDor == 255, BANDor)
                                BANDth = np.ma.filled(BANDth, np.nan)
                                Bpc = np.nanpercentile(BANDth, THR)
                                print(THR, 'th percentile for index', IDX[i][0], 'is: ', Bpc, '\n')
                                BANDth = np.ma.masked_inside(BANDth, 1, Bpc)
                                BANDth = np.ma.filled(BANDth, np.nan)
                                SPIDX[IDX[i][0]] = BANDth
                                savename = IDX[i][0] + '_Thresholded'
                                _filename = os.path.join(SAVEPATH, savename)
                                cv.imwrite(_filename+'.png', BANDth.astype('uint8'))
                                np.save(_filename, BANDth)
                                bol = np.isfinite(BANDth)
                                SPIDXBOL[IDX[i][0]] = bol

                            elif IDX[i][0] in SPIDX and SPIDX.get(IDX[i][0]) != [255, 255]:
                                print('Present but already computed', IDX[i][0])



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--wdir', default='data',
                        help='Dir where data (source) is.')
    parser.add_argument('--odir', default='output',
                        help='Dir where (result) data is going to.')
    parser.add_argument('--thresh', default=10, type=float,
                        help='Threshold () to apply')
    args = parser.parse_args()

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
            shutil.rmtree(SAVEPATH)
            os.mkdir(SAVEPATH)
            print("Directory ", SAVEPATH, " Created ")
    else:
        print("Directory ", SAVEPATH, " already exists")

    THR = args.thresh

    main(WORKDIR, THR, SAVEPATH)
