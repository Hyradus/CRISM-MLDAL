
"""
CRISM Analysis TEST FILE
@Hyradus
"""
import os
import rasterio as rio
from spectral import envi
import numpy as np
from skimage.io import imsave
import csv

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

SPIDXor = {'OLINDEX3' : [255, 255],
         'LCPINDEX2' : [255, 255],
         'HCPINDEX2' : [255, 255],
         'BD2100_2' : [255, 255],
         'BD1900_2' : [255, 255],
         'BDI1000VIS' : [255, 255],
         'D2300' : [255, 255],
         'SINDEX2' : [255, 255],
         'R770' : [255, 255],
        }

SPIDXbol = {'OLINDEX3' : [255, 255],
         'LCPINDEX2' : [255, 255],
         'HCPINDEX2' : [255, 255],
         'BD2100_2' : [255, 255],
         'BD1900_2' : [255, 255],
         'BDI1000VIS' : [255, 255],
         'D2300' : [255, 255],
         'SINDEX2' : [255, 255],
         'R770' : [255, 255],
        }

######### SETTING DIRECTORIES #########

workdir = r'D:\TEMP\CRISM_Selected_Datasets\2008_067\Selected'
savepath = r'D:\TEMP\CRISM_Selected_Datasets\2008_067' + '/processed/' 

if not os.path.exists(savepath):
    os.mkdir(savepath)
    print("Directory " , savepath ,  " Created ")
else:    
    print("Directory " , savepath ,  " already exists")

######### SETTING VARIABLES #########

thr = 10

######### MAIN PROGRAM #########

for subdir, dirs, files in os.walk(workdir): # reading filepaths
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".img"): # opening *.img files
            #print(filepath)
            IMG = filepath
            IMAGE = rio.open(IMG)
            for filename in files:
                filepath = subdir + os.sep + filename
                if filename.endswith(".hdr"):                                                                                           # opening *.hdr files
                    #print(filepath)
                    FHDR = filepath
                    HDR = envi.read_envi_header(FHDR)
                    IDX = [(HDR['band names'][i], HDR['default bands'][i]) for i in range(0, len(HDR['band names']))]                   # Creating index for 'band names' and 'default bands' (channel number)
                    for i in range(len(HDR['default bands'])):                                                                          # for every n channel in default bands
                        if IDX[i][0] in SPIDX and SPIDX.get(IDX[i][0]) == [255, 255]:                                                   # check if the corresponding band names is present in the main dictionary and as a value of [255,255] aka nodata in the main dictionary
                            print('present', IDX[i][0], 'value: ', SPIDX[IDX[i][0]])
                            BANDor = (IMAGE.read(i+1, masked=True)).astype('float64')  
                            savename = IDX[i][0] + '_original'                                                                          # define savename
                            imsave(savepath + savename + '.png', BANDor.astype('uint8'))                                                  # save original image as png
                            BANDor.dump(savepath + savename)                                                                              # save original dataset as dump
                            SPIDXor[IDX[i][0]] = IMAGE.read(i+1, masked=True),                                                          # update original dictionary                            
                            BANDth = np.ma.masked_where(BANDor == 255, BANDor)                                                                # mask the band exluding nodata
                            BANDth = np.ma.filled(BANDth, np.nan)                                                                           # converting nodata to np.pnan
                            Bpc = np.nanpercentile(BANDth, thr)                                                                           # computing the percentile
                            print(thr,'th percentile for index', IDX[i][0],'is: ', Bpc, '\n')
                            BANDth = np.ma.masked_inside(BANDth, 1, Bpc)                                                                    # mask the band using the percentile
                            BANDth = np.ma.filled(BANDth, np.nan)                                                                           # converting the exluded values from the percentile to np.nan
                            SPIDX[IDX[i][0]] = BANDth,                                                                                    # updating the dictionary
                            savename = IDX[i][0] + '_thresholded'                       
                            imsave(savepath + savename + '.png', BANDth.astype('uint8'))                                                  # save the thresholded image as png
                            np.save(savepath + savename, BANDth)                                                                          # save the array as a numpy file
                            bol = np.isfinite(BANDth)
                            SPIDXbol[IDX[i][0]] = bol,

                        elif IDX[i][0] in SPIDX and SPIDX.get(IDX[i][0]) != [255, 255]:
                             print('Present but already computed', IDX[i][0])
                         
                                              
def save_images(array):
    pass
    