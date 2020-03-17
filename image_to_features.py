import pandas
import numpy as np
import cv2 as cv
from tkinter import Tk,filedialog
import os
import csv

root = Tk()
root.withdraw()
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

# reading config file for minerals
cfg = filedialog.askopenfilename(parent=root,initialdir=os.getcwd(),title="Please select spectral configuration file:", filetypes= (('csv files', '*.csv'), ('all files', '*.*)))')))
print('Spectral configuration file selected:', cfg)
with open(cfg, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

SPECTRAL_INDEX, MINERAL_LIST = map(list,zip(*data))


#global data_type
global file_type
global imgs
#data_type = 'Thresholded'
file_type = 'npy'

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
    ser = pandas.Series(data=vec, name=SPECTRAL_INDEX[i])
    return(ser)

def merge_series_to_df(df,ser):
    assert df is not None
    df[ser.name] = ser
    return (df)

def X(PATH):
    df = pandas.DataFrame()
    for i in range(len(SPECTRAL_INDEX)):
        img = read_single_image(SPECTRAL_INDEX[i], file_type, data_type='Thresholded')
        ser = array_to_series(img, i)
        #assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)
    return(df)

def Y(mineral_name, index_filename):
    index_array = read_single_image(SPECTRAL_INDEX=index_filename, file_type=file_type, data_type='Thresholded_bool')
    #arr = define_class_array(index_array)
    ser = array_to_series(img=index_array, i=SPECTRAL_INDEX.index(index_filename))
    return (ser)

dfX = X(PATH)

savepath = os.path.join(PATH + '/features/')
fename = os.path.join(savepath + 'index_features.csv')
dfX.to_csv(fename, index=False)

DfY = pandas.DataFrame()
for i in range(len(MINERAL_LIST)):
    dfY = Y(mineral_name=MINERAL_LIST[i], index_filename=SPECTRAL_INDEX[i],)
    #assert len(dfY) == len(dfX)
    mineral=MINERAL_LIST[i] + '.csv'
    dfY.to_csv(os.path.join(savepath, mineral.strip( )), index=False)


# DfY = pandas.DataFrame()
# for i in range(len(MINERAL_LIST)):
#     dfY = Y(mineral_name=MINERAL_LIST[i], index_filename=SPECTRAL_INDEX[i],)
#     #assert len(dfY) == len(dfX)
#     DfY = DfY.append(dfY)

savepath = os.path.join(PATH + '/features/')
fename = os.path.join(savepath + 'index_classes.csv')
DfY.to_csv(fename, index=False)
