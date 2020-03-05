import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tkinter import Tk,filedialog
import os
import csv

root = Tk()
root.withdraw()
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

with open('G:\Il mio Drive\BREMEN_PHD\Python_scripts\CRISM_ML_Data_Analysis_Tool/SpIndex_Minerals.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
#print(data)

#SPECTRAL_INDEX2 = ('OLINDEX3', 'LCPINDEX2', 'HCPINDEX2', 'BD2100_2', 'BD1900_2', 'BDI1000VIS', 'D2300', 'SINDEX2', 'R770')

SPECTRAL_INDEX, MINERAL_LIST = zip(*data)


#global data_type
global file_type
global imgs
#data_type = 'Thresholded'
file_type = 'npy'

# read the 9 index-images (either original or the thresholde)
def read_single_image(SPECTRAL_INDEX, data_type, file_type, i):
    if file_type == 'dump':
        print('dump')
    elif file_type == 'npy':
        img = np.load(PATH + '/' + SPECTRAL_INDEX[i] + '_' + data_type + '.' + file_type, allow_pickle=False)
    elif file_type == 'png':
        img = cv.imread(PATH + '/' + SPECTRAL_INDEX[i] + '_' + data_type + '.' + file_type)
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
    for index in SPECTRAL_INDEX:
        img = read_single_image(file_type, SPECTRAL_INDEX, data_type='Thresholded')
        ser = array_to_series(img, i)
        #assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)
    return(df)




def define_class_array(index_array):
    return index_array.astype(bool)

def Y(mineral_name, index_filename):
    index_array = read_single_image(SPECTRAL_INDEX,  file_type, data_type='Thresholded_bool')
    #arr = define_class_array(index_array)
    ser = array_to_series(mineral_name, index_array)
    return (ser)

# =============================================================================
# def Y(mineral_name, index_filename=""):
#     #we know there is direct map from index to mineral
#     image_array_index = read_images(image_type="bol")
#     arr = define_class_array(image_array_index)
#     ser = array_to_series(mineral_name, arr)
#     return ser
# =============================================================================

dfX = X(PATH)
dfX.to_csv("index_features-csv")

for i in range(len(MINERAL_LIST)):
    dfY = Y(MINERAL_LIST[i], SPECTRAL_INDEX[i])
    #assert len(dfY) == len(dfX)
    dfY.to_csv(mineral.csv)
