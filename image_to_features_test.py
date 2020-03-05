import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tkinter import Tk,filedialog
import os

root = Tk()
root.withdraw()
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

SPECTRAL_INDEX = ('OLINDEX3', 'LCPINDEX2', 'HCPINDEX2', 'BD2100_2', 'BD1900_2', 'BDI1000VIS', 'D2300', 'SINDEX2', 'R770')

global data_type
global file_type
global imgs
data_type = 'Thresholded'
file_type = 'npy'
def read_single_image(SPECTRAL_INDEX, data_type, file_type):
    if file_type == 'dump':
        print('dump')
    elif file_type == 'npy':
        img = np.load(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type, allow_pickle=False)
    elif file_type == 'png':
        img = cv.imread(PATH + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type)
    #plt.imshow(img)
    return (img)

def merge_series_to_df(df,ser):
    assert df is not None
    df[ser.name] = ser
    return df

df = pandas.DataFrame()
for i in range(len(SPECTRAL_INDEX)):
    img = read_single_image(SPECTRAL_INDEX=SPECTRAL_INDEX[i],data_type=data_type, file_type=file_type)
    vec = img.flatten()
    ser = pandas.Series(data=vec, name=SPECTRAL_INDEX[i])
    #assert len(ser) == len(df)
    df[ser.name] = ser
