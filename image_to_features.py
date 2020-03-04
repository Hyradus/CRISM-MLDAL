import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
path= r'D:\TEMP\CRISM_Selected_Datasets\2008_067\processed'
SPECTRAL_INDEX = ('OLINDEX3', 'LCPINDEX2', 'HCPINDEX2', 'BD2100_2', 'BD1900_2', 'BDI1000VIS', 'D2300', 'SINDEX2', 'R770')
data_type = 'Thresholded'
file_type = 'npy'

# read the 9 index-images (either original or the thresholded)img
def read_single_image(SPECTRAL_INDEX, data_type, file_type):
    if file_type == 'dump':
        print('dump')
    elif file_type == 'npy':
        img = np.load(path + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type, allow_pickle=False)
    elif file_type == 'png':
        img = cv.imread(path + '/' + SPECTRAL_INDEX + '_' + data_type + '.' + file_type)
    plt.imshow(img)
    return (img)

def read_images(data_type, file_type):
    imgs = []
    for indx in SPECTRAL_INDEX:
        img = read_single_image(SPECTRAL_INDEX=indx, data_type=data_type, file_type=file_type)
        imgs.append(img)
    return (imgs)

def array_to_series(SPECTRAL_INDEX,imgs):
    vec = imgs.flatten()
    ser = pandas.Series(data=vec,name=SPECTRAL_INDEX)

def merge_series_to_df(df,ser):
    assert df is not None
    df[ser.name] = ser
    return df

def X(data_type, file_type, directory=""):
    image_arrays = read_images()
    df = pandas.DataFrame()
    for image_array in image_arrays:
        ser = array_to_series(image_array)
        assert len(ser) == len(df)
        df = merge_series_to_df(df,ser)

def define_class_array(index_array):
    return index_array.astype(bool)

def Y(mineral_name, index_filename=""):
    #we know there is direct map from index to mineral
    image_array_index = read_images(image_type="booleano")
    arr = define_class_array(image_array_index)
    ser = array_to_series(mineral_name, arr)
    return ser

dfX = X("images_directory/")
dfX.to_csv("index_features-csv")

for mineral in list_minerals:
    dfY = Y()
    assert len(dfY) == len(dfX)
    dfY.to_csv(mineral.csv)

    imm = cv.imread(path + '/HCPINDEX2_thresholded.png')
    cv.imshow('image', imm)
    cv.waitKey(0)
