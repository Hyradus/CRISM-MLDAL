import pandas
import numpy as np
from skimage.io import imread, imsave, imshow
path= r'D:\TEMP\CRISM_Selected_Datasets\2008_067\processed'
spectral_index = ('OLINDEX3', 'LCPINDEX2', 'HCPINDEX2', 'BD2100_2', 'BD1900_2', 'BDI1000VIS', 'D2300', 'SINDEX2', 'R770')
# read the 9 index-images (either original or the thresholded)img
def read_images():
    
    #if file_type == 'dump':
    #    print('dump')
    #elif file_type == '.npy':
    #img=np.load(path + '/' + '/R770' +'_' + image_type + '.' + file_type)
    img=np.load(path + '/R770_thresholded.npy', allow_pickle=False)
    imshow(img)
       
    return (img)


def array_to_series(lbl,arr):
    vec = arr.flatten()
    ser = pandas.Series(data=vec,name=lbl)
    
def merge_series_to_df(df,ser):
    assert df is not None 
    df[ser.name] = ser
    return df

def X(directory=""):
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