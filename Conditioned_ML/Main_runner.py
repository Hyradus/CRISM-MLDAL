# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:14:42 2020

@author: HyraN
"""


import os
import glob

def band_extractor(rootdir):
    
    datasets = [f.path for f in os.scandir(rootdir) if f.is_dir()]
    folders = []
    for paths in datasets:
        subfolder = [f.path for f in os.scandir(paths) if f.is_dir()]
        folders.append(subfolder)
        print(subfolder)
       
    for i in range(len(folders)):
        command = "Multiple_Band_extractor.py" + ' --wdir ' + folders[i][0]
        os.system(command)
        print('Extracting bands in ', datasets[i], ' folder')


def target_features_creator(rootdir, cfg):
    extrdir = rootdir
    print(extrdir)
    datasets = [f.path for f in os.scandir(extrdir) if f.is_dir()]
    extracted_folder = []
    
    for paths in datasets:
        subfolder = [f.path for f in os.scandir(paths) if f.is_dir()]
        extracted_folder.append(subfolder)
        print(subfolder)
        
    
    for i in range(len(extracted_folder)):
        file_folder = extracted_folder[i][0]+'\\'+"Extracted\\"
        print(file_folder)
        command = "Features-Target_creator.py" + ' --wdir ' + file_folder + ' --cfg ' +  cfg 
        #print('Executing: ', command())
        os.system(command)
        print('Creating features and targets in ', extracted_folder[i], ' folder')

# def multi_trainer(rootdir, cfg):
#     extrdir = 'D:\Mars_CRISM_DATA\Datasets'
#     datasets = [f.path for f in os.scandir(rootdir) if f.is_dir()]
#     extracted_folder = []
    
#     for paths in datasets:
#         subfolder = [f.path for f in os.scandir(paths) if f.is_dir()]
#         extracted_folder.append(subfolder)
#         print(subfolder)
        
    
#     for i in range(len(extracted_folder)):
#         ext = 'csv'
#         os.chdir()
#         file_folder = extracted_folder[i][0]+'\\'+"Extracted\\Processed\\"
#         extracted_bands = [i for i in glob.glob('*d.{}'.format(ext), recursive=True)]
#         fea = file_folder + ''
#         print(file_folder)
#         command = "ML_Multiple-single_class_trainer.py" + ' --wdir ' + file_folder + ' --cfg ' +  cfg + ' --fea ' + fea + ' --cla ' + cla)
#         os.system(command)
#         print('Creating features and targets in ', extracted_folder[i], ' folder')


        
def main(rootdir, cfg):
    band_extractor(rootdir)
    target_features_creator(rootdir, cfg)
    

rootdir = 'D:\Mars_CRISM_DATA\Datasets'
cfg = "D:/Mars_CRISM_DATA/SpIdx_DetectedPhase.csv"
main(rootdir, cfg)