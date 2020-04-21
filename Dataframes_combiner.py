# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:52:53 2020

@author: HyraN
"""

import shutil
import os
import pandas as pd
import glob
from tkinter import Tk,filedialog

root = Tk()
root.withdraw()
PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Please select the working folder:")
print('Working folder:', PATH)

def MergeReadCsv():
    os.chdir(PATH)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    CombData = pd.concat([pd.read_csv(f) for f in all_filenames ], ignore_index=True)
    #export to csv
    ln = len(all_filenames)
    savefolder = "combined_"+str(ln)+'_classes'
    if os.path.exists(savefolder):
        shutil.rmtree(savefolder)
        os.mkdir(savefolder)
    else:
        os.chdir(PATH)
        os.mkdir(savefolder)
    CombData.to_csv(savefolder+'/'+str(ln)+'_classes_combined.csv', encoding='utf-8-sig',index = False)
    # CombData = pd.read_csv(savefolder+"/combined_csv.csv",
    #                             parse_dates=['data'],
    #                             index_col=['data'])
    return(CombData)

comb_feat = MergeReadCsv()

#comb_class = MergeReadCsv()
# savefolder = "combined"
# comb_class.to_csv(savefolder+"/classes_combined.csv", encoding='utf-8-sig')
