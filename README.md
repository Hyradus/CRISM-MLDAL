# CRISM_ML_Data_Analysis_Tool

Scripts to read, extract, assemble features and mineral classes to train a ML.

## Brief description.

What is a MTRDR? A map-projected Targeted Reduced Data Record (MTRDR) consists of the TER corrected I/F spectral information after map projection and the removal of spectral channels with suspect radiometry (“bad bands”).  See https://bit.ly/2IuEkIT for detailed informations.

MTRDR are available for download from MARS Orbital Data Explorer website: https://bit.ly/339xJNF

Downloaded images consist of different files, the most important are the *.img (data file) and *.hdr (ancillary metadata file).


This repository is composed by:

- a simple configuration file in csv format (SpIndex_Minerals.csv), in which there are two columns, Spectral bands (1st) and Relative mineral (2nd).

- SpIdx_To_Dicts.py: a script that iterate over all files of a user-specified folder and look for every spectral band specified into the configuration file and:
    1) Extract the original data and store it into a dump file and a *.png;
    2) Apply a mask for NaN values, apply a user defined threshold (default is 10) and store into a *.npy array and a *.png;
    3) Convert the masked, tresholded image into a boolean image and store into a *.npy array and a *.png.
    
- image_to_features.py: a script that read files exported with SpIdx_to_Dicts.py, converts them into a series and create a csv pandas dataframe and a boolean csv for each relative mineral specified in the configuration file.
In the pandas dataframe columns are spectral indexes (e.g. OLINDEX3) and rows are the pixels of the image with relative DN values.

## Getting Started

### Prerequisites

To run both of the scripts the following modules are needed to be installed:
- Rasterio 
- spectral
- numpy
- cv2
- pandas

A requirements.txt files is also provided.

If you are using conda
```
conda install -c conda-forge rasterio
conda install -c conda-forge spectral
conda install -c conda-forge opencv
conda install -c conda-forge 
```

### Installing

To install it, just clone the repository and double check the prerequisites.

## Running the tests

In the repository there is an example dataset into the folder "CRISM_dataset" with both extracted data (processed folder) and features data (into features sub-folder)

To test the script, just delete every file inside inside the processed and features folder, then run SpIdx_to_Dicts.py using --argparser or using the filedialog (if no --arguments are passed) and select the Selected_original_dataset folder as a working folder and SpIndex_Minerals.csv as configuration file (root folder).

Then run image_to_features.py double check that a "features" folder is present inside the processed foldeer and select the processed folder and the same SpIndex_Minerals.csv. 

## Authors

* **Giacomo Nodjoumi** - *Initial work* - [Hyradus](https://github.com/Hyradus)

* **Carlos H Brandt** - *contributors* - [chbrandt](https://github.com/chbrandt)

## License

GNU General Public License v3.0

