# CRISM-MLDAL - Machine Learning Data Analysis Tool

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

## Executing

1) Edit SpIndex_Minerals.csv and add/delete the Spectral indexes and relative Minerals.
2) Use SpIdx_to_Dicts.py to generate all extracted files.
    Simply run the script and browse in the popup window to the *root* folders where all *.imgs and *.hdr*.
    In the next popup dialog select the SpIndex_Minerals.csv file.
    Default threshold is set to 10.
3) Use image_to_features.py to generate the spectral index dataframe in csv format and boolean csv files for each band.
    Run the script and browse in the popup window to the *processed* folder located in the previously selected *root* folder.
    Default selected datasets are:
        - data_type = Thresholded
        - file_type = npy

_________________________________________________________________________________________________________________________
# Detailed description

## Description of Specialized Browse Product Mosaics

These materials are an early release of new analysis products resulting from an ongoing end-to-end upgrade of the CRISM data-processing pipeline. http://crism.jhuapl.edu/msl_landing_sites/2012July_Mosaics/Planetary_Data_Workshop_submit_seelofp1_120430.pdf

CRISM browse products are RGB composites of individual summary parameters (or a grayscale image of a single parameter) chosen to have thematic geological and/or spectral characteristics.

Spectral summary parameters are band math calculations that quantify diagnostic or indicative spectral structure and collectively capture the mineralogical diversity of the surface. Mathematical functions applied to reflectance values at key wavelengths allow the relative depth of particular absorption features (for example) to be quantified, and produce grayscale images indicating the presence or absence of particular phases.

## Summary: 

Further information on: http://crism.jhuapl.edu/msl_landing_sites/index_news.php

# Pipelines
 ## SpIndex_to_Dicts.py pipeline

* Ask for the root working folder that contains all the RGB Composite images in *.IMG+*.hdr format (automatically looks into subfolders) e.g. “Selected_original_dataset”

* Ask for configuration file in *.csv format. This file is composed of two columns:
    * Spectral summary, that correspond to a band/channel name in each image. E.g. OLINDEX3
    * Phases detected, or relative mineral informations.
* Automatically create 3 dictionaries and initialize all with Phase Detected and [255, 255] values (no-data values):
    * SPIDXOR: containing the original dataset
    * SPIDX: containing the original dataset with applied a user-selected threshold.
    * SPIDXBOL: same as SPIDX but containing boolean data.
* A function looks into every folder for *.img file and the corresponding *.hdr and:
    * Read for each *.img the bands (channels) present and check if the band is present into the dictionary then:
    * IF the band is not present, skip to the next band, if none band is present skip to the next *.img.
    * IF the band is present, check if it has [255, 255] values.
        * IF not, skip to the next band.
        * IF yes, read the corresponding data from the *.img and:
    * Store the data in SPIDXOR and save into a dump file and *.png in “processed” subfolder;
    * Apply the user-defined threshold and store the data in SPIDX and save into a numpy file and *.png in “processed” subfolder;
    * Convert thresholded data into boolean and store in SPIDXBOL and save into a numpy file and *.png in “processed” subfolder.
    e.g of configuration file


## image_to_features.py pipeline
* Ask for the root working folder that contains all the RGB Composite images in *.IMG+*.hdr format (automatically looks into subfolders) e.g. “Selected_original_dataset”
* Ask for configuration file in *.csv format, the same of SpIdx_to_Dicts.py. This file is composed of two columns:
    * Spectral summary, that correspond to a band/channel name in each image. E.g. OLINDEX3
    * Phases detected, or relative mineral informations.
* The function defX:
    * Read the image in the format defined the user in “file_type” and type of data defined by the user in “data_type” (original, Thresholded, Thresholded_bool);
    * Convert the images into a series and append to a dataframe in which every column (feature) correspond to a Spectral Summary (Spectral Index) and every row correspond to a pixel;
    * Save the dataframe into “features” subfolder with name “index_features.csv”.
* The function defY:
    * Read the image in the format defined the user in “file_type” and type of data defined by the user in “data_type” (original, Thresholded, Thresholded_bool);
    * For each “Phase detected” – “Mineral”, present in the configuration csv, convert the images into a dataframe series.
    * Save each series into a csv “features” subfolder with the corresponding “Phase detected” – “Mineral” name.


## WIP_ML_single_class.py pipeline

This script is a first attempt to realize a ML model for predicting mineral phases.
Ask for:
    * working folder, this folder is the "features" folder, where all the *csv produced by image_to_features.py are located
    *same configuration files used by previous scripts
    * index_features.csv, this is the files containing all the features to be trained
    * the mineral phase csv to to be tested with the model
Then: 
* remove the nan values
* split the index_features into train and test datasets 
* normalize and encode both datasets
* select the user-defined algorithm (decisional trees or randomforest, other methods not implemented yet)
* train the model
* predict the "Phase detected - Mineral" choosed and print accuracy and log_loss.

## Tests

### Conditioned tests
* Using the complete dataset, the values of accuracy and log loss are always equal to 1.0 for every "Phase detected - Mineral" choosed of the same dataset. This is correct and intended because in this case, the "Phase detected -Mineral" csv are extracted from the same dataset of the index_features. It's a validation attempt.
* The next test was conducted by completely removing a spectral index from the index_features.csv e.g. OLINDEX3, and selecting as "Phase detected - Mineral" the OLINDEX3 corresponding csv. In this case the results of accuracy and log_loss are not 1.0 anymore but values between 0.98 and 0.99 because there are features related to other phase-detected Fe-minerals of the same dataset. Replicating this test with other phase-detected Fe-mineral features e.g. BDI1000VIS or D2300 give similar results.
* The next test was conducted by completely removing all major phase-detected Fe-mineral features (OLINDEX3, BDI1000VIS, D2300) and using one of the "Phase detected - Mineral" corresponding to the removed features and results of accuracy and log_loss are in a range of 0.96-0.98.

### Unconditioned tests

Nest cycle of tests were similar to the conditioned test but using "Phase detected - Mineral" csv files generated by image_to_features.py on a different dataset e.g. different year of image acquisitions. 

In this case the results are slightly different but consistent. In fact, predicting each "Phase detected - Mineral", without removing any feature index, gave results above 0.98/0.99.


## Authors

* **Giacomo Nodjoumi** - *Initial work* - [Hyradus](https://github.com/Hyradus)

* **Carlos H Brandt** - *contributors* - [chbrandt](https://github.com/chbrandt)

## License

GNU General Public License v3.0

