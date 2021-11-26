# ARIA
Automated ROI-based Image Analysis

## About
This is a python script to enable composition analysis for IHC slide images. Especially designed for human thrombi extracted by endovascular therapy for ischemic stroke patients. This script specifically performes the following tasks:

1. Automatically read slide formats (.svs, .tiff..)
2. [Optional] Crops a user-selected area of the whole-slide image
3. Automatically draws a contour for the thrombus (contour can be adjusted by the user)
4. Automatically sets a threshold using Otsu's automatic thresholding
5. [Optional] Threshold can be manually adjusted by the user
6. Saves intermediate image files
7. Creates a .csv file containing the processed results (both automatic and fixed thresholding)

# Installing ARIA
Using a virtual environment is strongly recommended. The script requires the following libraries:

* Numpy (of course)
* OpenCV2 (processing images and provides GUI)
* Scikit-Image (uses rgb2hed and hed2rgb for color deconvolution) 
* OpenSlide (reading slide formats)

We will use conda virtual environment in this guide.

## Install OpenSlide
Openslide should be installed on your system. Google install openslide on "Windows / Mac" for further instructions.
https://openslide.org/download/

## Install conda
Google installing conda or use the following official instructions to install conda
https://docs.anaconda.com/anaconda/install/

## Create conda venv
use whatever name you want (instead of ARIA) and version of > python 3.8 is recommended.
```
conda create --name ARIA python=3.8
```

## Activate conda env
```
conda activate ARIA
```

## Install necessary packages
### Install numpy
```
conda install numpy
```

### Install OpenCV2
```
conda install -c conda-forge opencv
```

### Install Scikit-image
```
conda install scikit-image
```

### Install Openslide-python
You **should have installed openslide** on your system before this python interface.
Refer to the official guide on OpenSlide : https://openslide.org/download/

```
conda install -c bioconda openslide-python
```
⚠️ If conda installation fails to install the package (no package error) then try pip!
```
pip install openslide-python
```


## Clone or download aria.py
Simply download the `aria.py` file or use git clone https://github.com/jnheo-md/aria.git


# Using ARIA
Use ARIA with the following command on the terminal.

```
python aria.py [skip-crop(optional)] [manual-threshold(optional)] [filename]
```

#### Example code
An example would be the following:
```
python aria.py skip-crop FILENAME.svs
```

## Process
Image processing steps that are provided by ARIA are the following:

1. Automatically read slide formats (.svs, .tiff..)
2. [Optional] Crops a user-selected area of the whole-slide image
3. Automatically draws a contour for the thrombus (contour can be adjusted by the user)
4. Automatically sets a threshold using Otsu's automatic thresholding
5. [Optional] Threshold can be manually adjusted by the user
6. Saves intermediate image files
7. Creates a .csv file containing the processed results (both automatic and fixed thresholding)

## Options
Two options are available : Skip-crop and manual-threshold

### skip-crop option
This skips the cropping step of the processing pipeline, which is the step number 2 of the above 'Process'.

#### Example code
```
python aria.py skip-crop FILENAME.svs
```

### manual-threshold option
This option introduces an additional step where the user can adjust the threshold manually.

#### Example code
```
python aria.py manual-threshold FILENAME.svs
```

Of course, both options can be used together.

#### Example code
```
python aria.py skip-crop manual-threshold FILENAME.svs
```


# Contact
Contact author JoonNyung Heo at jnheo@jnheo.com!
