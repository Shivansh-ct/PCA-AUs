# PCA-based Automated Facial Expressions Encoding
Code for the paper "A PCA-Based Keypoint Tracking Approach to Automated Facial Expressions Encoding" accepted at International Conference on Pattern Recognition and Machine Intelligence, 2023.  

## Installation
Required packages and versions of matlab is listed in [matlab versions](github.com/Shivansh-ct/PCA-AUs/matlab_versions.txt) and python3 are listed in [python versions](github.com/Shivansh-ct/PCA-AUs/python_versions.txt). The Facial Landmark Frontalization algorithm [paper](https://ieeexplore.ieee.org/document/9190989) [code](https://github.com/bbonik/facial-landmark-frontalization) is used in preprocessing. The original matlab code for larsen algorithm was obtained from [source](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3897)  
## Datasets
The datasets can be obtained at [CK+](http://www.jeffcohn.net/Resources/), [DISFA](http://mohammadmahoor.com/disfa/) and [BP4D-Spontaneous](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)  
## Code Usage
**data_and_feature_creation.ipynb** is used to create the final features from the datasets.  
### PCA_train  
contains the code for PCA training, visualizations and creating the comparison graphs.
**PCA_train** is used to train PCA model on the three datasets and create visualization. 
### PCA_vs_FACS
contains the code for creating comparison graph of PCA AUs, pure AUs, and comb AUs.



# Contents:
```tree
├── PCA_train                                      [Directory: PCA training, visualizations etc.]
│   ├── helper.py                   [functions for feature generation, metrics and face morphing] 
│   ├── pca_train.py                        [Training the PCA on any dataset]
│   ├── pca_train_report.py                                [Generates the train-test table at 95% Train VE]
│   ├── pca_train_report_plot.ipynb  [Plots the train-test plot of variance explained]
│   ├── pca_train_report_plot.py  [compile the metrics performance for train and test from one to total components]
│   ├── pca_train_report_statistics.ipynb  [generates p-value for the significance test]
│   └── pca_train_report_statistics.py       [used for significance test]
└── PCA_vs_FACS                                        [Directory: compares PCA AUs, pure AUs, and comb AUs]
│   ├── choldelete.m                    [helper function for LARSEN algorithm]
│   ├── cholinsert.m              [helper function for LARSEN algorithm]
│   ├── larsen.m                [LARSEN algorithm to generate encodings] 
│   ├── pca_vs_facs_plot.ipynb   [plot graphs for comparing PCA AUs, pure AUs, and comb AUs] 
│   └── pca_vs_facs_plot.m                               [compile the metrics performance for all AUs from one to total components] 
│   data_and_feature_creation.ipynb                                  [code to generate features from facial keypoints]
│   matlab_versions.txt                [Environment file for the required version of libraries]
└── python_versions.txt                [Environment file for the required version of libraries]