from google.colab import drive
drive.mount('/content/drive')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, kurtosis
from scipy.stats import f_oneway, ttest_ind

%cd Cerebra_aatlas_python
!pip install 

from cerebra_atlas_python.plotting.plotting_3d import Plots3D

import statistics
!pip install nilearn
from nilearn.plotting.cm import cold_hot
from nilearn import datasets, plotting # Import datasets from nilearn
from nilearn.image import new_img_like # Import new_img_like from nilearn.image

#!git clone https://github.com/kdotdot/cerebra_atlas_python.git
!git clone https://github.com/marowakamalaldeen/Cerebra_aatlas_python.git
!pip install -r Cerebra_aatlas_python/requirements.txt
!pip install --editable Cerebra_aatlas_python/
!apt-get update
!apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y --fix-missing