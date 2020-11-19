# Visualizing the OpenSense Kinematics and parsing that data into train/dev/test sets
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import natsort as nt
from google.colab import drive
import shutil
import math
import random
import pickle as pkl
import json

# connect to data from google drive
drive.mount('/content/drive/')
rawkin_data_folder = "drive/My Drive/Motion Prediction/Data_collection/Opensense_kinematics/"
rawaccel_data_folder = "drive/My Drive/Motion Prediction/Data_collection/Processed_quaternions/NPY/"
save_folder = "drive/My Drive/Motion Prediction/Data_collection/Datasets/"

import importlib
try: 
  import import_ipynb
except:
  !pip install import_ipynb
  import import_ipynb
pktm_path = "drive/My Drive/Motion Prediction/Code/pipeline_kinematic_to_model"
# pktm = importlib.import_module(pktm_path)
pktm = importlib.reload(importlib.import_module(pktm_path))

def show_data_file_structure(data_folder=rawkin_data_folder):
	"""
Show directory structure of the data files 
args: either `rawkin_data_folder` or `rawaccel_data_folder`
	"""
  files = {}
  subjects = os.listdir(data_folder)
  for subject in subjects:
    date_folders = os.listdir("{}{}".format(data_folder, subject))
    date_folders.sort()
    files[subject] = []
    for date_folder in date_folders:
      data_files = os.listdir(("{}{}/{}".format(data_folder, subject, date_folder)))
      files[subject].append("{} : {}".format(date_folder, len(data_files)))
  print(json.dumps(files, indent=4))


def plot_hists(metrics, bins, xlim, labels, suptitle=""):
	"""
Plot 4 histograms for the distributions of the columns of the data matrix. This
was written for showing the standard deviation
@params metrics: array shape (n_samples, n_features), currently written with 
  n_features=4. 
@param bins: bin size for histogram
@param xlim: 2-element array/tuple for x limit
@param labels: lables for the features, used for axis titles
@param suptitle: figure title
	"""	
  # Do plotting 
  f, axs = plt.subplots(2,2, figsize=(10,10))
  axs=axs.flatten()
  f.suptitle(suptitle)

  for i in range(metrics.shape[1]):
    axs[i].hist(metrics[:,i], bins=bins)
    axs[i].set(title=labels[i], xlim=xlim)
  return axs


if __name__=="__main__":
	## Display data file structure
	print("Data structure - Opensense Kinematics")
	print("Source directory: {}".format(rawkin_data_folder))
	show_data_file_structure(data_folder=rawkin_data_folder)
	print('-'*80)
	print("Data structure - Processed Quaternions")
	show_data_file_structure(data_folder=rawaccel_data_folder)

	# pull all data
	# read data for all subjects
	all_data, all_data_meta =\
  			pktm.get_data_files(data_folder_kin=rawkin_data_folder
                   , subjects=None
                   , days=None
                   )

  	# display the hours per file 
	 arrays = [[a[0] for a in all_data_meta], [a[1] for a in all_data_meta]]
	indx = pd.MultiIndex.from_arrays(arrays, names=('subject', 'date'))
	samples_sizes = [a.shape[0] / (60**3) for a in all_data] 
	df = pd.DataFrame(data=samples_sizes, index=indx, columns=["Hours"])
	print("Hours of data by subject:")
	hours_by_subject = df.groupby(level=0).sum()
	print(hours_by_subject)

	print('-'*80, "\nHours of data by subject / day:")
	hours_by_subject_day = df.groupby(level=[0,1]).sum()
	print(hours_by_subject_day )

	pcnt_hours_by_subject = (hours_by_subject_day / hours_by_subject).round(3)
	print('-'*80, "\nPcnt of days total subject's hours by day:")
	print(pcnt_hours_by_subject)

	# histogram of minutes per file 
	rate=60 # 60hz
	mins_per_file = [a.shape[0]/(rate*60) for a in all_data]
	f, axs = plt.subplots(1,1)
	axs.hist(mins_per_file)
	axs.set(title="Minutes of data per file", xlabel="minutes",ylabel="count")
	
	# chunking the data to 10s windows 
	# Get 'all_data', which is an array of numpy arrays, and break each into chunks 
	# so that we have a doubly-nested list where the elements are numpy arrays.
	all_data_chunks = [pktm.chunk_data(d, window_s=10) for d in all_data]
	all_data_chunks_flat = []
	for l1 in all_data_chunks:
	  for l2 in l1:
	    all_data_chunks_flat.append(l2)

	all_data_chunks_flat_stds = np.array(
	    [np.std(d, axis=0) for d in all_data_chunks_flat]
	)

	# plot histograms of standard deviatins 
	# Do plots
	labels = pktm.read_column_labels()
	plot_hists(all_data_chunks_flat_stds
	           , bins=100
	           , xlim=None
	           , labels=labels
	           , suptitle="Std Deviation; No Zoom, 100 bins")

	plot_hists(all_data_chunks_flat_stds
	           , bins=1000
	           , xlim=[0,5]
	           , labels=labels
	           , suptitle="Std Deviation; Zoom to [0,5], 1000 bins")
	;