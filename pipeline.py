# Visualizing the OpenSense Kinematics and parsing that data into train/dev/test sets
import numpy as np
from numpy import genfromtxt
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
import itertools
import pandas as pd
import logging as log 

# connect to data from google drive
drive.mount('/content/drive/')
rawkin_data_folder = "drive/My Drive/Motion Prediction/Data_collection/Opensense_kinematics/"
rawaccel_data_folder = "drive/My Drive/Motion Prediction/Data_collection/Processed_quaternions/NPY/"
save_folder = "drive/My Drive/Motion Prediction/Data_collection/Datasets/"
# Note - setting skipheader to 8, since everything is zeroed for the first row in the .mot files
kin_data_skipheader=8

"""
For lists of lists, flatten elements into one list
"""
flatten = lambda t: [item for sublist in t for item in sublist]

def get_data_files(data_folder_kin=rawkin_data_folder, subjects=None, days=None
                   , kin_inds = [7,10,15,18], include_IMU=False
                   , data_folder_imu=rawaccel_data_folder):
  """
Read data matrices from disk (or cloud).
@param data_folder: path to data folder. This dir has subdir's for subjects, 
  which have subdirs for dates, and then numpy files. 
@param subjects: subjects in `data_folder` to read. `None` means read all. 
@param days: days in `data_folder/subjects` to read. `None` means read all. 
@return all_data: list with elements being 2d numpy arrays
@return: all_data_meta: list with same dimension as `all_data`. Each element is
  a 3-element list [<subject>,<day>,<data_length>,<fname>]
Warning: have skipped first row of kinematic & IMU data because the first row 
of .mot files all seem to be zeros before jumping up to some other value
  """
  all_subjects = True if subjects is None else False
  all_days = True if days is None else False

  all_data, all_data_meta = [], []
  if all_subjects: subjects = os.listdir(data_folder_kin)

  for i, subject in enumerate(subjects):
    print("Subject {}".format(subject))
    if all_days: days_lst = os.listdir("{}{}".format(data_folder_kin, subject))
    else: days_lst=days[i]
    days_lst.sort()

    print(subject, days_lst)
    for day in days_lst:
      print("Day {}".format(day))
      files = os.listdir(("{}{}/{}".format(data_folder_kin, subject, day)))
      for f in files:
        fname_kin ="{}{}/{}/{}".format(data_folder_kin, subject, day, f)
        data = genfromtxt(fname_kin, delimiter='\t'
                , skip_header=kin_data_skipheader, usecols=kin_inds)
        if include_IMU:
          fname_IMU="{}{}/{}/{}_IMU.npy".format(data_folder_imu
                                            , subject, day, f[3:-4])
          data_IMU = np.load(fname_IMU)[1:]
          data  = np.concatenate((data, data_IMU), axis=1)

        all_data.append(data)
        all_data_meta.append([subject, day, len(all_data[-1]), fname_kin])
  return all_data, all_data_meta

# Read column labels from kinematics that will be included
def read_column_labels(fname="{}{}/{}/ik_0.mot".\
              format(rawkin_data_folder, 'S1','2020-04-18')
              , kin_inds=[7,10,15,18]):
  """
Get the column labels from .mot files
@param fname: full path. Defaults to a random .mot file
@param kind_ids: column indices from .mot file. Use `None` for all indices
"""
  return np.loadtxt(fname, dtype=str, delimiter='\t', skiprows=6
                      , max_rows=1, usecols=kin_inds)
  
def make_IMU_labels(sensor_list=['Lshank','Lthigh','Rshank','Rthigh','Pelvis']
                    ,sens_comp=['accel','gyro','mag']
                    ,dirs=['x','y','z']):
  """
Generate an IMU label for every combination of the input list.
@param sensor_list, sens_comp, dirs: list of strings to be combined
@returns: list with format "<a>_<b>_<c" where a,b,c are elements of the 1st, 
  2nd, and 3rd lists respectively.
  """
  cross_product = list(itertools.product(sensor_list, sens_comp, dirs))
  return ["{}_{}_{}".format(a,b,c) for a,b,c in cross_product]


def chunk_data(data, window_s=10, keep_last_samples=False, rate=60):
  """
Split dataset into chunks of equally-sized, non-overlapping windows. If the data
cannot be equally chunked, the end is included only if `keep_last_samples`=True
@param data: data matrix with shape (n_samples,n_features). The data is at 
  <rate>Hz (that is <rate> rows per second).
@param window_s: window size in seconds. 
@param keep_last_samples: Bool. If true, the last element returned is an array 
  of the leftover data points that could not be fit into windows size `window_s`
@param rate: number of samples per second. E.g. 60Hz.
@return: A list of arrays that are the same as `data` but with window_s seconds
  of data. 
  """
  N_windows = len(data) // (window_s*rate)
  if N_windows < 1:
    ret = [] 
  else: 
    ret = np.split(data[:(N_windows*window_s*rate)], N_windows)
  if keep_last_samples: 
    ret.append(data[(N_windows*window_s*rate):])
  return ret

def chop_by_motion(data, window_s=10, mot_thresh = 35, mot_indcs=[0,1,2,3]
                   , rate=60):
  """
Split a file into chunks that have significant motion. First split data into 
chunks of size `window_ts` by calling `chunk_data()`. Throw away any chunks that
don't have motion. If there are consecutive chunks with motion, then join them 
together. 
@param data: array with shape (n_samples, n_features). The data is at 
  <rate>Hz (that is <rate> rows per second).
@param window_ts: Size of window 
@param mot_thres: difference between max and min value in `window_ts` size 
  window. If any featurue exceeds this range, then keep the data. 
@param mot_indcs: the feature indices that are tested for motion.
@param rate: number of samples per second. E.g. 60Hz.
@return: list of data chunks. Each element has shape[1] the same as `data`. 
  """
  chunked_data = chunk_data(data, window_s, rate)
  # bool array for whether there is motion in each window
  mot_data_only = [c[:,mot_indcs] for c in chunked_data]
  is_chunked_data_gt_thresh = [abs(np.max(d, axis=0)-np.min(d, axis=0)).max() 
                              > mot_thresh for d in mot_data_only 
                               if d.shape[0]>0]

  ret, streak = [], []
  # loop over `chunked_data`. Concat consec elms with motion; put in `ret`.
  for i in range(len(is_chunked_data_gt_thresh)):
    if not is_chunked_data_gt_thresh[i]:
      if len(streak) > 0:
        ret.append( np.concatenate(streak))
        streak=[]
    else:
      streak.append(chunked_data[i])
  # if the streak is still alive, then append it  
  if len(streak) > 0: ret.append( np.concatenate(streak) )

  return ret

def split_timeseries(data, input_samples, output_samples
                     , train_step_samples, training=False, no_overlap=False):
  """
Turn a `data` input matrix into a series of (x,y) pairs, where x is a sample 
with lenth `input_samples` and y is the label with length `output_samples`. 
Input numbers refer to number of steps along axis 0 of `data`. If calling 
functions take time in seconds, it must make a conversion. 

@param data: Continuous time series array with shape (n_points, n_features). 
@param input_samples: int. Number of samples in each input
@param output_samples: int. Number samples in each output.
@param train_step_samples: int. How far to slide the window for next sample. Eg
  if the last window began at n_sample=1000, and train_step_samples=500, then 
  the next window begins at 1500.
@param no_overlap: bool. If True, then assert that no data is repeated in any 
  sample-label pair. If `train_step_samples` is too short and causes repeated 
  samples, then start the next window at the end of the last window
@param training: does nothing at the moment. 
  """
  assert(train_step_samples <= input_samples+output_samples)
  if no_overlap: 
    train_step_samples = input_samples+output_samples
  
  n_points, n_features = data.shape
  n_windows = (n_points-input_samples-output_samples) // train_step_samples 
  if n_windows<1: return [], []
  
  # generate array of indices for slicing the windows
  in_start_indices = np.linspace(0, (n_windows-1)*train_step_samples
                                 , n_windows, dtype=int)
  in_stop_indices = in_start_indices+input_samples
  out_start_indices = in_stop_indices.copy()
  out_stop_indices = out_start_indices+output_samples

  # generate data
  in_samples = np.array([data[in_start_indices[i]:in_stop_indices[i]] 
                for i in range(n_windows)])
  out_samples = np.array([data[out_start_indices[i]:out_stop_indices[i]] 
                for i in range(n_windows)])
  
  # ensure the list>>array conversion actually worked (it only does if each 
  # individual window was the same length)
  assert len(in_samples.shape)==len(out_samples.shape)==3
  return in_samples, out_samples 
;

def process_data(dataset_name, subjects, train_days, test_days, rate, input_len
                 , output_len, train_step, val_size, kin_inds, include_IMU
                 , window_s=10, mot_thresh=35, mot_indcs=[0,1,2,3]
                 , data_folder_kin=rawkin_data_folder, save_bouts_only=False):
  """
Process data set into train, test, eval sets. Print activity on the relative 
size of each. 
@param dataset_name: string to identify the folder
@param subjects: list of subject strings, or None to include all 
@param train_days: List of lists. Element i is the days for `subjects[i]`. Set 
  to None for all days. If the days are invalid, there will be an error. 
@param test_days: List of lists. Same as `train_days` but for test data.
@param input_len= 
@param val_size: percent of train/val set that is val. 
@param rate: number of samples (rows of the array) per second. If 60Hz, is 60.


@param save_bouts_only: Bool. If true, then get data that has motion and save it
  but don't split into train and test sets. If False, then don't save it and do 
  split into train and test sets. 
"""
  # directory for this dataset
  dataset_fn = "{}{}_IMU{}_{}_{}_{}/"\
      .format(save_folder, dataset_name, include_IMU, str(input_len)
      , str(output_len), str(train_step))
  print("Writing data to: {}".format(dataset_fn))
  if not os.path.exists(dataset_fn): 
    os.makedirs(dataset_fn)
  
  # Read column labels from kinematics
  labels = read_column_labels("{}{}/{}/ik_0.mot".\
          format(rawkin_data_folder, subjects[0], train_days[0][0]))
  if include_IMU: 
    labels = list(labels) + make_IMU_labels()
  print("Reading data labels: {}".format(labels))

  # Get train and test data from files 
  train_data_all, _ = get_data_files(data_folder_kin, subjects=subjects
                                 ,days=train_days, include_IMU=include_IMU)
  
  test_data_all, _ = get_data_files(data_folder_kin, subjects=subjects
                                ,days=test_days, include_IMU=include_IMU)
  
  # Filter data to list of bouts that have motion. Shuffle the order.
  print("Chopping ", subjects, train_days)
  train_data_motion = [chop_by_motion(d, window_s=window_s
                      , mot_thresh=mot_thresh, mot_indcs=mot_indcs, rate=rate)
                    for i, d in enumerate(train_data_all)]
  train_data_motion = flatten(train_data_motion)                   
  np.random.shuffle(train_data_motion)

  print("doing ", subjects, test_days)
  test_data_motion = [chop_by_motion(d, window_s=window_s
                      , mot_thresh=mot_thresh, mot_indcs=mot_indcs, rate=rate)
                    for d in test_data_all]
  test_data_motion = flatten(test_data_motion)
  np.random.shuffle(test_data_motion)

  # Split test dataset into test and validation. Need to check how many samples
  # are in each list el of `train_data_motion` to get the right split percentage
  train_cumsum = np.cumsum(np.array([t.shape[0] for t in train_data_motion]))
  
  break_point = train_cumsum[-1] *val_size
  val_set_last_indx = np.argwhere(train_cumsum > break_point )[0,0] 

  val_data_motion = train_data_motion[:val_set_last_indx]
  train_data_motion = train_data_motion[val_set_last_indx:]

  # Save motion bouts, with option to stop there if `if save_bouts_only`=True
  np.save("{}train_data_motion.npy".format(dataset_fn), train_data_motion)
  np.save("{}val_data_motion.npy".format(dataset_fn), val_data_motion)
  np.save("{}test_data_motion.npy".format(dataset_fn), test_data_motion)
  if save_bouts_only:
    return 
  
  # If `not save_bouts_only`, continue. Split data into windows, and save
  input_samples=int(input_len*rate)
  output_samples=int(output_len*rate)
  train_step_samples=int(train_step*rate)
  # generate list of tuples (<x>, <y>) where <x> is a list of x windows with len
  # input_samples, and <y> is a list of y windows with len output_samples
  train_splits = [split_timeseries(d, input_samples=input_samples
          , output_samples=output_samples, train_step_samples=train_step_samples
          , no_overlap=False) for d in train_data_motion]
  train_x = np.array(flatten([d[0] for d in train_splits]))
  train_y = np.array(flatten([d[1] for d in train_splits]))

  val_splits = [split_timeseries(d, input_samples=input_samples
          , output_samples=output_samples, train_step_samples=train_step_samples
          , no_overlap=False) for d in val_data_motion]
  val_x = np.array(flatten([d[0] for d in val_splits]))
  val_y = np.array(flatten([d[1] for d in val_splits]))

  test_splits = [split_timeseries(d, input_samples=input_samples
            , output_samples=output_samples, train_step_samples=train_step_samples
            , no_overlap=False) for d in test_data_motion]
  test_x = np.array(flatten([d[0] for d in test_splits]))
  test_y = np.array(flatten([d[1] for d in test_splits]))
  
  print("{} train samples\n{} val samples\n{} test samples"\
        .format(len(train_x), len(val_x), len(test_x)))
  print("Val set is {:.3f}% of train+val"\
        .format(len(val_x)/(len(val_x)+len(train_x))))
  print("Saving datasets")
  
  print(train_x.shape, val_x.shape, test_x.shape)
  # save the datasets
  fnames = ['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y']
  objs = [train_x, train_y, val_x, val_y, test_x, test_y]
  for i in range(len(objs)):
    np.save("{}{}".format(dataset_fn, fnames[i]), objs[i])
  print("Finished pipeline")
  
  return

if __name__ == "__main__":
	train_days = [
              ['2020-04-19','2020-04-20','2020-04-21','2020-04-23'],
              ['2020-04-30','2020-05-01','2020-05-04','2020-05-06'],
              ['2020-05-14','2020-05-15'],
              ]
	test_days =[
	            ['2020-04-24'],
	             ['2020-05-07'],
	            ['2020-05-16'],
	            ]
	subjects = [
	            'S1',
	            'S2',
	            'S3'
	            ]


	rate=60                  # Hz
	input_len = 2.0          # sec
	output_len = 1.0         # sec
	train_step = 0.5         # sec
	val_size = 0.13          # %

	# data to include
	kin_inds = [7,10,15,18]
	include_IMU = False # add IMU data to the files
	dataset_name='all-S3-data'

	process_data(dataset_name, subjects, train_days, test_days, rate, input_len
	            , output_len, train_step, val_size, kin_inds, include_IMU
	            , window_s=10, mot_thresh=35, mot_indcs=[0,1,2,3])
