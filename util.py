# !pip install -q h5py pyyaml
#!pip install -q tensorflow==2.0.0-beta1

# Load dependencies and mount google drive, go to datasets folder
from tensorflow import keras
import os
from google.colab import drive
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skt


from google.colab import drive
drive.mount('/content/drive')
dataset_name = 'tst-pipeline_IMUFalse_2.0_1.0_0.5/'
data_folder = "drive/My Drive/Motion Prediction/Data_collection/Datasets/"
save_folder = "drive/My Drive/Motion Prediction/Code/Results/"
model_folder = "drive/My Drive/Motion Prediction/Code/Models/saved_models/"
plot_folder = "drive/My Drive/Motion Prediction/Code/Plots/"

def load_data(dataset_path, flatten_x_dim=True, flatten_y_dim=False
              , merge_all=False, merge_train_val=False):
  """
@param dataset_name: 
@param split_out_val: Bool. If true, merge the train and validation sets, which
  would be appropriate if doing k-fold CV. If False, then there will be a 
  designated validation set that doesn't change 
@return: 6-tuple (train_x, train_y, val_x, val_y, test_x, test_y, )
  Each dataset has shape (n_windows, n_samples_p_window, n_features)
  `train_x` and `train_y` have same `n_windows` (dim 0). Same for other pairs
  All `_x` have the same `n_samples_p_window` (dim 1). Same with all `_y`
  All have the same `n_features (dim 2). 
  If `split_out_val`=True, then `val_x` and `val_y` are None.

  @param merge_all: over-writes merge_train_val
  """
  fnames = ['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y']
  res = []
  for fname in fnames:
    res.append(np.load("{}{}.npy".format(dataset_path, fname), allow_pickle=True))
  if flatten_x_dim:
    for i in [0,2,4]: res[i] = flatten_samples(res[i])
  if flatten_y_dim:
    for i in [1,3,5]: res[i] = flatten_samples(res[i])

  if merge_all:
    res[0] = np.concatenate((res[0], res[2], res[4]))
    res[1] = np.concatenate((res[1], res[3], res[5]))
    res[2:] = [None]*4
  elif merge_train_val:
    res[0] = np.concatenate((res[0], res[2]))
    res[1] = np.concatenate((res[1], res[3]))
    res[2], res[3] = None, None

  return res

def flatten_samples(data):
  """
  We have data.shape = (n_windows, n_samples, n_features) 
  Reshape to (n_windows, n_samples*n_features), but we have all n_samples from 
  feature 0, then all n_samples of feature 1, etc. This requrires a swapaxis()
  
  """
  return np.reshape(
      np.swapaxes(data, 2,1)
      , (data.shape[0], -1)
      )
def unflatten_samples(data, n_kins=4):
  """
  We have data.shape = (n_windows, n_samples, n_features) 
  Reshape to (n_windows, n_samples*n_features), but we have all n_samples from 
  feature 0, then all n_samples of feature 1, etc. This requrires a swapaxis()
  
  """
  return np.swapaxes(
      np.reshape(data, (data.shape[0], 4, -1))
      , 2,1)

def get_fixed_point_y(y, seconds, rate=60):
  """
  @param y: array shape (n_windows, n_samples, n_fatures)
  """
  indx = int(seconds*rate) - 1
  return y[:, indx, :]