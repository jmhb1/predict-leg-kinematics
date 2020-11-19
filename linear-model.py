!pip install -q h5py pyyaml
#!pip install -q tensorflow==2.0.0-beta1

# Load dependencies and mount google drive, go to datasets folder
from tensorflow import keras
import os
from google.colab import drive
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR
import importlib

try: import import_ipynb
except: 
  !pip install import_ipynb
  import import_ipynb

drive.mount('/content/drive/')

util_path = "drive/My Drive/Motion Prediction/Code/util"
util = importlib.import_module(util_path)
util = importlib.reload(importlib.import_module(util_path))

dataset_name = 'tst-pipeline_IMUFalse_2.0_1.0_0.5/'
data_folder = "drive/My Drive/Motion Prediction/Data_collection/Datasets/"
save_folder = "drive/My Drive/Motion Prediction/Code/Results/"
model_folder = "drive/My Drive/Motion Prediction/Code/Models/saved_models/"
plot_folder = "drive/My Drive/Motion Prediction/Code/Plots/"

def validation_curve_ridge_1d(X, Y, alpha_range, cv=5
                    , error_metric='abs_mean', threshold=10):
  """
  Assume the X and Y are already shuffled 
  @param error: valid options 'abs_mean', 'mse'
  """
  X_splits, Y_splits = np.array_split(X, cv), np.array_split(Y, cv)
  res = np.zeros((len(alpha_range), cv))
  res_train = np.zeros((len(alpha_range), cv))

  print("Testing {} hypers".format(len(alpha_range)))
  for i in range(len(alpha_range)):
    print(i)
    for j in range(cv):
      X_train = np.concatenate([X_splits[k] for k in range(cv) if k!=j])
      Y_train = np.concatenate([Y_splits[k] for k in range(cv) if k!=j])
      X_test = X_splits[j]
      Y_test = Y_splits[j]

      lm = Ridge(alpha=alpha_range[i])
      lm.fit(X_train, Y_train)
      Y_pred = lm.predict(X_test)
      Y_pred_train = lm.predict(X_train)
      
      if error_metric=='abs_mean':
        res[i,j] = np.mean(np.mean(np.abs(Y_test- Y_pred), axis=0), axis=0)
        res_train[i,j] = np.mean(np.mean(np.abs(Y_train- Y_pred_train)
        							, axis=0), axis=0)
      elif error_metric=='mse':
        res[i,j] = np.mean((Y_pred-Y_test)**2)**0.5
        res_train[i,j] = np.mean((Y_pred_train-Y_train)**2)**0.5
      elif error_metric=='threshold':
        res[i,j] = np.mean(abs(Y_pred-Y_test)>threshold)
        res_train[i,j] = np.mean(abs(Y_pred_train-Y_train)>threshold)
  return res, res_train

def plot_validation(res, res_train, fs=18):
  """ 
  Inputs are compatible with outputs of `validation_curve_ridge_1d()`.
  Plot the validation curve. Outputs to a cell if run in a notebook. 
  Otherwise the funcition needs modifying to save or open plots. 
"""
  errs = np.mean(res, axis=1)
  stds = np.std(res, axis=1)

  errs_train = np.mean(res_train, axis=1)
  stds_train = np.std(res_train, axis=1)

  f, ax = plt.subplots(1, 1, figsize=(6,6))
  ax.plot(alpha_range, errs, label="Avg Over CV's", color='k')
  ax.scatter(alpha_range, errs, color='k')
  ax.fill_between(alpha_range, errs-stds, errs+stds, alpha=0.1
                  , label='1 Std Range', color='k')

  ax.plot(alpha_range, errs_train, label="Train", color='y')
  ax.scatter(alpha_range, errs_train, color='y')

  ax.set(xscale='log', xlabel='Ridge regularizer', ylabel='Mean Error (deg)')
    # , title="Ridge regularizer validation curve")
  ax.legend()
  ax.xaxis.label.set_size(fs)
  ax.yaxis.label.set_size(fs)

def run_validation_curve(dataset_name="all-data_IMUFalse_2.0_1.0_0.5"):
	"""
	Read data, call `validation_curve_ridge_1d()` and run plot.
	"""
	dataset_name = "all-data_IMUFalse_2.0_1.0_0.5"
	# dataset_name = "all-S1-data_IMUFalse_2.0_1.0_0.5"
	dataset_path = "{}{}/".format(data_folder, dataset_name)
	print(dataset_path)
	res = util.load_data(dataset_path, flatten_x_dim=True, flatten_y_dim=True
	              , merge_all=True)
	X, Y, _, _, _, _ = res
	print("X shape {}, Y shape {}".format(X.shape,Y.shape))

	alpha_range=np.logspace(-7, 7, 20)
	res , res_train = validation_curve_ridge_1d(X, Y, alpha_range, cv=5
	                                            , error_metric='abs_mean')
	
	do_validation_plot(res, res_train, fs=26)


def learning_curve_ridge_1d(X, Y, cv=5, alpha=100, steps=10
			, error_metric='abs_mean'):
  """
  Given a dataset, build a learning curve, where the test results are given 
  by 10-fold cross validation. This generates the data but does not do the 
  plotting
  """
  print(X.shape)
  X_splits, Y_splits = np.array_split(X, steps), np.array_split(Y, steps)

  res = np.zeros((steps, cv))
  res_train = np.zeros((steps, cv))
  n_samples_arr = np.zeros((steps))

  for i in range(steps):
    X_, Y_ = np.concatenate(X_splits[:(i+1)]), np.concatenate(Y_splits[:(i+1)])
    n_samples_arr[i] = X_.shape[0]
    X_splits_, Y_splits_ = np.array_split(X_, cv), np.array_split(Y_, cv)
    for j in range(cv):
        X_train = np.concatenate([X_splits_[k] for k in range(cv) if k!=j])
        Y_train = np.concatenate([Y_splits_[k] for k in range(cv) if k!=j])
        X_test = X_splits_[j]
        Y_test = Y_splits_[j]
        
        lm = Ridge(alpha=alpha)
        lm.fit(X_train, Y_train)
        Y_pred = lm.predict(X_test) 
        Y_pred_train = lm.predict(X_train)
        # print(X_train.shape, X_test.shape)

        if error_metric=='abs_mean':
          res[i,j] = np.mean(np.mean(np.abs(Y_test- Y_pred), axis=0), axis=0)
          res_train[i,j] = np.mean(np.mean(np.abs(Y_train- Y_pred_train)
          	, axis=0), axis=0)  
        elif error_metric=='mse':
          res[i,j] = np.mean((Y_pred-Y_test)**2)**0.5
          res_train[i,j] = np.mean((Y_pred_train-Y_train)**2)**0.5

  return res, res_train, n_samples_arr

def plot_learning_curve(res, res_train, n_sample_arr):
	""" 
  Inputs are compatible with outputs of `validation_curve_ridge_1d()`.
  Plot the validation curve. Outputs to a cell if run in a notebook. 
  Otherwise the funcition needs modifying to save or open plots. 
"""
	res, res_train, n_samples_arr = res_all_data, res_train_all_data, n_samples_arr_all_data
	errs = np.mean(res, axis=1)
	errs_train = np.mean(res_train, axis=1)
	stds = np.std(res, axis=1)
	stds_train = np.std(res_train, axis=1)
	 

	f, ax = plt.subplots(1,1, figsize=(6,6))

	ax.plot(n_samples_arr, errs, c='k', label="10 Cross-Val") 
	ax.scatter(n_samples_arr, errs, c='k') 
	ax.fill_between(n_samples_arr, errs-stds, errs+stds, alpha=0.1
	                , label='1 Std Range')

	ax.plot(n_samples_arr, errs_train, c='y', label="Train")
	ax.scatter(n_samples_arr, errs_train, c='y')
	# ax.fill_between(n_samples_arr, errs_train-stds_train, errs_train-stds_train, alpha=0.1
	#                 , color='y')

	ax.set(xlabel="Number samples")
	ax.set(ylabel="Mean absolute error")
	ax.legend()

	fs=26
	ax.xaxis.label.set_size(fs)
	ax.yaxis.label.set_size(fs)

def run_learning_curve(dataset_name="all-data_IMUFalse_2.0_1.0_0.5"):
	"""
	Read data, call `validation_curve_ridge_1d()` and run plot.
	"""
	dataset_path = "{}{}/".format(data_folder, dataset_name)
	print(dataset_path)
	res = util.load_data(dataset_path, flatten_x_dim=True, flatten_y_dim=True
	              , merge_all=True)
	X, Y, _, _, _, _ = res
	print("X shape {}, Y shape {}".format(X.shape,Y.shape))

	alpha=100
	steps=30
	# shuffle the deck
	indxs = np.arange(0, len(X), dtype=int)
	np.random.shuffle(indxs)
	res, res_train, n_samples_arr = learning_curve_ridge_1d(X[indxs], Y[indxs], cv=5
	              , alpha=alpha, steps=steps, error_metric='abs_mean')  
	plot_learning_curve(res, res_train, fs=26)


def gen_0th_order_error(X_test, Y_test, error_metric='abs_mean'):
	X_test_unflattened = util.unflatten_samples(X_test)
	Y_test_unflattened = util.unflatten_samples(Y_test)
	Y_pred_0th = X_test_unflattened[:,-1,:]
	Y_pred_0th = np.repeat(Y_pred_0th[:,None,:], Y_test_unflattened.shape[1], axis=1)
	if error_metric=='abs_mean':
	error_0th = np.mean(np.abs(Y_pred_0th-Y_test_unflattened), axis=0)
	elif error_metric=='mse':  
	error_0th = np.mean((Y_pred_0th-Y_test_unflattened)**2, axis=0)**0.5
	else:
	throw: ValueError()
	error_0th_mean_over_features = np.mean(error_0th, axis=1)
	return error_0th_mean_over_features

def gen_1st_order_error(X_test, Y_test, error_metric='abs_mean'):
	X_test_unflattened = util.unflatten_samples(X_test)
	Y_test_unflattened = util.unflatten_samples(Y_test)
	slope = (X_test_unflattened[:,-1,:] - X_test_unflattened[:,-2,:])
	steps = Y_test_unflattened.shape[1]
	steps_arr = np.arange(0,steps,1)+1

	intcpt = X_test_unflattened[:,-1,:] # starting point for each sample 
	Y_pred = np.repeat(intcpt[:,None,:], steps, axis=1) \
	    + np.repeat(slope[:,None,:], steps, axis=1) * steps_arr[:,None]
	if error_metric=='abs_mean':
	  error_1st = np.mean(np.abs(Y_pred-Y_test_unflattened), axis=0)
	elif error_metric=='mse':  
	error_1st = np.mean((Y_pred-Y_test_unflattened)**2, axis=0)**0.5
	else:
	throw: ValueError()
	error_1st_mean_over_features = np.mean(error_1st, axis=1)
	return error_1st_mean_over_features



def run_and_plot_full_prediction_window(
	dataset_name= "all-data_IMUFalse_2.0_1.0_0.5"
	):
	"""
	Plot all 4 time series prediction errors (means) for the ridge regression 
	over the 1 second prediction window. 
	Also run the functions for 0th and 1st order 

	Outputs to a cell if run in a notebook. 
  	Otherwise the funcition needs modifying to save or open plots. 

	
	"""
	dataset_path = "{}{}/".format(data_folder, dataset_name)
	print(dataset_path)
	res = util.load_data(dataset_path, flatten_x_dim=True, flatten_y_dim=True
	              , merge_all=True)
	X, Y, _, _, _, _ = res
	print("X shape {}, Y shape {}".format(X.shape,Y.shape))

	err_0th = gen_0th_order_error(X_test, Y_test)
	err_1st = gen_1st_order_error(X_test, Y_test, error_metric='abs_mean')

	# run the ridge model 
	alpha = 100
	cv = 5
	lm = Ridge(alpha=alpha)
	lm.fit(X_train, Y_train)
	Y_pred = lm.predict(X_test)
	Y_pred.shape, Y_test.shape
	
	err_mean_abs = np.mean(
    	util.unflatten_samples(np.abs(Y_pred-Y_test)
    	), axis=0)

	f, axs = plt.subplots(1,1, figsize=(12,6))
	x_pnts = np.array(range(len(err_mean_abs)))/60*1000
	err_mean_abs_avg = np.mean(err_mean_abs, axis=1)

	labels = ['hip_flexion_r', 'knee_angle_r', 'hip_flexion_l', 'knee_angle_l']
	lw_joints=1
	axs.plot(x_pnts, err_mean_abs[:,0], c='y', label=labels[0], lw=lw_joints )
	axs.plot(x_pnts, err_mean_abs[:,1], c='b', label=labels[1], lw=lw_joints )
	axs.plot(x_pnts, err_mean_abs[:,2], c='r', label=labels[2], lw=lw_joints )
	axs.plot(x_pnts, err_mean_abs[:,3], c='c', label=labels[3], lw=lw_joints )

	error_0th = gen_0th_order_error(X_test, Y_test, error_metric='abs_mean')
	axs.plot(x_pnts, error_0th, c='y', ls='--', lw=3, label='0th order estimate')

	error_1st = gen_1st_order_error(X_test, Y_test, error_metric='abs_mean')
	axs.plot(x_pnts, error_1st, c='r', ls='--', lw=3, label='1st order estimate')

	axs.plot(x_pnts, err_mean_abs_avg, lw=5, c='k', label='mean')
	axs.legend()
	axs.set(xlabel='Prediction time ahead (ms)'
	    , ylabel="Mean abs error (deg)"
	    , ylim=[0,15]
	    , xlim=[0,1000]
	    )
	fs=26$
	axs.xaxis.label.set_size(fs)
	axs.yaxis.label.set_size(fs)
	
