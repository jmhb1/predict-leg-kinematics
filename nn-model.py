!pip install -q h5py pyyaml
#!pip install -q tensorflow==2.0.0-beta1
# Load dependencies and mount google drive, go to datasets folder
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import os
from google.colab import drive
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

drive.mount('/content/drive/')
try: 
  import import_ipynb
except: 
  !pip install import_ipynb
  import import_ipynb

drive.mount('/content/drive/')
import importlib
util_path = "drive/My Drive/Motion Prediction/Code/util"
util = importlib.import_module(util_path)
util = importlib.reload(importlib.import_module(util_path))

!pip install -q -U keras-tuner
import kerastuner as kt

dataset_name = 'tst-pipeline_IMUFalse_2.0_1.0_0.5/'
data_folder = "drive/My Drive/Motion Prediction/Data_collection/Datasets/"
save_folder = "drive/My Drive/Motion Prediction/Code/Results/"
model_folder = "drive/My Drive/Motion Prediction/Code/Models/saved_models/"
plot_folder = "drive/My Drive/Motion Prediction/Code/Plots/"

def linear_model(train_x, train_y, test_x, test_y, alpha=100, error='abs_mean'):
	""" 
	Train a simple ridge regerssion on `train` and test on `test`. Avg errors 
	over each feature=time point (that is, each element of test_y), then 
	average those errors over the 4 features. Suitable for plotting errors 
	as a function of prediction window time
	"""
	lm = Ridge(alpha=100).fit(train_x, train_y)
	y_pred = lm.predict(test_x)
	y_pred_train = lm.predict(train_x)
	err = np.mean(np.mean(util.unflatten_samples(np.abs(y_pred-test_y))
					, axis=0), axis=1)
	print("Train loss ",np.mean(np.abs(y_pred_train-train_y)))
	print("Test loss ",np.mean(np.abs(y_pred-test_y)))
	return err


def get_loss_timeseries(model):
	"""
	Given an trained `model`, that supports the standard `predict()` method, 
	produce the same error array as returned by `linear_model` method. 
	"""
	y_pred = model.predict(test_x)
	y_pred_train = model.predict(train_x)
	err = np.mean(np.mean(util.unflatten_samples(np.abs(y_pred-test_y))
		, axis=0), axis=1)
	print("Train loss ",np.mean(np.abs(y_pred_train-train_y)))
	print("Test loss ",np.mean(np.abs(y_pred-test_y)))
	return err


def get_vanilla_nn_train_and_test_predictions(train_x, train_y, val_x, val_y
                 , h_units=500, lr=0.001, epochs=20
                 , second_layer=False, third_layer=False, activation='relu'
                 , leaky=False, leaky_alpha=0.1
                 , dropout=False, dropout_rate=0.2, early_stopping=True):
	"""
	Build densely connected neural nets. The arguments are indended to give 
	flexibility about hyperparamters. 
	Return model predictions `Y_pred` (for test) and `Y_train` (for train)
	"""
	n_train, n_in_features = train_x.shape
	_, n_out_features = train_y.shape

	lr=0.001
	if leaky: h1_activation = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)
	else: h1_activation = activation

	inputs = keras.Input(shape=(n_in_features,)) 
	h1 = keras.layers.Dense(units=h_units
	                      , activation=h1_activation
	                      , name='dense-1')(inputs)
	if dropout: 
		h1 = keras.layers.Dropout(dropout_rate)(h1)

	if second_layer: 
		last_dense_layer = keras.layers.Dense(units=h_units
	                      , activation=h1_activation
	                      , name='dense-2')(h1)                        
	else: 
		last_dense_layer = h1
	if third_layer: 
		last_dense_layer = keras.layers.Dense(units=h_units
	                      , activation=h1_activation
	                      , name='dense-3')(last_dense_layer)                        

	if dropout: last_dense_layer = keras.layers.Dropout(dropout_rate)(last_dense_layer)
		outputs = keras.layers.Dense(units=n_out_features
	                          , activation=None
	                          , name='predictions')(last_dense_layer)

	model = keras.Model(inputs=inputs, outputs=outputs, name='nn_1layer')
	model.compile(optimizer=keras.optimizers.Adam(lr=lr)
	          , loss=keras.losses.mean_absolute_error
	          , metrics=['MeanAbsoluteError']
	          ) 

	if early_stopping:
		early_stopping_cb = tf.keras.callbacks.EarlyStopping(
	        monitor='loss', min_delta=0.1, patience=3, verbose=1, mode='auto',
	        baseline=None, restore_best_weights=False
	    	)

	history = model.fit(
	  train_x,
	  train_y,
	  # batch_size=64,
	  epochs=epochs,
	  # This is for monitoring validation loss and metrics
	  # at the end of each epoch
	  validation_data=(val_x, val_y),
	  callbacks=[early_stopping_cb]
	)

	return model.predict(test_x), model.predict(train_x)

def learning_curve_vanilla_nn(train_x, train_y, test_x, test_y,
          lr=0.001, h_units=700
          , sizes = [500,1000,2000,5000,10000,20000,50000]
          , steps=10, epochs=20, error_metric='abs_mean'
          , early_stopping=True, second_layer=False, third_layer=False):
  print(train_x.shape)
  
  n = len(sizes)
  res = np.zeros((n))
  res_train = np.zeros((n))
  n_samples_arr = np.zeros((n))

  for i in range(len(sizes)):
    print(i)
    # X_, Y_ = np.concatenate(X_splits[:(i+1)]), np.concatenate(Y_splits[:(i+1)])
    X_, Y_ = train_x[:sizes[i]], train_y[:sizes[i]]
    print("Training size: {}, Test size: {}".format(X_.shape[0], test_x.shape[0]))
    n_samples_arr[i] = X_.shape[0]
    # X_splits_, Y_splits_ = np.array_split(X_, cv), np.array_split(Y_, cv)

    Y_pred, Y_pred_train = \
      get_vanilla_nn_train_and_test_predictions(X_, Y_, test_x, test_y
        , h_units=h_units, lr=lr, epochs=epochs, early_stopping=early_stopping
        , second_layer=second_layer, third_layer=third_layer)
    
    if error_metric=='abs_mean':
      res[i] = np.mean(np.mean(np.abs(test_y- Y_pred), axis=0), axis=0)
      res_train[i] = np.mean(np.mean(np.abs(Y_- Y_pred_train), axis=0), axis=0)  
    elif error_metric=='mse':
      res[i] = np.mean((Y_pred-test_y)**2)**0.5
      res_train[i] = np.mean((Y_pred_train-Y_)**2)**0.5
  return res, res_train, n_samples_arr

def plot_learning_curve(res, res_train, n_samples_arr):
	"""
	Intended to be run in a notebook, otherwise add code for saving figure. 
	Hand-coded in the linear benchmark. May need updating.
	"""
	f, axs = plt.subplots(1,1,figsize=(6,6))
	axs.plot(n_samples_arr, res, label='test', c='k')
	axs.scatter(n_samples_arr, res, c='k')

	axs.plot(n_samples_arr, res_train, label='train',c='y')
	axs.scatter(n_samples_arr, res_train, c='y')

	axs.plot([0,50000], [5.5,5.5], ls='--', label='Linear model')

	axs.set(xlabel='Number samples', ylabel="Loss, mean absolute error"
	, xlim=[0,50000], ylim=[0,10])
	axs.legend()

	fs=24
	axs.xaxis.label.set_size(fs)
	axs.yaxis.label.set_size(fs)

def run_nn_learning_curve():
	steps=5
	h_units=700
	second_layer=True
	third_layer=False
	lr=0.001
	epochs=50
	early_stopping=True
	
	# shuffle the deck
	indxs = np.arange(0, len(train_x), dtype=int)
	np.random.shuffle(indxs)

	res, res_train, n_samples_arr = learning_curve_vanilla_nn(train_x[indxs], train_y[indxs]
	    , test_x, test_y, lr=lr, h_units=h_units, steps=steps, early_stopping=early_stopping
	    , epochs=epochs, error_metric='abs_mean'
	    # , sizes = [50,100,500,1000,2000,5000,10000,20000,35000,50000]
	    , sizes=[80000]
	    , second_layer=second_layer, third_layer=third_layer)

	plot_learning_curve(res, res_train, n_samples_arr)


def plot_prediction_window(test_y, Y_pred, errs_linear):
	"""
	"""
	errs_nn = np.mean(
    np.mean(util.unflatten_samples(np.abs(test_y- Y_pred)), axis=0)
    , axis=1)
	errs_train_nn = errs_nn = np.mean(
	    np.mean(util.unflatten_samples(np.abs(test_y- Y_pred)), axis=0)
	    , axis=1)


	f, axs = plt.subplots(1,1, figsize=(12,6))
	x_pnts = np.array(range(len(errs_nn)))/60*1000
	axs.plot(x_pnts, errs_linear, label='Linear', c='y', lw=3)
	axs.plot(x_pnts, errs_nn, label='Feedforward NN', c='k', lw=3)

	import linear_model as lm
	error_0th = lm.gen_0th_order_error(test_x, test_y, error_metric='abs_mean')
	error_1st = lm.gen_1st_order_error(test_x, test_y, error_metric='abs_mean')
	axs.plot(x_pnts, error_0th, c='r', ls='--', lw=3, label='0th order estimate')
	axs.plot(x_pnts, error_1st, c='b', ls='--', lw=3, label='1st order estimate')

	axs.legend()
	axs.set(xlabel="Prediction time ahead (ms)", ylabel="Mean abs error (deg)"
	, xlim=[0,1000], ylim=[0,8])

	fs=24
	axs.xaxis.label.set_size(fs)
	axs.yaxis.label.set_size(fs)


def model_builder(hp): 
	"""Hyperparameter search with a bandit. 
	"""
	inputs = keras.Input(shape=(n_in_features,))

	hp_units = hp.Int('units', min_value = 32, max_value = 1024, step = 64)
	h1 = keras.layers.Dense(units=hp_units
	                      , activation='relu'
	                      , name='dense-1')(inputs)

	outputs = keras.layers.Dense(units=n_out_features
	                          , activation=None
	                          , name='predictions')(h1)
	hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 

	model = keras.Model(inputs=inputs, outputs=outputs, name='nn_1layer')

	model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
	            loss = keras.losses.mean_absolute_error
	            , metrics=['MeanAbsoluteError']
	            )
	return model



