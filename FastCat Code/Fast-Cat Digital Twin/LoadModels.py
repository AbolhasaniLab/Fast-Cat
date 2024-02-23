import numpy as np
import math
import torch
import random
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import messagebox
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import RMSprop
from scipy.stats import linregress
from tensorflow.keras.models import Sequential, load_model
from matplotlib import pyplot
import os


messagebox.showinfo("Input Directory", "Please select directory")
project_dir = askdirectory()
npins = pd.read_csv(project_dir + "//" + "HFNonDimInputs.csv", header=None).to_numpy()
dim_in = npins.shape[1]
npouts = pd.read_csv(project_dir + "//" + "HFNonDimOutputs.csv", header=None).to_numpy()
dim_out = npouts.shape[1]
out_shape = npouts.shape[0]
npins = npins[:out_shape, :]

# Concatenate input and output data
npdata = np.concatenate([npins, npouts], axis=1)

################# LOAD MODEL ###################
messagebox.showinfo("Input Directory", "Please select the model directory")
model_dir = askdirectory()
ensemble = load_model(model_dir)

n_train = math.floor(0.8 * npins.shape[0])
n_val = math.floor(0.9 * npins.shape[0])



ens_y_pred = ensemble.predict(npdata[:, :-2])

pyplot.scatter(npdata[:,-2:-1].squeeze(), ens_y_pred[0][:,0])
pyplot.scatter(npdata[:,-1:].squeeze(), ens_y_pred[0][:,1])
pyplot.xlabel("True value")
pyplot.ylabel("Prediction")
pyplot.plot([0,1],[0,1])
pyplot.legend(['Yield','Selectivity','Parity'])
pyplot.show()

ens_full_yield = linregress(npdata[:,-2:-1].squeeze(), ens_y_pred[0][:,0])
ens_full_sele = linregress(npdata[:,-1:].squeeze(), ens_y_pred[0][:,1])
ens_test_yield = linregress(npdata[n_val:,-2:-1].squeeze(), ens_y_pred[0][n_val:,0])
ens_test_sele = linregress(npdata[n_val:,-1:].squeeze(), ens_y_pred[0][n_val:,1])


print('Ensemble Full Set:')
print(f'Yield_R^2: {ens_full_yield.rvalue**2}, Sele_R^2: {ens_full_sele.rvalue**2}')