import numpy as np
import math
import torch
import random
import shap
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import messagebox
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import RMSprop
from scipy.stats import linregress
from tensorflow.keras.models import Sequential, load_model
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
model = load_model(model_dir)
def f_y(x):
    return model.predict(x)[0][:,0]
    # return model.predict(x[i,:].reshape(1,x.shape[1]) for i in range(x.shape[0]))[0][:,0].flatten()
def f_s(x):
    return model.predict(x)[0][:, 1]
    # return model.predict(x[i,:].reshape(1,x.shape[1]) for i in range(x.shape[0]))[0][:,1].flatten()
explainer_s= shap.KernelExplainer(f_s, npins)
explainer_y= shap.KernelExplainer(f_y, npins)
shap_values_s = explainer_s.shap_values(npins[:,:], nsamples = 100)
shap_values_y = explainer_y.shap_values(npins[:,:], nsamples = 100)
shap.force_plot(explainer_s.expected_value, shap_values_s, npins[:,:])
shap.force_plot(explainer_y.expected_value, shap_values_y, npins[:,:])
fig_s = shap.summary_plot(shap_values = shap_values_s,features=npins[:,:])
fig_y = shap.summary_plot(shap_values = shap_values_y,features=npins[:,:])