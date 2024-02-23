import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from tkinter.filedialog import askdirectory
from tkinter import messagebox

from scipy.stats import linregress
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def gen_cascade_model(nodes=None, hidden_layers=None, dropout=0.1, input_shape=7, output_shape=2):
    if hidden_layers is None:
        hidden_layers = [2, 5]
    if nodes is None:
        nodes = [10, 20]
    inputs = keras.Input(shape=(input_shape))
    Out = layers.Dense(output_shape, activation='linear')
    Drop = keras.layers.Dropout(dropout)
    x = ()
    x = (*x, inputs)
    for layer in range(random.randint(*hidden_layers)):
        new_layer = layers.Dense(random.randint(*nodes),activation='relu')
        new_layer_output = new_layer(keras.layers.concatenate([*x]))
        x = (*x, new_layer_output)
    dropout = Drop(x[-1])
    output = Out(dropout)

    model = keras.Model(inputs=inputs, outputs=output)

    return model



################ Getting the input data ##############################
messagebox.showinfo("Input Directory", "Please select directory")
project_dir = askdirectory()

npins = pd.read_csv(project_dir + "//" + "HFNonDimInputs.csv", header=None).to_numpy()
dim_in = npins.shape[1]
npouts = pd.read_csv(project_dir + "//" + "HFNonDimOutputs.csv", header=None).to_numpy()
dim_out = npouts.shape[1]

npdata = np.concatenate([npins, npouts], axis=1)

# Shuffle input data   #Not necessary for Arch Study
# rng = np.random.default_rng()
# rng.shuffle(npdata)
# Training Testing Validation Split
n_train = math.floor(0.8 * npins.shape[0])
n_val = math.floor(0.9 * npins.shape[0])
trainX, valX, testX = npdata[:n_train, :-1 * dim_out], npdata[n_train:n_val, :-1 * dim_out], npdata[n_val:, :-1 * dim_out]
trainy, valy, testy = npdata[:n_train, -1 * dim_out:], npdata[n_train:n_val, -1 * dim_out:], npdata[n_val:, -1 * dim_out:]

################### Arch Search ##############################
num_layers = [2, 11]
num_nodes = [5, 20]
indiv_models = ()
yield_rsq = []
sele_rsq = []
dropout = 0.1
training_history = list()
for i in range(*num_layers):
    for j in range(*num_nodes):
        new_model = gen_cascade_model(nodes = [j,j], hidden_layers=[i,i])
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.00001, mode='min')
        new_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(learning_rate=0.0008),
                          metrics=[keras.metrics.RootMeanSquaredError()])
        new_model.fit(trainX, trainy, validation_data=(valX, valy), epochs=900, verbose=1, callbacks=early_stopping)
        indiv_models = (*indiv_models, new_model)
        y_pred = new_model.predict(npdata[:, :-2])
        full_yield = linregress(npdata[:, -2:-1].squeeze(), y_pred[:, 0])
        full_sele = linregress(npdata[:, -1:].squeeze(), y_pred[:, 1])
        yield_rsq = full_yield.rvalue**2
        sele_rsq = full_sele.rvalue**2
        training_history.append([i,j,yield_rsq,sele_rsq])

    X = np.arange(2, 11, 1)
    Y = np.arange(5, 20, 1)
    X, Y = np.meshgrid(X, Y)
    R_yield = np.zeros([X.shape[0], X.shape[1]]) - 1.0
    R_sele = np.zeros([X.shape[0], X.shape[1]]) - 1.0
    for arch in training_history:
        k = arch[0]
        m = arch[1]
        r_yield = arch[2]
        r_sele = arch[3]
        R_yield[m - num_nodes[0], k - num_layers[0]] = r_yield
        R_sele[m - num_nodes[0], k - num_layers[0]] = r_sele

fig, ax = plt.subplots()
cmap = ax.pcolormesh(X, Y, R_sele)
fig.colorbar(cmap)
plt.title('Selectivity R^2')
plt.xlabel('#Layers')
plt.ylabel('#Nodes')
plt.show()

fig, ax = plt.subplots()
cmap = ax.pcolormesh(X, Y, R_yield)
fig.colorbar(cmap)
plt.title('Yield R^2')
plt.xlabel('#Layers')
plt.ylabel('#Nodes')
plt.show()
