import numpy as np
import math
import random
import pandas as pd
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import RootMeanSquaredError

from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import messagebox

from scipy.stats import linregress
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def gen_cascade_model(nodes=None, hidden_layers=None, dropout=0.1, input_shape=7, output_shape=2):
    if hidden_layers is None:
        hidden_layers = [2, 5]
    if nodes is None:
        nodes = [15, 20]
    inputs = keras.Input(shape=(input_shape))
    Out = layers.Dense(output_shape, activation='linear')
    Drop = keras.layers.Dropout(dropout)
    x = ()
    x = (*x, inputs)
    for layer in range(random.randint(*hidden_layers)):
        new_layer = layers.Dense(random.randint(*nodes),activation='relu')
        Cat = Concatenate()
        if len(x) == 1:
            new_layer_output = new_layer(*x)
        else:
            new_layer_output = new_layer(Cat([*x]))
        x = (*x, new_layer_output)
    dropout = Drop(x[-1])
    output = Out(dropout)

    model = keras.Model(inputs=inputs, outputs=output)

    return model


def gen_ensemble_model(models, num_nodes=10, num_layers=1):
    inputs = keras.Input(shape=models[0].input_shape[1])
    keras_models = ()
    for model in models:
        m = model(inputs)
        keras_models = (*keras_models, m)

    cat_out = Concatenate()([*keras_models])
    x = cat_out
    for i in range(num_layers):
        Ens_Layer = layers.Dense(num_nodes, activation="relu")
        x = Ens_Layer(x)

    Ens_Out = layers.Dense(models[0].output_shape[1], activation='linear')
    ens_out = Ens_Out(x)
    # ens_out = relu(ens_out)
    ens_out = relu(ens_out, alpha=0.1, max_value=1)

    ensemble_model = keras.Model(inputs=inputs, outputs=[ens_out, cat_out])
    return ensemble_model


# Generate Ensemble NN with Training and Validation data. Retain some data for testing.
# Hidden_layers and nodes take integer lists [min_value, max_value].
def create_ensemble_nn(train_x, train_y, val_x, val_y, path, hidden_layers=None, nodes=None, num_models=10):
    # Avoiding List definition in function default parameters which can lead to unexpected results.
    # Change desired default parameters here.
    if hidden_layers is None:
        hidden_layers = [6, 9]
    if nodes is None:
        nodes = [10, 20]
    # Output filepath for saving models
    path = path + r'\Ensemble'
    # Instantiate lists for trained models
    models = ()
    train_y_cat = train_y
    val_y_cat = val_y
    # iterate over number of models desired.
    for i in range(num_models):
        # Start model definition
        model = gen_cascade_model(nodes=nodes, hidden_layers=hidden_layers, input_shape=train_x.shape[1], output_shape=train_y.shape[1])
        models = (*models, model)
        if i == 0:
            train_y_cat = train_y
            val_y_cat = val_y
        else:
            train_y_cat = np.concatenate((train_y_cat, train_y), axis=1)
            val_y_cat = np.concatenate((val_y_cat, val_y), axis=1)

    ensemble = gen_ensemble_model(models)

    # Early stopping terminates training if model accuracy begins to decrease due to over-fitting.
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.00001, mode='min')
    # Model compile with loss, optimizer, and performance metric of interest.
    ensemble.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.0005),
                  metrics=[RootMeanSquaredError()])
    # Perform Model Training with training and validation data with early stopping.
    ensemble_history = ensemble.fit(train_x, [train_y, train_y_cat], validation_data=(val_x, [val_y, val_y_cat]),
                     epochs=700,
                     verbose=0,
                     callbacks=[early_stopping])
    # Save model to the filepath and add trained model + history to the ensemble lists
    filepath = path + fr'\ensemble_model'
    ensemble.save(filepath)

    # Return ensemble of Tensorflow models and training history for each member in the ensemble.
    return ensemble, ensemble_history


# Test ensemble for accuracy given test data and ensemble list of models.



################ Getting the input data ##############################
messagebox.showinfo("Input Directory", "Please select directory")
project_dir = askdirectory()

npins = pd.read_csv(project_dir + "//" + "HFNonDimInputs.csv", header=None).to_numpy()
dim_in = npins.shape[1]
npouts = pd.read_csv(project_dir + "//" + "HFNonDimOutputs.csv", header=None).to_numpy()
dim_out = npouts.shape[1]

npdata = np.concatenate([npins, npouts], axis=1)
########### For a Fixed training set
# messagebox.showinfo("Input Directory", "Please select directory for training set")
# project_dir = askdirectory()
# npdata =  pd.read_csv(project_dir, header=None).to_numpy()

# Shuffle input data
rng = np.random.default_rng()
rng.shuffle(npdata)
# Training Testing Validation Split
n_train = math.floor(0.75 * npins.shape[0])
n_val = math.floor(0.9 * npins.shape[0])
trainX, valX, testX = npdata[:n_train, :-1 * dim_out], npdata[n_train:n_val, :-1 * dim_out], npdata[n_val:, :-1 * dim_out]
trainy, valy, testy = npdata[:n_train, -1 * dim_out:], npdata[n_train:n_val, -1 * dim_out:], npdata[n_val:, -1 * dim_out:]
ensemble, ensemble_hist = create_ensemble_nn(trainX, trainy, valX, valy, project_dir)

ens_val_loss = ensemble_hist.history['val_loss']
ens_loss = ensemble_hist.history['loss']
pyplot.plot(ens_loss)
pyplot.plot(ens_val_loss)
pyplot.legend(['loss','val_loss'])
pyplot.show()
########## To Save your Learning Process
# ens_loss_df = pd.DataFrame(np.array(ens_loss).reshape(-1))
# ens_loss_df.to_csv(project_dir+r'/LearningCurve/ens_loss.csv',index=False)
# ens_val_loss_df = pd.DataFrame(np.array(ens_loss).reshape(-1))
# ens_val_loss_df.to_csv(project_dir+r'/LearningCurve/ens_val_loss.csv',index=False)

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

print('Ensemble:')
print('Test Set:')
print(f'Yield_R^2: {ens_test_yield.rvalue**2}, Sele_R^2: {ens_test_sele.rvalue**2}')
print('Full Set:')
print(f'Yield_R^2: {ens_full_yield.rvalue**2}, Sele_R^2: {ens_full_sele.rvalue**2}')
#

####### This part is just to color code the training validation and test set on the parity plot #########
# from matplotlib import pyplot as plt
# plt.scatter(npdata[:n_train,-2:-1].squeeze(), ens_y_pred[0][:n_train,0], c='orange')
# plt.scatter(npdata[:n_train,-1:].squeeze(), ens_y_pred[0][:n_train,1], c='red')
# plt.scatter(npdata[n_train:n_val,-2:-1].squeeze(), ens_y_pred[0][n_train:n_val,0], c='blue')
# plt.scatter(npdata[n_train:n_val,-1:].squeeze(), ens_y_pred[0][n_train:n_val,1], c='black')
# plt.scatter(npdata[n_val:,-2:-1].squeeze(), ens_y_pred[0][n_val:,0], c='green')
# plt.scatter(npdata[n_val:,-1:].squeeze(), ens_y_pred[0][n_val:,1], c='pink')
# plt.plot([0,1],[0,1])
# plt.title('Prediction vs True values')
# plt.xlabel('True value')
# plt.ylabel('Predicted value')
# plt.legend(['Yield_train','Selectivity_train','Yield_val','Selectivity_val','Yield_test','Selectivity_test','Parity'])
# plt.show()
