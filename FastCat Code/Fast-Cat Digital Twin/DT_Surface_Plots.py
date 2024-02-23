
if __name__ == '__main__':
    import os
    import glob
    import torch

    from matplotlib import pyplot

    import csv
    import math
    import numpy as np
    import random
    import pandas as pd

    from tkinter.filedialog import askopenfilename, askdirectory
    from tkinter import messagebox

    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import RootMeanSquaredError

    from scipy.stats import linregress
    import os
    # If you were getting errors related to OS, use the following line
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    messagebox.showinfo("Input Directory", "Please select directory")
    project_dir = askdirectory()
    ensemble = load_model(project_dir)
    F1 = np.arange(0.01, 1, 0.01)
    F2 = F1
    std_list = []
    out_list = []
    Data = np.ones([99, 99])
    Datay = np.ones([99, 99])
    Datas = np.ones([99, 99])
    for v in [0.25,0.5,0.75]:
        inputs_list = []
        for i, F1_n in enumerate(F1):
            for j, F2_n in enumerate(F2):
                inputs = np.array([v, v, F2_n, F1_n, v, v, v])
                inputs_list.append(inputs)
        a = np.array(inputs_list).reshape((9801, 7))
        y = ensemble.predict(a)
        M = y[1][:, ::2] # For Yield
        # M = y[1][:,1::2] # For Regioselectivity
        std = np.std(M, axis=1)
        out_std = std.reshape((99,99))
        std_list.append(out_std)
        # y = y[0][:, 1]        #For getting Regioselectivity
        y = y[0][:, 0]          #For getting Yield ** Use This line only for L4 and L5 - based on the model directory that you have selected, it will give you either regioselectivity or yield.
        out = np.array(y).reshape((99, 99))
        out_list.append(out)




