from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# Function executed on a new file being created in the observed directory
def on_created(event):
    # Print observed file path
    print(f"{event.src_path} created")
    time.sleep(5)
    # Initialize Hardware and monte carlo parameters
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    test = ''
    if test == 'y' or test == 'Y':
        SMOKE_TEST = True
    else:
        SMOKE_TEST = False

    # User defined Noise
    NOISE_SE = torch.tensor([0.02, 0.02], **tkwargs)
    # points per batch
    BATCH_SIZE = 10
    # number of monte carlo restarts per cycle if no better condition has been found
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    # Number of Ensemble Models
    N_MODELS = 10 if not SMOKE_TEST else 2
    # number of monte carlo samples on GP
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    # Number of batches of points
    N_BATCH = 50 if not SMOKE_TEST else 10
    # Initial number of MC samples to define model
    MC_SAMPLES = 512 if not SMOKE_TEST else 16

    standard_bounds = torch.zeros(2, 7, **tkwargs)
    standard_bounds[1] = 1

    # Import data from new analysis file
    data = None
    while data is None:
        try:
            data = read_gc(event.src_path)
            if isinstance(data, str):
                print('Wash Output Detected')
                return
            print('Data in file:\n')
            print(data)
            time.sleep(5)
        except Exception as e:
            print(e.message, e.args)
            time.sleep(10)

    # Add new data to compiled data
    processed_data = None
    while processed_data is None:
        try:
            processed_data = process_gc(data, project_dir, tkwargs)
            print('Appended Results:\n')
            print(processed_data)
            time.sleep(5)
        except Exception as e:
            print(e.message, e.args)
            time.sleep(10)

    # use compiled data to train new model and generate new conditions
    train_x = None
    while train_x is None:
        try:
            # load inputs and outputs and remove conditions with no corresponding results
            all_x = load_inputs(project_dir, tkwargs)
            train_obj = processed_data
            num_points = train_obj.shape[0]
            train_x = all_x[:num_points, :]

            # Model train/test split
            split = math.floor(0.8 * train_x.shape[0])
            perm = torch.randperm(train_x.shape[0])
            train_x = train_x[perm]
            train_obj = train_obj[perm]
            max_obj = torch.max(train_obj[:, 1])
            min_obj = torch.min(train_obj[:, 1])
            min_isomer_fraction = max_obj - 0.25 * (max_obj - min_obj)
            if not is_linear:
                min_isomer_fraction = 1 - (min_obj + 0.25 * (max_obj - min_obj))
            print(f'Minimum Isomer Fraction Threshold:{min_isomer_fraction}, Is_Linear:{is_linear}')
            print(train_x, train_obj)
            # Train new model
            ensemble, ensemble_history = create_ensemble_nn(train_x[:split].cpu().numpy(),
                                                            train_obj[:split].cpu().numpy(),
                                                            train_x[split:].cpu().numpy(),
                                                            train_obj[split:].cpu().numpy(), project_dir,
                                                            num_models=N_MODELS)
            # obtain new condition
            valid_x = condition_filter(ensemble, tkwargs, policy=policy, q=1, max_tau=max_tau,
                                                 min_isomer_fraction=min_isomer_fraction, is_linear=is_linear)
            # append new condition to hardware control .csv
            for point in valid_x:
                settings = generate_equipment_settings(point.reshape(7).cpu().numpy(), project_dir)

            print('New Experimental Condition:\n')
            print(valid_x)
            time.sleep(5)
        except Exception as e:
            print(e.message, e.args)
            time.sleep(10)

    conditions = None
    while conditions is None:
        try:
            # append new condition to HFNonDimInputs.csv
            conditions = test_run_gc(valid_x, project_dir)
            print('Appended Experimental Conditions:\n')
            print(conditions)
            time.sleep(5)
        except Exception as e:
            print(e.message, e.args)
            time.sleep(10)
    return conditions, data


def on_deleted(event):
    print(f"{event.src_path} deleted")


def on_moved(event):
    print(f"{event.src_path} moved")


def on_modified(event):
    print(f"{event.src_path} changed")


def test_read_gc(path):
    # Open GC File
    with open(path, 'r') as f:
        lines = f.readlines()

        data = []
        for idx, line in enumerate(lines):
            # Trim Newlines
            datastring = line.split('\n')[0]
            # Split along commas
            split_data = datastring.split(',')
            # Append [Elution Time, Peak Area, Concentration, Component Name]
            data.append([float(split_data[0]), float(split_data[1])])

    return data


# Obtain yield and selectivity from GC output file
def read_gc(path):
    # Open GC File
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in lines:
            if row.find('Wash') >= 0:
                return 'Wash'
            # Extract Peak Table Index
            startword = r'[Compound Results(Ch1)]'
            if row.find(startword) == 0:
                start = lines.index(row)
        table = lines[start + 3:-1]
    # Instantiate Data and fill from Peak Table
    gc_data = []

    tridecane_area = None
    for idx, peak in enumerate(table):
        # Trim Newlines
        datastring = peak.split('\n')[0]
        # Split along commas
        split_data = datastring.split(',')
        if split_data[1].find('Tridecane') == 0:
            tridecane_area = float(split_data[3])
        # Append [Compound, Peak Area]
        gc_data.append([split_data[1], float(split_data[3])])

    # Species initialization and calibration factors
    octene = 0
    isomers = 0
    nonanal = 0
    branched = 0
    octene_cal = 0.0342
    octene_t_cal = 0.0332
    octene_c_cal = 0.0344
    nonanal_cal = 0.0298
    # extract concentration from relative peak area of species and internal standard
    for peak in gc_data:
        peak[1] = peak[1] / tridecane_area
        if peak[0].find('1-Octene') == 0:
            octene = peak[1] / octene_cal
            peak.append(octene)
        if peak[0].find('2-c-Octene') == 0:
            conc = peak[1] / octene_c_cal
            isomers = isomers + conc
            peak.append(conc)
        if peak[0].find('2-t-Octene') == 0:
            conc = peak[1] / octene_t_cal
            isomers = isomers + conc
            peak.append(conc)
        if peak[0].find('3-Octene') == 0:
            conc = peak[1] / octene_c_cal
            isomers = isomers + conc
            peak.append(conc)
        if peak[0].find('4-Octene') == 0:
            conc = peak[1] / octene_c_cal
            isomers = isomers + conc
            peak.append(conc)
        if peak[0].find('2-PrHexanal') == 0:
            conc = peak[1] / nonanal_cal
            branched = branched + conc
            peak.append(conc)
        if peak[0].find('2-EtHeptanal') == 0:
            conc = peak[1] / nonanal_cal
            branched = branched + conc
            peak.append(conc)
        if peak[0].find('2-MeOctanal') == 0:
            conc = peak[1] / nonanal_cal
            branched = branched + conc
            peak.append(conc)
        if peak[0].find('Nonanal') == 0:
            nonanal = peak[1] / nonanal_cal
            peak.append(nonanal)

    # calculate selectivity and yield
    if branched > 0:
        selectivity = nonanal / branched
        x_n = selectivity / (selectivity + 1)
    else:
        x_n = 1
    if octene + nonanal + branched + isomers > 0:
        conversion = (nonanal + branched) / (octene + nonanal + branched + isomers)
    else:
        conversion = 0
    data = [[conversion, x_n]]

    return data


# load data from input file to torch device
def load_inputs(path, device_args):
    while True:
        try:
            old_x = pd.read_csv(path + r"/HFNonDimInputs.csv", header=None)
            old_x = np.array(old_x)
            old_x = torch.from_numpy(old_x)
            old_x.to(device=device_args['device'], dtype=device_args['dtype'])
            return old_x
        except:
            print("Waiting to load input values")
            time.sleep(1)


# add new data to compiled output data file
def process_gc(data, path, device_args):
    new_y = pd.DataFrame(data)
    try:
        old_y = pd.read_csv(path + r"/HFNonDimOutputs.csv", header=None)
        old_y = old_y.append(new_y)
    except:
        old_y = new_y
    old_y.to_csv(path + r"/HFNonDimOutputs.csv", header=False, index=False)
    old_y = np.array(old_y)
    old_y = torch.from_numpy(old_y)
    old_y.to(device=device_args['device'], dtype=device_args['dtype'])

    return old_y


# append new condition to compiled input file
def test_run_gc(x, path, write_file=False):
    try:
        old_x = pd.read_csv(path + r"/HFNonDimInputs.csv", header=None)
        old_x = old_x.append(pd.DataFrame(x.cpu()))
    except:
        old_x = pd.DataFrame(x.cpu())
    old_x.to_csv(path + r"/HFNonDimInputs.csv", header=False, index=False)

    if write_file:
        new_y = test_func(x)
        t = time.localtime()
        *s, _, _, _ = t
        time_str = f"{s[0]}-{s[1]}-{s[2]}-{s[3]}-{s[4]}-{s[5]}"
        with open(path + r"\TestGCOutput" + time_str + ".txt", 'w') as f:
            for row in new_y:
                outstr = f"{row[0]},{row[1]}\n"
                f.writelines(outstr)

    return old_x


def test_func(x):
    n = x.shape[1]
    y = torch.zeros(x.shape[0], 2, device=x.device)
    sqrt_n_tensor = torch.ones(n, device=x.device) * (1 / math.sqrt(n))
    for i in range(x.shape[0]):
        y[i, 0] = 1 - math.exp(-1 / n * sum((x[i, :] - sqrt_n_tensor) ** 2))
        y[i, 1] = 1 - math.exp(-1 / n * sum((x[i, :] + sqrt_n_tensor) ** 2))
    return y


def gen_test_function_data(device_args, n=16, write_data=True):
    # Create normalized boundaries [0, 1] from input data on correct botorch device
    bounds = torch.tensor([[0.] * 7, [1.] * 7],
                          device=device_args['device'],
                          dtype=device_args['dtype'])
    # Monte Carlo sampling random values within bounds for each input dimension.
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)

    # Sample model at generated  points and add noise defined by NOISE_SE
    train_obj_true = test_func(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE

    if write_data:
        path = project_dir
        x_df = pd.DataFrame(train_x.cpu())
        y_df = pd.DataFrame(train_obj.cpu())

        x_df.to_csv(path + r"\TestInput.csv", header=False, index=False)
        y_df.to_csv(path + r"\TestOutput.csv", header=False, index=False)

    # Return Randomized initial values with noisy model evaluation and true model evaluation.
    return train_x, train_obj, train_obj_true


# generate hardware settings from nondimensionalized input conditions and append to hardware control file
def generate_equipment_settings(inputs, path, replicates=0):
    file = path + r"\FlowConditions.csv"

    # Re-Dimensionalize Inputs
    v_CO = inputs[0] * (1.9 - 0.1) + 0.1  # mLn/min
    v_H2 = inputs[1] * (1.9 - 0.1) + 0.1
    v_gas = v_CO + v_H2

    pressure = inputs[2] * (300 - 150) + 150  # psi
    pressure_factor = (pressure / 14.7) + 1

    v_gas_p = v_gas * 1000 / pressure_factor  # uL/min
    v_liquid = v_gas_p / 3

    temperature = inputs[3] * (110 - 75) + 75  # C

    solvent_fraction = inputs[4] * (0.4 - 0.1) + 0.1
    v_solvent = solvent_fraction * v_liquid  # uL/min
    ligand_fraction = inputs[5] * (0.8 - 0.2) + 0.2
    olefin_fraction = inputs[6] * (0.8 - 0.2) + 0.2
    # uL/min
    v_Rh = v_liquid * (1 - solvent_fraction) / (
                1 + ligand_fraction / (1 - ligand_fraction) + olefin_fraction / (1 - olefin_fraction))
    v_Ligand = v_Rh * ligand_fraction / (1 - ligand_fraction)
    v_Olefin = v_Rh * olefin_fraction / (1 - olefin_fraction)

    # N2, CO, H2, Propylene, T, Pump1, Pump2, Pump3, Pump4, Collection, P, Eq, Rxn, Wash, Replicate
    equipment_string = f"0,{v_CO},{v_H2},0,{temperature},{v_Olefin},{v_Ligand},{v_Rh},{v_solvent},9,{pressure},1,3,1,0"
    replicate_string = f"0,{v_CO},{v_H2},0,{temperature},{v_Olefin},{v_Ligand},{v_Rh},{v_solvent},9,{pressure},0,0,0,1"

    equipment_settings = (v_CO, v_H2, temperature, v_Olefin, v_Rh, v_Ligand, v_solvent, pressure)
    while True:
        try:
            with open(file, 'a') as f:

                f.write(equipment_string + "\n")
                for i in range(replicates):
                    f.write(replicate_string + "\n")
            return equipment_settings
        except:
            print("Waiting for File Access")
            print(f"File: {file} in use")
            print("\n")
            time.sleep(1)


# calculate residence time from nondimensionalized input conditions
def get_residence_times(inputs):
    # Re-Dimensionalize Inputs
    v_CO = inputs[0] * (1.9 - 0.1) + 0.1  # mLn/min
    v_H2 = inputs[1] * (1.9 - 0.1) + 0.1
    v_gas = v_CO + v_H2

    pressure = inputs[2] * (300 - 150) + 150  # psi
    pressure_factor = (pressure / 14.7) + 1

    v_gas_p = v_gas * 1000 / pressure_factor  # uL/min
    v_liquid = v_gas_p / 3
    v_total = v_gas_p + v_liquid
    tau = 7000 / v_total  # Residence time in min

    return tau


def make_single_objective(y_obj, weights=None):
    # Pull Objective Tensor off GPU
    y_obj = y_obj.cpu() if y_obj.device.type == 'cuda' else y_obj
    # Default to equal weight average
    if weights is None:
        weights = 1 / y_obj.shape[1] * torch.ones(1, y_obj.shape[1], dtype=y_obj.dtype)
    # Create Weighted Average of all Objectives
    # Z = (W*Y')' (1xN)x(NxM) -> Mx1. N objectives M data points
    new_obj = torch.mm(weights, y_obj.transpose(1, 0)).transpose(1, 0)

    return new_obj


# create single randomized cascade model
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


# create ensemble of supplied model list
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
    ens_out = relu(ens_out, alpha=0.1, max_value=1)

    ensemble_model = keras.Model(inputs=inputs, outputs=[ens_out, cat_out])
    return ensemble_model


# Generate Ensemble NN with Training and Validation data. Retain some data for testing.
# Hidden_layers and nodes take integer lists [min_value, max_value].
def create_ensemble_nn(train_x, train_y, val_x, val_y, path, hidden_layers=None, nodes=None, num_models=10):
    # Avoiding List definition in function default parameters which can lead to unexpected results.
    # Change desired default parameters here.
    if hidden_layers is None:
        hidden_layers = [2, 5]
    if nodes is None:
        nodes = [15, 20]
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, mode='min')
    # Model compile with loss, optimizer, and performance metric of interest.
    ensemble.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.001),
                  metrics=[RootMeanSquaredError()])
    # Perform Model Training with training and validation data with early stopping.
    ensemble_history = ensemble.fit(train_x, [train_y, train_y_cat], validation_data=(val_x, [val_y, val_y_cat]),
                     epochs=500,
                     verbose=0,
                     callbacks=[early_stopping])
    # Save model to the filepath and add trained model + history to the ensemble lists
    filepath = path + fr'\ensemble_model'
    ensemble.save(filepath)

    # Return ensemble of Tensorflow models and training history for each member in the ensemble.
    return ensemble, ensemble_history


# Test ensemble for accuracy given test data and ensemble list of models.
def evaluate_ensemble(test_x, test_y, ensemble):
    # Instantiate list for individual model accuracies.
    accuracy = list()
    # Iterate over each model and evaluate testing data, append to list.
    for model in ensemble:
        _, test_acc = model.evaluate(test_x, test_y, verbose=0)
        accuracy.append(test_acc)

    # Return list of accuracies for each model in the ensemble.
    return accuracy


# Method for evaluating all ensemble models at inputs in x and returning average and standard deviation for each.
def sample_ensemble(x, ensemble, is_linear=1):
    # Instantiate list for averages.
    ensemble_y = torch.zeros(x.shape[0], ensemble[0].output_shape[1], len(ensemble))
    for i, model in enumerate(ensemble):
        # Sample x for each model in ensemble.
        ensemble_y[:, :, i] = torch.from_numpy(model.predict(np.array(x.cpu())))
        # Oxo Selectivity check.
        if not is_linear:
            ensemble_y[:, 1, i] = 1 - ensemble_y[:, 1, i]
    # Calculate mean and standard deviation across ensemble outputs.
    y = torch.mean(ensemble_y, 2)
    std = torch.std(ensemble_y, 2)

    # Return ensemble average and standard deviation for each point in x (num_conditions by input_dimension).
    return y, std


# Evaluate ensemble model at a given set of inputs x
def sample_cascade_ensemble(x, ensemble, is_linear=1):
    # averaged ensemble prediction
    ensemble_prediction = ensemble.predict(np.array(x.cpu()))
    y = torch.from_numpy(ensemble_prediction[0])
    # individual sub model predictions and uncertainty
    sub_yield = torch.from_numpy(ensemble_prediction[1][:, ::2])
    sub_sele = torch.from_numpy(ensemble_prediction[1][:, 1::2])
    # target selectivity fraction check
    if not is_linear:
        y[:, 1] = 1 - y[:, 1]
        sub_sele = 1 - sub_sele

    model_y = torch.zeros((ensemble_prediction[1].shape[0], 2, int(ensemble_prediction[1].shape[1]/2)))
    model_y[:, 0, :] = sub_yield
    model_y[:, 1, :] = sub_sele
    std = torch.std(model_y, 2)

    return y, std, model_y


# Max variance BOTORCH acquisition
def ensemble_optimize_MV_and_get_observation(ensemble, device_args, q=1):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # Input boundaries
    bounds = torch.tensor([[0.0] * 7, [1.0] * 7], device=device_args['device'], dtype=device_args['dtype'])
    # Monte Carlo sampling
    mc_samples = draw_sobol_samples(bounds=bounds, n=RAW_SAMPLES, q=1).squeeze(1)
    # evaluate samples on model
    y_obj, std_obj, _ = sample_cascade_ensemble(mc_samples, ensemble)
    std_obj = make_single_objective(std_obj)
    # rank samples by uncertainty
    idx_mv = sorted(range(len(std_obj)), key=lambda i: std_obj[i])[-q:]

    new_x = mc_samples[idx_mv]
    exact_obj = test_func(new_x)

    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return new_x, new_obj, exact_obj


# Exploitation acquisition function
def ensemble_optimize_explt_and_get_observation(ensemble, weights, device_args, q=1, is_linear=1):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # Input boundaries
    bounds = torch.tensor([[0.0] * 7, [1.0] * 7], device=device_args['device'], dtype=device_args['dtype'])
    # Monte Carlo Sampling
    mc_samples = draw_sobol_samples(bounds=bounds, n=RAW_SAMPLES, q=1).squeeze(1)
    # Evaluate samples on model
    y_obj, std_obj, _ = sample_cascade_ensemble(mc_samples, ensemble, is_linear=is_linear)
    y_obj = make_single_objective(y_obj, weights=weights)
    # Rank samples by mean value
    idx_explt = sorted(range(len(y_obj)), key=lambda i: y_obj[i])[-q:]
    new_x = mc_samples[idx_explt]
    exact_obj = test_func(new_x)

    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return new_x, new_obj, exact_obj


def ensemble_noisy_read_nn(x, ensemble, alpha, sigma, is_linear=1):
    # Sample NN model given x. y = model(x)
    y, std, _ = sample_cascade_ensemble(torch.from_numpy(x), ensemble, is_linear=is_linear)
    # Instantiate noise array and fill with normal gaussian noise of standard deviation sigma for a given model output.
    noise = np.zeros_like(y)
    for i, s in enumerate(sigma):
        noise[:, i] = np.random.normal(0, s, y.shape[0])
    # Sum true model output and noise.
    y = y + noise
    # Oxo Selectivity check
    if not is_linear:
        y[:, 1] = 1 - y[:, 1]
    # Perform alpha weighted average of noisy model outputs and return.
    z = np.dot(y, alpha)

    # Return weighted average acquisition function value.
    return z


def ensemble_optimize_dragonfly_and_get_observation(ensemble, alpha, sigma, device_args, q=1, is_linear=1):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    bounds = torch.tensor([[0.0] * 7, [1.0] * 7], device=device_args['device'], dtype=device_args['dtype'])
    # Create domain boundaries from data input shape. MUST BE NORMALIZED
    dom = [[0, 1]] * ensemble[0].input_shape[1]
    # Call Dragonfly maximize for normal selectivity
    min_val_n, min_pt_n, opt_history_n = maximize_function(lambda x: ensemble_noisy_read_nn(
        x.reshape(1, ensemble[0].input_shape[1]), ensemble, alpha, sigma, is_linear=is_linear)[0], dom, n)

    idx_explt = sorted(range(len(y_obj)), key=lambda i: y_obj[i])[-q:]
    new_x = mc_samples[idx_explt]
    exact_obj = test_func(new_x)

    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return new_x, new_obj, exact_obj


# sample model and create initial data for building BOTORCH Gaussian Process
def generate_initial_data(model, device_args, is_linear=1, n=5):
    # Create normalized boundaries [0, 1] from input data on correct botorch device
    bounds = torch.tensor([[0.] * model.input_shape[1], [1.] * model.input_shape[1]],
                          device=device_args['device'],
                          dtype=device_args['dtype'])
    # Monte Carlo sampling random values within bounds for each input dimension.
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)

    # Sample model at generated  points and add noise defined by NOISE_SE
    train_obj_true, std, _ = sample_cascade_ensemble(train_x.cpu(), model, is_linear=is_linear)
    train_obj_true = train_obj_true.to(device=device_args['device'])
    train_obj = train_obj_true + torch.randn_like(train_obj_true).to(device=device_args['device']) * NOISE_SE

    # Return Randomized initial values with noisy model evaluation and true model evaluation.
    return train_x, train_obj, train_obj_true


# Initialize BOTORCH gaussian process model
def initialize_model(train_x, train_obj):
    # Normalize input data given bounds. Data should be pre-normalized in most cases.
    train_x = normalize(train_x, torch.tensor([[0.] * train_x.shape[1], [1.] * train_x.shape[1]],
                                              dtype=torch.float64,
                                              device=train_x.device.type))
    # Instantiate GP model list
    models = []
    # Create GP model for each output in objective data.
    for i in range(train_obj.shape[-1]):
        # Extract objective data for single objective
        train_y = train_obj[..., i:i + 1]
        # Create Variance array from NOISE_SE in shape of train_y
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        # Add trained FixedNoiseGP for objective i to list of GP models
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
    # Create ModelListGP model for multi-objective botorch functions
    bo_model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(bo_model.likelihood, bo_model)

    # Return model marginal log likelihood and multi-objective Botorch GP model
    return mll, bo_model


# Obtain new BOTORCH qNEHVI predictions
def optimize_qnehvi_and_get_observation(model, bo_model, train_x, sampler, is_linear=1):
    # Pass inputs to Botorch qNEHVI: model, reference, baseline, and MC sampler
    # Reference point of [0]*N and trim estimates to within normalized boundaries [0, 1]
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=bo_model,
        ref_point=torch.from_numpy(np.array([0, 0])).to(device=train_x.device.type, dtype=torch.float64),
        X_baseline=normalize(train_x, torch.tensor([[0.] * model.input_shape[1], [1.] * model.input_shape[1]],
                                                   device=train_x.device.type,
                                                   dtype=torch.float64)),
        prune_baseline=True,
        sampler=sampler,
    )
    # Get list of possible candidates after optimizing qNEHVI
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # Get new possible inputs for maximizing hypervolume
    bounds = torch.tensor([[0.] * model.input_shape[1], [1.] * model.input_shape[1]],
                          device=train_x.device.type,
                          dtype=torch.float64)
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    # Sample new values from surrogate NN model and apply noise
    new_obj_true, std, _ = sample_cascade_ensemble(new_x.cpu(), model, is_linear=is_linear)
    new_obj_true = new_obj_true.to(device=train_x.device.type)
    new_obj = new_obj_true + torch.randn_like(new_obj_true).to(device=train_x.device.type) * NOISE_SE

    # Return new input values with noisy model evaluation and true model evaluation.
    return new_x, new_obj, new_obj_true


# qNEHVI wrapper and hypervolume partitioning
def ensemble_optimize_qnehvi_and_get_observation(ensemble, device_args, q=1, is_linear=1):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Instantiate lists for hypervolumes of different acquisition functions

    hvs_qnehvi = []

    # Generate initial data for all acquisition functions for normal optimization
    # Initial points = 2*(dim_inputs + 1)
    n = 256  # 2 * (ensemble[0].input_shape[1] + 1)
    train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = generate_initial_data(ensemble, device_args,
                                                                                    is_linear=is_linear, n=n)

    # Initialize model with initial data
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

    # Create hypervolume partitioning given reference point and data for normal optimization
    bd = DominatedPartitioning(ref_point=torch.zeros(ensemble.output_shape[0][1],
                                                     device=device_args['device'],
                                                     dtype=torch.float64),
                               Y=train_obj_true_qnehvi)
    # compute starting hypervolume
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)
    fit_gpytorch_model(mll_qnehvi)
    qnehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
    new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi = optimize_qnehvi_and_get_observation(
        ensemble, model_qnehvi, train_x_qnehvi, qnehvi_sampler
    )
    new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi = new_x_qnehvi[:q, :], new_obj_qnehvi[:q, :], new_obj_true_qnehvi[
                                                                                                    :q, :]
    return new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi


# filtering generated conditions by residence time and observed selectivity range
def condition_filter(ensemble, tkwargs, policy=2, q=1, max_tau=60, min_isomer_fraction=0.7, is_linear=1):
    # initialize allowed condition lists
    valid_x = torch.zeros([q, 7], device=tkwargs['device'])
    test_conditions = torch.zeros([1, 9], device=tkwargs['device'])
    i = 0
    loop = 0
    # while needing more valid conditions:
    while i < q:
        print(f'Optimization Cycle #{loop + 1}:\n')
        # if over 50 cycles have been performed, terminate with best observed condition to prevent infinite loop
        if loop > 50:
            print('Exceeded Loop Count, Using best obtained values:\n')
            temp_conditions = test_conditions[test_conditions[:, -1].sort().indices[-q:], :]
            valid_x = temp_conditions[:, :-2]
            for x in valid_x:
                print(x, end='\n')
            return valid_x

        # obtain new predicted conditions based on acquisition function
        if policy == '2':
            new_x, new_obj, exact_obj = ensemble_optimize_qnehvi_and_get_observation(ensemble, tkwargs, q=10,
                                                                                     is_linear=is_linear)
        else:
            new_x, new_obj, exact_obj = ensemble_optimize_MV_and_get_observation(ensemble, tkwargs, q=10)

        loop = loop + 1

        # evaluate predicted conditions for residence time
        for x, y in zip(new_x, new_obj):
            tau = get_residence_times(x)
            # Accept condition if residence time and selectivity pass check
            if tau < max_tau and i < q and y[1] > min_isomer_fraction:
                valid_x[i, :] = x
                i = i + 1
                test_conditions = torch.cat((test_conditions, torch.cat((x, y), dim=0).reshape([1, 9])), dim=0)
                print('Valid Condition:\n')
                print(x, end='\n')
                print(f'Tau = {tau}\n')
                print(f'Yield = {y[0]}\n')
                if is_linear:
                    print(f'Predicted Normal Isomer Fraction: {y[1]}\n')
                else:
                    print(f'Predicted Iso Isomer Fraction: {y[1]}\n')
            # selectivity condition failed, resample
            elif tau < max_tau and i < q:
                test_conditions = torch.cat((test_conditions, torch.cat((x, y), dim=0).reshape([1, 9])), dim=0)
                if is_linear:
                    print(f'Invalid X_n, Tau = {tau}, Yield = {y[0]}, X_n = {y[1]}. Resampling')
                else:
                    print(f'Invalid X_i, Tau = {tau}, Yield = {y[0]}, X_i = {y[1]}. Resampling')
            # obtained all requested new conditions
            elif i >= q:
                print('Obtained Valid Conditions')
                break
            # residence time invalid, resample
            else:
                if is_linear:
                    print(f'Invalid Condition, Tau = {tau}, Yield = {y[0]}, X_n = {y[1]}. Resampling')
                else:
                    print(f'Invalid Condition, Tau = {tau}, Yield = {y[0]}, X_i = {y[1]}. Resampling')

    return valid_x


# Initialization script
if __name__ == '__main__':

    # Imports
    import torch

    import math
    import numpy as np
    import random
    import pandas as pd

    from tkinter.filedialog import askdirectory
    from tkinter import messagebox

    from tensorflow.keras.activations import relu
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import RootMeanSquaredError
    from tensorflow import keras
    from tensorflow.keras import layers
    from dragonfly import maximize_function

    from botorch.models.gp_regression import FixedNoiseGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    from botorch.utils.transforms import unnormalize, normalize
    from botorch.utils.sampling import draw_sobol_samples

    from botorch.optim.optimize import optimize_acqf
    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

    from botorch import fit_gpytorch_model
    from botorch.sampling.samplers import SobolQMCNormalSampler
    from botorch.exceptions import BadInitialCandidatesWarning
    from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

    import time
    import warnings

    # Getting devices for gpu methods
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # user input for test cycle numbers vs full run
    test = input('Test Run? (y/[n]):')
    if test.lower() == 'y':
        SMOKE_TEST = True
    else:
        SMOKE_TEST = False

    # User defined Noise
    NOISE_SE = torch.tensor([0.02, 0.02], **tkwargs)
    # points per batch
    BATCH_SIZE = 10
    # number of monte carlo restarts per cycle if no better condition has been found
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    # Number of models in Ensemble
    N_MODELS = 10 if not SMOKE_TEST else 2
    # number of monte carlo samples on GP
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    # Number of batches of points
    N_BATCH = 50 if not SMOKE_TEST else 10
    # Initial number of MC samples to define model
    MC_SAMPLES = 512 if not SMOKE_TEST else 16

    # User supplied directory containing prior data and hardware control .csv
    messagebox.showinfo("Input Directory", "Please select directory")
    project_dir = askdirectory()

    # Directory for supplying new GC analysis files as .txt
    messagebox.showinfo("GC Output File Directory", "Please select directory")
    GC_path = askdirectory()

    # Select acquisition function from implemented list
    policy = input('Choose Selection Policy ([1: MV], 2: qNEHVI):')

    # Choose aldehyde target
    is_linear = input('Choose Optimization Target ([1: Linear], 0: Branched):')
    if int(is_linear) != 0:
        is_linear = 1
    else:
        is_linear = 0

    # normalized bounds for data
    standard_bounds = torch.zeros(2, 7, **tkwargs)
    standard_bounds[1] = 1

    # Import prior data and find range
    npins = pd.read_csv(project_dir + "//" + "HFNonDimInputs.csv", header=None).to_numpy()
    dim_in = npins.shape[1]
    npouts = pd.read_csv(project_dir + "//" + "HFNonDimOutputs.csv", header=None).to_numpy()
    dim_out = npouts.shape[1]
    out_shape = npouts.shape[0]
    npins = npins[:out_shape, :]
    sele_max = np.max(npouts[:, 1])
    sele_min = np.min(npouts[:, 1])
    sele_range = sele_max - sele_min

    # Concatenate input and output data
    npdata = np.concatenate([npins, npouts], axis=1)

    # Shuffle input data
    rng = np.random.default_rng()
    rng.shuffle(npdata)

    # Training Testing Validation Split
    n_train = math.floor(0.8 * npins.shape[0])

    trainX, valX = npdata[:n_train, :-1 * dim_out], npdata[n_train:, :-1 * dim_out]
    trainy, valy = npdata[:n_train, -1 * dim_out:], npdata[n_train:, -1 * dim_out:]

    # Create ensemble NN model
    ensemble, ensemble_history = create_ensemble_nn(trainX, trainy, valX, valy, project_dir, num_models=N_MODELS)

    # Maximum allowed residence time
    max_tau = 60

    # Selectivity cutoff based on observed inputs
    min_isomer_fraction = sele_max - 0.25 * sele_range
    if not is_linear:
        min_isomer_fraction = 1 - (sele_min + 0.25 * sele_range)

    print(f'Minimum Isomer Fraction Threshold:{min_isomer_fraction}, Is_Linear:{is_linear}')

    # Obtain new allowed conditions
    valid_x = condition_filter(ensemble, tkwargs, policy=policy, q=2, max_tau=max_tau,
                               min_isomer_fraction=min_isomer_fraction, is_linear=is_linear)

    # Append valid conditions to hardware control .csv and HFNonDimInputs.csv
    for point in valid_x:
        settings = generate_equipment_settings(point.reshape(7).cpu().numpy(), project_dir)

    test_run_gc(valid_x, project_dir)
    # Initialize Observer for GC Output Files
    pattern = ["*.txt"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    handler = PatternMatchingEventHandler(pattern, ignore_patterns, ignore_directories, case_sensitive)

    handler.on_created = on_created
    handler.on_deleted = on_deleted
    handler.on_moved = on_moved
    handler.on_modified = on_modified

    run_recursive = True
    observer = Observer()
    observer.schedule(handler, GC_path, run_recursive)

    observer.start()


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
