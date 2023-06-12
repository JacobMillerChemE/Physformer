import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import yaml
import os.path as path


def load_config(config_path: str) -> dict:
    """Takes the path to the config file and returns the config dictionary."""
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def integration_time(config, mode: str) -> tuple[tuple, np.ndarray]: 
    """Takes config file/mode & returns the time span & list of time steps.
    
    This function takes the configuration file and mode as input. Mode can be:
    t_train, t_test, or t_val. Mode is used to create t_span which contains the 
    initial and final time of the simulation. The function returns t_span and 
    t which is a 1D numpy array of time values starting at t_span[0] and ending
    at t_span[1] each separated by size dt. 
    """

    dt = config["dt"]  # Time step size of simulation from config file
    t_span = (0, config[mode])  # Initial and final times
    t = np.arange(t_span[0], t_span[1], dt)  # 1D array of times 
    return t_span, t


def initial_state_generator() -> tuple:
    """Generates and returns random integer initial values for x, y, and z."""

    x0 = np.random.uniform(-20, 20)  # Sample float from -20 < x < 20
    y0 = np.random.uniform(-20, 20)
    z0 = np.random.uniform(-20, 20)
    f0 = (x0, y0, z0)
    return f0


def lorenz_system(t: np.ndarray, initial_state: tuple) -> list:
    """Takes list of times and tuple with initial conditions and returns a list.
    
    This function is defined so to be used in the SciPy solve_ivp solver. It
    accepts a list of times to be evaluated at and a tuple of initial states
    which is necessary for solve_ivp. Sigma, rho, and beta are parameters
    that define the behavior of the system and are kept constant. The function
    returns a list of values for each ODE for x, y, and z. 
    """


    sigma = 10
    rho = 28
    beta = 8/3
    x, y, z = initial_state

    dx = sigma*(y-x)
    dy = x*(rho-z) - y
    dz = x*y - beta*z

    return [dx, dy, dz]


def simulation_count_splitter(sim_num: int, train_val_test_split: tuple):
    """Takes total number of sims and splits it amongst train, val and test.

    This function takes the total number of desired simulation for the 
    experiment and breaks it into three parts according to the split provided
    in train_val_test_split. It returns a tuple of integer values corresponding 
    to the number of simulations for the training, validation and test sets. 
    
    """
    # Unpack split values (e.g. train_val_test_split = (0.7, 0.2, 0.1))
    train_split, validation_split, test_split = train_val_test_split

    # Calculate the number of simulations by multiplying total sims by split
    train_sim_count = train_split*sim_num  # E.g. 3000 * 0.7 = 2100
    validation_sim_count = validation_split*sim_num
    test_sim_count = test_split*sim_num

    # Check to make sure all splits are even integer values to avoid confusion
    assert train_sim_count == int(train_sim_count), \
        "No even training split"
    assert validation_sim_count == int(validation_sim_count), \
        "No even validation split"
    assert test_sim_count == int(test_sim_count), \
        "No even test split"
    
    return train_sim_count, validation_sim_count, test_sim_count


def lorenz_solver(sim_num: int, f0_list: set, config, mode: str) -> tuple:
    """Returns the solution to the Lorenz system given a set of ICs. 

    This function accepts the number of simulations for a given dataset (e.g.
    training, validation or testing), a list of initial conditions already used,
    and the mode (e.g. training/testing) and returns a dataframe that contains
    the solution of the Lorenz equation for each simulation. The dataframe 
    returned is of shape: (3, number of timesteps, number of simulations).

    Parameters: 
    -----------
    sim_num : int
        Number of simulations for training, validation or testing
    f0_list : set
        A set of initial conditions already used
    config : dict
        The dictionary generated from yaml config file
    mode : str
        Can be t_train, t_val, or t_test 
    
    Returns:
    --------
    data_frame : np.ndarray
        A numpy array of shape (3, number of timesteps, number of simulations)
    f0_list : 
        Update list of used initial conditions

    """
    # See integration_time defintion for meaning of "mode"
    t_span, t = integration_time(config, mode)  # Generate time steps

    # Initialize data frame to store simulation results
    data_frame = np.zeros((config['lorenz_dim'], len(t), int(sim_num)))

    i = 0
    while i < sim_num:
        f0 = initial_state_generator()  # Generate initial state
        # Check to see if initial condition has been used. We do this to make
        # sure there are no shared data point in training, validation & testing
        if f0 in f0_list:
            print("Same initial conditions!")  # Note we don't increment i
            continue
        else:
            f0_list.add(f0)  # Add initial condition to initial condition list
            soln = solve_ivp(lorenz_system, t_span, f0, t_eval=t)  #solve Lorenz
            data_frame[:, :, i] = soln.y  # Insert solution at simulation slice
            i += 1  # Increment i
    return data_frame, f0_list


def data_generator(sim_num, config, train_val_test_split):
    """Creates the training, validation and testing data sets.
    
    Accepts the total number of simulations and the desired train/test split. 
    Generates training, validation and test data sets split according to the 
    train/test split and ensure that each simulation was given a unique set 
    of initial conditions. Returns the training, validation and test sets. 
    """
    f0_list = set()

    (train_sim_count, 
     validation_sim_count, 
     test_sim_count) = simulation_count_splitter(sim_num, train_val_test_split)
    
    train_data, f0_list = lorenz_solver(train_sim_count, 
                                        f0_list, 
                                        config,
                                        mode='t_train')
    validation_data, f0_list = lorenz_solver(validation_sim_count, 
                                             f0_list,
                                             config,
                                             mode='t_val')
    test_data, f0_list = lorenz_solver(test_sim_count,
                                       f0_list, 
                                       config, 
                                       mode='t_test')
    return train_data, validation_data, test_data


def feature_scaling(data: np.ndarray, scalar_dict=None) -> tuple:
    """Scales x, y, z values of Lorenz to be between -1 and 1.
    
    Accepts a numpy array of shape (3, # of time steps, # of simulations) and
    performs a form of min max scaling that scales the data between -1 and 1. If
    a dictionary of scaling values are not provided one is created and returned.

    Parameters:
    -----------
    data : np.ndarray
        A numpy array that contains the data for the Lorenz simulations 
    scalar_dict : dict
        A dictionary that contains the scaling values for x, y and z if the 
        dataset input is not the training dataset
    
    Returns:
    --------
    data : np.ndarray
        A numpy array of scaled features
    scalar_dict : dict
        A dictionary containing the scalars for the x,y and z variables
    
    Notes:
    ------
    I created this function to scale the input feautures between -1 and 1, 
    because normal min max scaling scales between 0 and 1. I wanted to retain 
    the ability to have negative values for the Lorenz data. Also, 
    the if statement is to ensure that the training data is always used to 
    determine scaling factors. The scaling factors should always come from
    the dataset to avoid data leaks from testingg and to ensure any inferences
    are occuring on a similar distribution as the training data. 

    """
    if scalar_dict == None:  # Check to see if it's the training dataset

        # Create copies of input data array
        #FIXME 
        data = np.copy(data)
        all_x_values = data[0, :, :]
        all_y_values = data[1, :, :]
        all_z_values = data[2, :, :]

        # Find min and max vaues for x, y and z values
        x_min, x_max = np.min(all_x_values), np.max(all_x_values)
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
        z_min, z_max = np.min(all_z_values), np.max(all_z_values)

        # Calculate scaling scalars to be used for testing and validation data
        # sets.
        x_range, x_offset = x_max - x_min, x_min / (x_max-x_min)
        y_range, y_offset = y_max - y_min, y_min / (y_max-y_min)
        z_range, z_offset = z_max - z_min, z_min / (z_max-z_min)

        # Scale the training data set using the scalars to push data -1 < x < 1
        data[0, :, :] = 2*(all_x_values/x_range - x_offset) - 1
        data[1, :, :] = 2*(all_y_values/y_range - y_offset) - 1
        data[2, :, :] = 2*(all_z_values/z_range - z_offset) - 1

        # Create a dictionary of scalars to be used for validation and testing
        scalar_dict = {"x": (x_range, x_offset),
                       "y": (y_range, y_offset),
                       "z": (z_range, z_offset)}
        return data, scalar_dict
    else: 
        data = np.copy(data)

        # If the dataset is testing or validation, unpack scalars for x,y, and z
        x_range, x_offset = scalar_dict["x"][0], scalar_dict["x"][1]
        y_range, y_offset = scalar_dict["y"][0], scalar_dict["y"][1]
        z_range, z_offset = scalar_dict["z"][0], scalar_dict["z"][1]

        # Scale the data between -1 and 1
        data[0, :, :] = 2*(data[0, :, :]/x_range - x_offset) - 1
        data[1, :, :] = 2*(data[1, :, :]/y_range - y_offset) - 1
        data[2, :, :] = 2*(data[2, :, :]/z_range - z_offset) - 1
        return data


def data_chunker(data, data_scaled, config):
    """Converts raw Lorenz data into a format to be used for Physformer training

    This function accepts the raw Lorenz data set and a scaled version of that 
    same dataset to create new datasets that can be used for training 
    Physformer. The input data is "chunked" using the context window so that  
    new data is of shape:
        (Number of training points, context length, pre-embedding dimension)
    The function creates and saves two datasets: a scaled dataset used for 
    feature input to Physformer and an unscaled dataset used for targets for 
    Physformer. 

    Parameters:
    -----------
    data : np.ndarray
        A numpy array containing the solution for the Lorenz simulation for 
        multiple simulations
    data_scaled : np.ndarray
        A numpy array that is equal to the data array when passed through the 
        feature scaling function
    config : dict
        Dictionary generated from config file


    Notes:
    ------
    "Chunked" is probably not the correct terminology to describe the saved 
    data frame. To be specific, this function takes the dataset from the 
    Lorenz simulations, which have shape (3, # of time steps, # of simulations),
    and breaks it into multiple data points for each simulation. The number of 
    time steps of each data points is equal to the context length used by 
    Physformer. For example, if the simulation is 256 time steps and the context
    length is 16, then there will be 256 - 16 = 240 data points generated from 
    that simulation. This corresponds to Physformer using time steps (n) n:n+16 
    to predict n+17. 
    
    """
    num_timesteps = data.shape[1]
    num_sims = data.shape[-1]
    context_length = config['block_size']
    # The number of data points generated for each simulation
    num_sequences_per_sim = int(num_timesteps - context_length)
    # The total number of data points for the entire dataset
    num_data_points = num_sims*num_sequences_per_sim
    input_dim = config['lorenz_dim']

    # Initialize feature and target arrays
    input_array = np.zeros((num_data_points, context_length, input_dim))
    target_array = np.zeros((num_data_points, context_length, input_dim))

    for sim in tqdm(range(num_sims), "Parsing dataset"):
        for t in range(num_sequences_per_sim):
            # For each time step in each simulation, create a feature array 
            # from data_scaled with the number of time steps equal to 
            # t -> t+context_length. And create a target array shifted over 
            # one time step that will be used for calculating loss in Physformer
            x = data_scaled[:, t:t+context_length, sim]
            y = data[:, t+1:t+context_length+1, sim]
            input_array[t+(sim*num_sequences_per_sim), :, :] = x.T
            target_array[t+(sim*num_sequences_per_sim), :, :] = y.T
    
    return input_array, target_array



if __name__ == '__main__':

    # Define configuration and root file paths
    CONFIG_PATH = "packages/configs/base.yaml"
    root_dir = "packages/data"
    
    # Load configuration file, define total # of sims, and how to split the sims
    config = load_config(CONFIG_PATH)
    sim_count = 3000
    train_val_test = config['data_split']

    # Generates raw data matrices of shape: (3, # of time steps, # of sims)
    (train_data, 
     val_data, 
     test_data) = data_generator(sim_count, config, train_val_test)
    
    # Save the raw data for safe keeping
    np.save(path.join(root_dir, "training_data/raw.npy"), train_data)
    np.save(path.join(root_dir, "validation_data/raw.npy"), val_data)
    np.save(path.join(root_dir, "test_data/raw.npy"), test_data)

    

    



