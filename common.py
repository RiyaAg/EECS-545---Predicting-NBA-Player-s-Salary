import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

# Constants
NEXT_Y_SAL = 'next_year_salary'
INF_ADJ_SAL = 'inflationAdjSalary'
SZN_START_Y = 'seasonStartYear'
P_NAME = 'playerName'
NN_FILENAME = 'neural_network.model'
RF_FILENAME = 'random_forest.model'
EXT_NEXT_Y_FILENAME = 'external_cleaned_next_year_salary.csv'

# Returns the salary class of the given salaryl
def get_salary_class(salary: float, max: float, min:float) :
    num_classes = 10
    interval = (max-min)/num_classes
    return math.floor((salary - min)/interval)

def get_baseline_data(file: str) :
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace('$', '')
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace(',', '')
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].astype(int)
    nba['inflationAdjSalary_log'] = nba['inflationAdjSalary'].apply(lambda x: np.log(x))
    return nba

def get_nn() :
    p = Path(os.path.abspath(__file__)).parent / ('models/saved_models/' + NN_FILENAME)
    return pickle.load(open(p, 'rb'))

def get_base_features() :
    return ['MP', 'PTS', 'Age', 'games', 'games_started', 'PER', 'FTr', 'AST', 'STL', 'TRB', 'FT', '3P', 'FG', 'height', 'weight']

def get_extern_features() :
    return get_base_features() + ['startYear', 'all_star_total', 'all_star_enc', 'all_nba_enc', 'all_nba_total','draft_pick', 'champion', 'conference_champ', 
       'mvp', 'mvp_rank','mvp_total', 'player_week_enc', 'player_week_total', 'dpoy','dpoy_rank', 'dpoy_total']

def get_rf() :
    p = Path(os.path.abspath(__file__)).parent / ('models/saved_models/' + RF_FILENAME)
    return pickle.load(open(p, 'rb'))

def add_log_y_values(dataset: pd.DataFrame):
    dataset['inflationAdjSalary_log'] = dataset['inflationAdjSalary'].apply(lambda x: np.log(x))
    return dataset

def get_next_year_external_data() :
    p = Path(os.path.abspath(__file__)).parent / ('data/cleaned_data/' + EXT_NEXT_Y_FILENAME)
    nba = pd.read_csv(p)
    nba = add_log_y_values(nba)
    nba = nba.drop(columns=['Unnamed: 0.2', 'Unnamed: 0', 'Unnamed: 0.1', 'college','archetype','all_star','all_nba', 'position','team'])
    nba = nba.dropna()
    nba = nba.loc[:, (nba != 0).any(axis=0)]
    return nba

import numpy as np
from scipy import stats

def confidence_interval_numpy(predicted_array, actual_array, alpha=0.95):
    # Calculate the error between predicted and actual values
    # error_array = np.sqrt(np.square(predicted_array - actual_array))
    # error_array = predicted_array - actual_array
    error_array = np.abs(predicted_array - actual_array)
    
    # Calculate the mean error and standard error of the mean (SEM) for each column
    mean_error = error_array.mean(axis=0)
    sem = error_array.std(axis=0, ddof=1) / np.sqrt(len(error_array)) 
    # this is used as an approximation of the actual std. dev. because we 
    # don't know the actual std. dev and are using an approximation we need to 
    # use the t-distribution here instead of approximating the difference 
    # between the sample mean and the poopulation mean with a normal distribution
    # This confidence interval tells you how confident we are that the true RMSE value lies within the provided bounds

    # Calculate the degrees of freedom (a t distribution thing)
    dof = len(error_array) - 1

    # Calculate the confidence interval using the t-distribution
    t_distribution = stats.t.ppf((1 + alpha) / 2, dof)
    margin_of_error = t_distribution * sem
    confidence_interval = np.column_stack((mean_error - margin_of_error, mean_error + margin_of_error))

    return confidence_interval
    
def bootstrap_confidence_interval(predicted, actual, alpha=0.95, n_samples=1000):
    # Calculate the error between predicted and actual values
    # error = np.sqrt(np.square(predicted - actual))
    # error = predicted - actual
    error = np.abs(predicted - actual)
    # Initialize an empty dictionary to store confidence intervals
    confidence_interval = dict()

    # Iterate over columns
    for i in range(error.shape[1]):
        # Sample from the error values many times
        bootstrap_samples = np.random.choice(error[:, i], (n_samples, len(error)))

        # Calculate the mean error for each sample
        bootstrap_means = np.mean(bootstrap_samples, axis=1)

        # Calculate the confidence interval
        lower_bound = np.percentile(bootstrap_means, (1 - alpha) / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 + alpha) / 2 * 100)
        confidence_interval[f"column_{i}"] = (lower_bound, upper_bound)

    return confidence_interval