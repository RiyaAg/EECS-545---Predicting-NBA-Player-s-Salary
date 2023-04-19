import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import math
from copy import deepcopy
from sklearn.model_selection import train_test_split


# Constants
NEXT_Y_SAL = 'next_year_salary'
PREV_Y_SAL = 'prevYearSalary'
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
    return ['seasonStartYear', 'MP', 'PTS', 'Age', 'games', 'games_started', 'PER', 'FTr', 'AST', 'STL', 'TRB', 'FT', '3P', 'FG', 'height', 'weight', 'position', 'team', 'inflationAdjSalary']

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




# FROM PREPROCESSING


def one_hot_encode(data_frame: pd.DataFrame, category: str):
    """Create a one-hot encoding for a category"""
    
    one_hot = pd.get_dummies(data_frame[category])
    data_frame = data_frame.drop(category, axis=1)
    return data_frame.join(one_hot)


def get_cleaned_baseline_data(prev_year_salary=False, encoding=False):
    """Get Cleaned Data from just the baseline dataset. If encoding is set to true, it will One-Hot-Encode the categorical features."""
    
    file = '../data/cleaned_data/base_cleaned.csv'
    features = get_base_features()
    
    if prev_year_salary:
        file = '../data/cleaned_data/base_cleaned_prev_year_salary.csv'
        features = features + [PREV_Y_SAL]
    
    
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    nba = nba[features]
    
    if encoding:
        # One-hot encoding for the position
        nba = one_hot_encode(nba, 'position')
        
        # One-hot encoding for the team
        nba = one_hot_encode(nba, 'team')
    else:
        nba = nba.drop(['position', 'team'], axis=1)
    
    return nba


def get_cleaned_external_data(prev_year_salary=False, next_year_salary=False, encoding=False):
    """Get Cleaned External Data"""
    
    file = '../data/cleaned_data/external_cleaned.csv'
    features = get_extern_features()
    
    if prev_year_salary:
        file = '../data/cleaned_data/external_cleaned_prev_year_salary.csv'
        features = features + [PREV_Y_SAL]
    elif next_year_salary:
        file = '../data/cleaned_data/external_cleaned_next_year_salary.csv'
        features = features + [NEXT_Y_SAL]
    
    nba = pd.read_csv(file)
    nba = nba.dropna()
    nba = nba.drop_duplicates()
    nba = nba[features]
    
    if encoding:
        nba = one_hot_encode(nba, 'position')
        
        nba = one_hot_encode(nba, 'team')
    else:
        nba = nba.drop(['position', 'team'], axis=1)
        
    return nba


def add_log_y_values(dataset: pd.DataFrame):
    # dataset['prevYearSalaryLog'] = dataset[PREV_Y_SAL].apply(lambda x: np.log(x))
    dataset['inflationAdjSalary_log'] = dataset['inflationAdjSalary'].apply(lambda x: np.log(x))
    # dataset = dataset.drop(['inflationAdjSalary', 'salary', PREV_Y_SAL, 'playerName'], axis=1)
    dataset = dataset.drop(['inflationAdjSalary'], axis=1)
    return dataset


def get_X_y_vals(dataset: pd.DataFrame):
    """Get X and y values for dataset"""
    nba_values = dataset.values
    
    # X = nba_values[:, 1:-1]
    # y = nba_values[:, -1]
    X = deepcopy(dataset)
    y = deepcopy(dataset)
    

    X = nba_values[:, 1:-1]
    y = nba_values[:, -1]
    return X, y


def split_data(dataset: pd.DataFrame, num_years_test=3, time_based_split=True, test_proportion=0.25):
    """Split data into training and testing sets"""
    
    if time_based_split:
        latest_year = np.max(dataset['seasonStartYear'])

        # Split dataset by the season start year
        train_set = dataset[dataset['seasonStartYear'] <= latest_year - num_years_test]
        test_set = dataset[dataset['seasonStartYear'] > latest_year - num_years_test]
    
        X_train, y_train = get_X_y_vals(train_set)
        X_test, y_test = get_X_y_vals(test_set)
    else:
        
        X, y = get_X_y_vals(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=42)
    
    
    return X_train, X_test, y_train, y_test