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
    return ['MP', 'PTS', 'Age', 'games', 'games_started', 'PER', 'FTr', 'AST', 'STL', 'TRB', 'FT', '3P', 'FG']

def get_extern_features() :
    return get_base_features() + ['height', 'weight','all_star_total', 'all_star_enc', 'all_nba_enc', 'all_nba_total','draft_pick', 'champion', 'conference_champ', 
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
