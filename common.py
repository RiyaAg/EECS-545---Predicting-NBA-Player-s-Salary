import pickle
import os
from pathlib import Path
from models.preprocessing import get_baseline_data
import pandas as pd
import numpy as np

# Returns the salary class of the given salaryl
def get_salary_class(salary: float, max: float, min:float) :
    num_classes = 10
    (max-min)/num_classes
    return 0

def get_baseline_data(file: str) :
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace('$', '')
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace(',', '')
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].astype(int)
    nba['inflationAdjSalary_log'] = nba['inflationAdjSalary'].apply(lambda x: np.log(x))
    return nba

NN_FILENAME = 'neural_network.model'
RF_FILENAME = 'random_forest.model'
def get_nn() :
    p = Path(os.path.abspath(__file__)).parent / ('models/saved_models/' + NN_FILENAME)
    return pickle.load(open(p, 'rb'))
    
def get_nn_features() :
    return ['MP', 'PTS', 'Age', 'games', 'games_started', 'PER', 'FTr', 'AST', 'STL', 'TRB', 'FT', '3P', 'FG']
    
def get_rf() :
    p = Path(os.path.abspath(__file__)).parent / ('models/saved_models/' + RF_FILENAME)
    return pickle.load(open(p, 'rb'))

# May want to consider averaging the parameters of two different models where one is tested on data that incorporates a certain statistic while another includes data which doesn't incorporate that statistic
# can also experiment with the best method for handling these nan values https://www.forbes.com/sites/markdeeks/2022/07/01/a-complete-history-of-nba-luxury-tax-payments-20012022/?sh=75991719432f
