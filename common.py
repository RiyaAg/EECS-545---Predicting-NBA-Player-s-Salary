import pandas as pd
import numpy as np

def get_baseline_data(file: str) :
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace('$', '')
    nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace(',', '')
    nba['inflationAdjSalary'] = nba['inflationAdjSalary'].astype(int)
    nba['inflationAdjSalary_log'] = nba['inflationAdjSalary'].apply(lambda x: np.log(x))
    X = nba[['MP', 'PTS', 'Age', 'games_started', 'PER', 'AST', 'STL', 'TRB', '3P']]
    y = nba[['inflationAdjSalary_log']]
    return nba