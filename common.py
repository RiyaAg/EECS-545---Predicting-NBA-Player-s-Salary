import pandas as pd
import numpy as np


def hello():
    print("Hello from common")


def get_baseline_data(file: str) :
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace('$', '')
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace(',', '')
    # nba['inflationAdjSalary'] = nba['inflationAdjSalary'].astype(int)
    nba['inflationAdjSalary_log'] = nba['inflationAdjSalary'].apply(lambda x: np.log(x))
    return nba


def get_cleaned_baseline_data(file: str, encoding=True):
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    
    nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace('$', '')
    nba['inflationAdjSalary'] = nba['inflationAdjSalary'].str.replace(',', '')
    nba['inflationAdjSalary'] = nba['inflationAdjSalary'].astype(int)
    
    
    """Get Cleaned Data from just the baseline dataset. If encoding is set to true, it will One-Hot-Encode the categorical features."""
    

# May want to consider averaging the parameters of two different models where one is tested on data that incorporates a certain statistic while another includes data which doesn't incorporate that statistic
# can also experiment with the best method for handling these nan values https://www.forbes.com/sites/markdeeks/2022/07/01/a-complete-history-of-nba-luxury-tax-payments-20012022/?sh=75991719432f
