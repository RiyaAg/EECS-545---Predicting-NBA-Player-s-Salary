import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


def one_hot_encode(data_frame: pd.DataFrame, category: str):
    """Create a one-hot encoding for a category"""
    
    one_hot = pd.get_dummies(data_frame[category])
    data_frame = data_frame.drop(category, axis=1)
    return data_frame.join(one_hot)


def get_cleaned_baseline_data(file: str, encoding=True):
    """Get Cleaned Data from just the baseline dataset. If encoding is set to true, it will One-Hot-Encode the categorical features."""
    
    nba_initial = pd.read_csv(file, index_col=[0])
    nba = nba_initial.dropna()
    
    if encoding:
        # One-hot encoding for the position
        nba = one_hot_encode(nba, 'position')
        
        # One-hot encoding for the team
        nba = one_hot_encode(nba, 'team')
    
    return nba


def add_log_y_values(dataset: pd.DataFrame):
    dataset['inflationAdjSalary_log'] = dataset['inflationAdjSalary'].apply(lambda x: np.log(x))
    return dataset


def get_X_y_vals(dataset: pd.DataFrame):
    """Get X and y values for dataset"""
    nba_values = dataset.values
    
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
    
    
    
    
    

# May want to consider averaging the parameters of two different models where one is tested on data that incorporates a certain statistic while another includes data which doesn't incorporate that statistic
# can also experiment with the best method for handling these nan values https://www.forbes.com/sites/markdeeks/2022/07/01/a-complete-history-of-nba-luxury-tax-payments-20012022/?sh=75991719432f
