import pickle
import os
from pathlib import Path

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
