import pandas as pd
import numpy as np


def get_player_year(dataframe: pd.DataFrame, name: str):
    return dataframe[dataframe['playerName'] == name]


def add_prev_year_salary(nba_cleaned: pd.DataFrame):
    # nba_cleaned['prevYearSalary'] = np.nan
    name_index = nba_cleaned.columns.get_loc('playerName')
    
    # print(nba_cleaned)
    for start_year in range(1997, 2018):
        prev_year = nba_cleaned[nba_cleaned['seasonStartYear'] == start_year - 1]
        cur_year = nba_cleaned[nba_cleaned['seasonStartYear'] == start_year]
        # print(cur_year)
        for row in cur_year.values:
            name = row[name_index]
            prev_year_val = get_player_year(prev_year, name)
            # print(prev_year_val)
            if not prev_year_val.empty:
                prev_salary = prev_year_val['inflationAdjSalary'].values[0]
                # print(log_salary)
                nba_cleaned.loc[(nba_cleaned['seasonStartYear'] == start_year) & (nba_cleaned['playerName'] == name), 'prevYearSalary'] = prev_salary
                # print(prev_year_val['inflationAdjSalary'])
                # print(nba_cleaned.loc[(nba_cleaned['seasonStartYear'] == start_year) & (nba_cleaned['playerName'] == name)])
                # cur_year['prevYearSalaryLog'] = np.log(prev_year_val['inflationAdjSalary'])
            # print(nba_cleaned[nba_cleaned['seasonStartYear'] == start_year])
            
            # break
            # row['prevYearSalary'] = 
        # print(nba_cleaned[nba_cleaned['seasonStartYear'] == start_year])
        # break
    nba_cleaned = nba_cleaned[nba_cleaned['seasonStartYear'] != 1996]
    
    return nba_cleaned

if __name__ == "__main__":
    nba_cleaned = pd.read_csv('cleaned_data/base_cleaned.csv')
    nba_cleaned = add_prev_year_salary(nba_cleaned)
    
    
    nba_cleaned.to_csv('cleaned_data/base_cleaned_prev_year_salary.csv')
    
        