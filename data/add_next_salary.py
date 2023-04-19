import pandas as pd

# Give yourself access to common
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from common import NEXT_Y_SAL, EXT_NEXT_Y_FILENAME, SZN_START_Y, P_NAME, INF_ADJ_SAL

def add_next_year_salary(nba_cleaned: pd.DataFrame):
    # nba_cleaned['prevYearSalary'] = np.nan
    name_index = nba_cleaned.columns.get_loc(P_NAME)
    
    # print(nba_cleaned)
    for start_year in range(1996, 2017):
        next_year = nba_cleaned[nba_cleaned[SZN_START_Y] == start_year + 1]
        cur_year = nba_cleaned[nba_cleaned[SZN_START_Y] == start_year]
        # print(cur_year)
        for row in cur_year.values:
            name = row[name_index]
            next_year_val = next_year[next_year[P_NAME] == name]
            # print(prev_year_val)
            if not next_year_val.empty:
                next_salary = next_year_val[INF_ADJ_SAL].values[0]
                # print(log_salary)
                nba_cleaned.loc[(nba_cleaned[SZN_START_Y] == start_year) & (nba_cleaned[P_NAME] == name), NEXT_Y_SAL] = next_salary
                # print(prev_year_val['inflationAdjSalary'])
                # print(nba_cleaned.loc[(nba_cleaned[SZN_START_Y] == start_year) & (nba_cleaned[P_NAME] == name)])
                # cur_year['prevYearSalaryLog'] = np.log(prev_year_val['inflationAdjSalary'])
            # print(nba_cleaned[nba_cleaned[SZN_START_Y] == start_year])
            
            # break
            # row['prevYearSalary'] = 
        # print(nba_cleaned[nba_cleaned[SZN_START_Y] == start_year])
        # break
    nba_cleaned = nba_cleaned[nba_cleaned[SZN_START_Y] != 1996]
    
    return nba_cleaned

if __name__ == "__main__":
    nba_cleaned = pd.read_csv('cleaned_data/base_cleaned.csv')
    nba_cleaned = add_next_year_salary(nba_cleaned)
    
    
    nba_cleaned.to_csv('cleaned_data/' + 'base_cleaned_next_year_salary.csv')