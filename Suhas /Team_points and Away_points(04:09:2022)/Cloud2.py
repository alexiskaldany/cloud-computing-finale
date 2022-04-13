#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('/Users/suhasburavalla/Desktop/Dataset.csv' )

#%%
cols = ['gmDate', 'gmTime', 'seasTyp', 'teamAbbr','teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFG%', 'team3P%', 'teamFT%', 'teamTRB','teamRslt', 'teamOrtg', 'teamDrtg',  'opptAbbr', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFG%', 'oppt3P%', 'opptFT%', 'opptTRB', 'opptOrtg', 'opptDrtg']

# %%
dfnew = df[cols]
#%%
df1 = dfnew[dfnew.index % 2 == 0]  # Excludes every other row 
#%%
df1 = df1.rename(columns={"teamFG%": "teamFG", "team3P%": "team3P", "teamFT%" : "teamFT","opptFG%": "opptFG", "oppt3P%": "oppt3P", "opptFT%" : "opptFT" })


#%%
X_home = df1[['teamAST', 'teamTO', 'opptAST', 'opptTO', 'teamDrtg', 'teamOrtg', 'opptDrtg', 'opptOrtg', 'teamFG', 'opptFG', 'teamTRB', 'opptTRB']]
y_home = df1['teamPTS']
# %%
from sklearn.model_selection import train_test_split
X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X_home, y_home, test_size=0.2, random_state=0)

#%%
from sklearn.linear_model import LinearRegression
regressor_home = LinearRegression()
regressor_home.fit(X_train_home, y_train_home)

#%%
y_pred_home = regressor_home.predict(X_test_home)

#%%
print(y_pred_home)

#%%
resultPTS = pd.DataFrame({'Actual Team Score': y_test_home, 'Predicted Team Score': y_pred_home})
resultPTS

#%%
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_home, y_pred_home))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_home, y_pred_home))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_home, y_pred_home)))


#%%
from sklearn.metrics import r2_score
r2_score(y_test_home, y_pred_home)

#%%
coefficients_team = regressor_home.coef_

intercept_team = regressor_home.intercept_
# %%
def calculate_TeamPTS(ASTteam, TOteam, ASToppt, TOoppt, Drtgteam, Ortgteam, Drtgoppt, Ortgoppt, FGteam, FGoppt, TRBteam, TRBoppt ):
  return (ASTteam * coefficients_team[0]) + ( TOteam * coefficients_team[1]) + ( ASToppt* coefficients_team[2]) + ( TOoppt* coefficients_team[3])+ + ( Drtgteam* coefficients_team[4])+ ( Ortgteam* coefficients_team[5])+ ( Drtgoppt* coefficients_team[6])+  ( Ortgoppt* coefficients_team[7])+ ( FGteam * coefficients_team[8]) + ( FGoppt * coefficients_team[9]) + ( TRBteam * coefficients_team[10]) + ( TRBoppt * coefficients_team[11]) + intercept_team




