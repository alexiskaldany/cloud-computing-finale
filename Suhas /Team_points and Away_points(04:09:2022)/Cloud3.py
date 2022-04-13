#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('/Users/alexiskaldany/school/cloud-computing-finale/Suhas /Dataset.csv' )

#%%
cols = ['gmDate', 'gmTime', 'seasTyp', 'teamAbbr','teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFG%', 'team3P%', 'teamFT%', 'teamTRB','teamRslt', 'teamOrtg', 'teamDrtg',  'opptAbbr', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFG%', 'oppt3P%', 'opptFT%', 'opptTRB', 'opptOrtg', 'opptDrtg']

# %%
dfnew = df[cols]
#%%
df1 = dfnew[dfnew.index % 2 == 0]  # Excludes every other row 
#%%
df1 = df1.rename(columns={"teamFG%": "teamFG", "team3P%": "team3P", "teamFT%" : "teamFT","opptFG%": "opptFG", "oppt3P%": "oppt3P", "opptFT%" : "opptFT" })
#%%
#%%
X_away = df1[['teamAST', 'teamTO', 'opptAST', 'opptTO', 'teamDrtg', 'teamOrtg', 'opptDrtg', 'opptOrtg', 'teamFG', 'opptFG', 'teamTRB', 'opptTRB']]
y_away = df1['opptPTS']

# %%
from sklearn.model_selection import train_test_split
X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X_away, y_away, test_size=0.2, random_state=0)

#%%
from sklearn.linear_model import LinearRegression
regressor_away = LinearRegression()
regressor_away.fit(X_train_away, y_train_away)

#%%
y_pred_away = regressor_away.predict(X_test_away)

#%%
print(y_pred_away)

#%%
resultPTS = pd.DataFrame({'Actual': y_test_away, 'Predicted': y_pred_away})
resultPTS

#%%
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_away, y_pred_away))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_away, y_pred_away))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_away, y_pred_away)))


#%%
from sklearn.metrics import r2_score
r2_score(y_test_away, y_pred_away)

#%%
coefficients_away = regressor_away.coef_

intercept_away = regressor_away.intercept_
# %%
def calculate_OpptPTS(ASTteam, TOteam, ASToppt, TOoppt, Drtgteam, Ortgteam, Drtgoppt, Ortgoppt, FGteam, FGoppt, TRBteam, TRBoppt ):
  return (ASTteam * coefficients_away[0]) + ( TOteam * coefficients_away[1]) + ( ASToppt* coefficients_away[2]) + ( TOoppt* coefficients_away[3])+ + ( Drtgteam* coefficients_away[4])+ ( Ortgteam* coefficients_away[5])+ ( Drtgoppt* coefficients_away[6])+  ( Ortgoppt* coefficients_away[7])+ ( FGteam * coefficients_away[8]) + ( FGoppt * coefficients_away[9]) + ( TRBteam * coefficients_away[10]) + ( TRBoppt * coefficients_away[11]) + intercept_away



#%%