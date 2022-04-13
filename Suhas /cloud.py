#%%
import pandas as pd 
import numpy as np 

#%%
df = pd.read_csv('/Users/suhasburavalla/Desktop/Dataset.csv')
# %%
print(df.head())


# %%
cols = ['gmDate', 'gmTime', 'seasTyp', 'teamAbbr','teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFG%', 'team3P%', 'teamFT%', 'teamTRB','teamRslt', 'teamOrtg', 'teamDrtg',  'opptAbbr', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFG%', 'oppt3P%', 'opptFT%', 'opptTRB', 'opptOrtg', 'opptDrtg']

# %%
dfnew = df[cols]
#%%
df1 = dfnew[dfnew.index % 2 == 0]  # Excludes every other row 
#%%
df1 = df1.rename(columns={"teamFG%": "teamFG", "team3P%": "team3P", "teamFT%" : "teamFT","opptFG%": "opptFG", "oppt3P%": "oppt3P", "opptFT%" : "opptFT" })




#%%
# from statsmodels.formula.api import ols

# modelteamscore = ols(formula='teamPTS ~   teamAbbr + opptAbbr + teamAST + teamTO + teamFG + team3P + teamFT + teamTRB + teamOrtg + teamDrtg + opptOrtg + opptDrtg', data=df1).fit()

# print( type(modelteamscore) )
# print( modelteamscore.summary() )

# modelpredictions = pd.DataFrame()

# modelpredictions['gre_GpaRankLM'] = modelteamscore.predict(df1)
# print(modelpredictions.head())

# print("\nReady to continue.")


# %%
from statsmodels.formula.api import ols

modelteamscore = ols(formula='teamPTS ~  teamAbbr + opptAbbr + teamAST + teamTO + opptAST + opptTO + teamOrtg + teamDrtg', data=df1).fit()

print( type(modelteamscore) )
print( modelteamscore.summary() )



#%%
from statsmodels.formula.api import ols

modelopptscore = ols(formula='opptPTS ~   teamAbbr +  opptAbbr + opptAST + opptTO + teamAST + teamTO + opptOrtg + opptDrtg ', data=df1).fit()

print( type(modelopptscore) )
print( modelopptscore.summary() )






# %%
#Creating test dataframe with matchups from the NBA schedule on 7th April 2022

data = {
    
    'teamAbbr' : ['CHA', 'TOR', 'MIL', 'MIN', 'NO', 'DEN', 'GS'],
    'opptAbbr' : ['ORL', 'PHI', 'BOS', 'SA', 'POR', 'MEM', 'LAL'],
    'teamOrtg' : [113.1, 112.1, 114.1, 113.6, 110.9, 113.6, 111.8], 
    'teamDrtg' : [113.3, 109.8, 111.0, 110.9, 111.5, 11.3, 106.7],
    'opptOrtg' : [103.8, 112.6, 113.2, 112.0, 108.0, 114.2, 109.6],
    'opptDrtg' : [111.9, 109.0, 106.1, 111.4, 116.0, 108.5, 111.2],
    'teamAST' : [27.9, 22.0, 23.8, 25.6, 24.9, 27.7, 26.9],
    'teamTO' : [12.7, 11.7, 12.8, 13.8, 13.3, 13.9, 14.3],
    'opptAST' : [23.6, 23.5, 24.6, 28.0, 22.9, 25.7, 24.1],
    'opptTO' : [13.9, 11.7, 13.0, 12.4, 13.5, 12.4, 13.9],
    'teamFG' : [0.468, 0.445, 0.468, 0.457, 0.457, 0.483, 0.469],
    'opptFG' : [0.434, 0.466, 0.466, 0.467, 0.442, 0.461, 0.469], 
    'teamTRB' : [45, 45, 47, 44, 45, 44, 46],
    'opptTRB' : [44, 42, 46, 45, 43, 49, 44 ]
    
    
}

test_df = pd.DataFrame(columns=['teamAbbr', 'opptAbbr', 'teamOrtg', 'teamDrtg', 'opptOrtg', 'opptDrtg', 'teamAST', 'teamTO', 'opptAST', 'opptTO', 'teamPTS', 'opptPTS' ], data=data )



# %%
#Testing the model on test data
#Team points
test_df['teamPTS'] = modelteamscore.predict(exog =test_df)
print(test_df.teamPTS)

print("\nReady to continue.")

#%%
#Opponent Points
test_df['opptPTS'] = modelopptscore.predict(exog =test_df)
print(test_df.opptPTS)

print("\nReady to continue.")
# %%
