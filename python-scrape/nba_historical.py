# -*- coding: utf-8 -*-
"""NBA_Historical

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1os9GRatmuUvbZJqBuX4cAjMPcJc0Ci8X

# NBA DATA SCRAPE - HISTORICAL

# PACKAGE IMPORT / INSTALL
"""

## TAKE CAUTION WITH INSTALL - MAY AFFECT SUBSEQUENT IMPORTS OR REQUIRE RUNTIME RESTART

#!pip install basketball-reference-scraper==v1.0.3
# LIBRARY IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics

import requests
from bs4 import BeautifulSoup
import json
import time
import nltk
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#from tabulate import tabulate
#from difflib import get_close_matches
#from itertools import islice

print("\nIMPORT SUCCESS")

drive.mount('drive')

"""# BASKETBALL REFERENCE

https://github.com/vishaalagartha/basketball_reference_scraper/blob/master/examples.py

## IMPORT / INSTALL
"""

#!pip install basketball-reference-scraper==v1.0.3
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
from basketball_reference_scraper.players import get_stats, get_game_logs
from basketball_reference_scraper.seasons import get_schedule, get_standings
from basketball_reference_scraper.box_scores import get_box_scores
from basketball_reference_scraper.pbp import get_pbp
from basketball_reference_scraper.shot_charts import get_shot_chart

"""## STAT VARIABLES"""

# Start Date fixed to reflect modern-day NBA Expansion to 30 teams in 2004

current_date = '2022-03-30'
current_year = 2022

start_2004 = '2003-11-01'
start_2000 = '1999-11-01'
start_2010 = '2009-11-01'


team_code_dict = {'Charlotte Hornets':'CHO', 'Dallas Mavericks':'DAL', 'Denver Nuggets':'DEN',
                  'Houston Rockets':'HOU', 'Los Angeles Clippers':'LAC', 'Miami Heat':'MIA',
                  'New Jersey Nets':'BRK', 'New York Knicks':'NYK', 'San Antonio Spurs':'SAS',
                  'Toronto Raptors':'TOR', 'Utah Jazz':'UTA', 'Vancouver Grizzlies':'MEM',
                  'Washington Wizards':'WAS', 'Boston Celtics':'BOS', 'Chicago Bulls':'CHI',
                  'Cleveland Cavaliers':'CLE', 'Los Angeles Lakers':'LAL', 'Orlando Magic':'ORL',
                  'Portland Trail Blazers':'POR', 'Atlanta Hawks':'ATL', 'Phoenix Suns':'PHO',
                  'Seattle SuperSonics':'OKC', 'Detroit Pistons':'DET', 'Sacramento Kings':'SAC',
                  'Golden State Warriors':'GSW', 'Indiana Pacers':'IND', 'Milwaukee Bucks':'MIL',
                  'Minnesota Timberwolves':'MIN', 'Philadelphia 76ers':'PHI', 'Memphis Grizzlies':'MEM',
                  'New Orleans Hornets':'NOP', 'Charlotte Bobcats':'CHO', 'New Orleans/Oklahoma City Hornets':'NOP',
                  'Oklahoma City Thunder':'OKC', 'Brooklyn Nets':'BRK', 'New Orleans Pelicans':'NOP'
                  }

                  # PENDING CODE CONVERSION / MAPPING:
                    # PHO = PHX
                    # CHO = CHA
                    # BRK = BKN

team_codes = team_code_dict.values()
print(team_codes)

"""## TEAM ROSTERS"""

# GENERATE TEAM ROSTERS DATAFRAME
# ** RUNTIME: ~20 SECONDS **

team_rosters_2022 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster(team, 2022)
  temp_df['TEAM'] = team
  team_rosters_2022 = team_rosters_2022.append(temp_df)

print(team_rosters_2022)
#print(team_rosters.info())

# SCRAPE TEAM ROSTER STATS - 2022
# ** RUNTIME: ~2 MIN **

team_roster_stats_2022 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2022, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2022 = team_roster_stats_2022.append(temp_df)

print(team_roster_stats_2022)

# SCRAPE TEAM ROSTER STATS - 2021
# ** RUNTIME: ~2.5 MIN **

team_roster_stats_2021 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2021, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2021 = team_roster_stats_2021.append(temp_df)

print(team_roster_stats_2021)

# SCRAPE TEAM ROSTER STATS - 2020
# ** RUNTIME: ~3 MIN **

team_roster_stats_2020 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2020, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2020 = team_roster_stats_2020.append(temp_df)

print(team_roster_stats_2020)

# SCRAPE TEAM ROSTER STATS - 2019
# ** RUNTIME: ~3 MIN **

team_roster_stats_2019 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2019, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2019 = team_roster_stats_2019.append(temp_df)

print(team_roster_stats_2019)

# SCRAPE TEAM ROSTER STATS - 2018
# ** RUNTIME: ~2 MIN **

team_roster_stats_2018 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2018, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2018 = team_roster_stats_2018.append(temp_df)

print(team_roster_stats_2018)

# SCRAPE TEAM ROSTER STATS - 2017
# ** RUNTIME: ~1.5 MIN **

team_roster_stats_2017 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2017, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2017 = team_roster_stats_2017.append(temp_df)

print(team_roster_stats_2017)

# SCRAPE TEAM ROSTER STATS - 2016
# ** RUNTIME: ~1.5 MIN **

team_roster_stats_2016 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2016, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2016 = team_roster_stats_2016.append(temp_df)

print(team_roster_stats_2016)

# SCRAPE TEAM ROSTER STATS - 2015
# ** RUNTIME: ~2.5 MIN **

team_roster_stats_2015 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2015, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2015 = team_roster_stats_2015.append(temp_df)

print(team_roster_stats_2015)

# SCRAPE TEAM ROSTER STATS - 2014
# ** RUNTIME: ~2 MIN **

team_roster_stats_2014 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2014, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2014 = team_roster_stats_2014.append(temp_df)

print(team_roster_stats_2014)

# SCRAPE TEAM ROSTER STATS - 2013
# ** RUNTIME: ~2 MIN **

team_roster_stats_2013 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2013, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2013 = team_roster_stats_2013.append(temp_df)

print(team_roster_stats_2013)

# SCRAPE TEAM ROSTER STATS - 2012
# ** RUNTIME: ~2 MIN **

team_roster_stats_2012 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2012, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2012 = team_roster_stats_2012.append(temp_df)

print(team_roster_stats_2012)

# SCRAPE TEAM ROSTER STATS - 2011
# ** RUNTIME: ~2 MIN **

team_roster_stats_2011 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2011, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2011 = team_roster_stats_2011.append(temp_df)

print(team_roster_stats_2011)

# SCRAPE TEAM ROSTER STATS - 2010
# ** RUNTIME: ~2 MIN **

team_roster_stats_2010 = pd.DataFrame()

for team in team_codes:
  temp_df = get_roster_stats(team, 2010, data_format='PER_GAME', playoffs=False)
  temp_df['TEAM'] = team
  team_roster_stats_2010 = team_roster_stats_2010.append(temp_df)

print(team_roster_stats_2010)



team_roster_stats_2015_2022 = pd.concat([team_roster_stats_2020, team_roster_stats_2019, team_roster_stats_2018, team_roster_stats_2017, team_roster_stats_2016, team_roster_stats_2015])
#team_roster_stats_2015_2022 = pd.merge(team_roster_stats_2022, team_roster_stats_2021) #, left_index=True # axis=0 , on = 'PLAYER'

#team_roster_stats_2020, team_roster_stats_2019, team_roster_stats_2018, team_roster_stats_2017, team_roster_stats_2016, team_roster_stats_2015,
team_roster_stats_2015_2022

player_info = pd.merge(team_rosters, team_roster_stats) #, left_index=True # axis=0 , on = 'PLAYER'
#player_info.info()
player_info[:25]

#team_roster_stats.info()
#player_info.describe()
#team_roster_stats.info()
team_rosters.columns



"""### DATA EXPORT"""

matchup_filepath = 'drive/My Drive/GWU/TEAM-7/data/team_roster_stats_2015_2022'

team_roster_stats_2015_2022.to_excel(matchup_filepath + '.xlsx', index=False)
team_roster_stats_2015_2022.to_csv(matchup_filepath + '.csv', index=False)

"""### DATA IMPORT"""

historical_roster_stats = pd.read_excel(matchup_filepath + '.xlsx', index_col='PLAYER')
#matchup_history = pd.read_csv(matchup_filepath + '.csv', index_col='DATE')
historical_roster_stats

historical_roster_stats.dropna(inplace=True)
historical_roster_stats.info()

def wtd_avg(df, values, weights):
  return sum(df[weights] * df[values]) / df[weights].sum()

hist_eFG_pct = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, 'eFG%', 'MP')
hist_3P_pct = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, '3P%', 'MP')
hist_2P_pct = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, '2P%', 'MP')
hist_FT_pct = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, 'FT%', 'MP')


three_point_attempts = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, '3PA', 'MP')
two_point_attempts = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, '2PA', 'MP')
#team_wtd_avg = pd.DataFrame(data=[three_point_attempts, two_point_attempts], columns=)
team_wtd_avg = pd.DataFrame(data=[hist_eFG_pct, hist_3P_pct, hist_2P_pct, hist_FT_pct], index=['eFG%', '3P%', '2P%', 'FT%']).T
team_wtd_avg

roster_num_cols = ['AGE','G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
                   '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
                   'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] #, 'HEIGHT', 'WEIGHT', 'EXPERIENCE'

def wtd_avg(df, values, weights):
    return sum(df[weights] * df[values]) / df[weights].sum()
#print(wtd_avg(historical_roster_stats, 'Grades', 'MP'))

for col in historical_roster_stats.columns:
  test2 = historical_roster_stats.groupby(by = ['TEAM', 'SEASON']).apply(wtd_avg, col, 'MP')
  team_wtd_avg[col] = .apply(wtd_avg, col, 'MP')

print(test2)

"""### DATA VIZ"""

roster_num_cols = ['AGE','G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
                   '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
                   'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] #, 'HEIGHT', 'WEIGHT', 'EXPERIENCE'

# EJECT OR IMPUTE MISSING VALUES:
  # FG%
  # 3P%
  # 2P%
  # eFG%
  # FT%

# CAST DATA AS NUMERIC FLOATS
player_info[roster_num_cols] = player_info[roster_num_cols].apply(lambda x: x.astype(float))
print(player_info.info())

player_info.dropna(inplace=True)
player_info.info()

college_mean = player_info.groupby(by=['COLLEGE']).mean()
college_sum = player_info.groupby(by=['COLLEGE']).sum()

college_mean.describe()

top_25_colleges = player_info['COLLEGE'].value_counts().sort_values(ascending=False)[:25]
top_10_colleges = player_info['COLLEGE'].sort_values(ascending=False).value_counts()[:10]
top_10_colleges

plt.figure(figsize=(12,10)) 
sns.scatterplot(data=college_mean, x='MP', y='PTS', hue='COLLEGE', palette='mako') #, rug=True

#plt.title('', fontsize=16)
#plt.xlabel('', fontsize=16)
#plt.ylabel('', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show();

"""## HISTORICAL MATCHUPS"""

#import pandas as pd

# ** RUNTIME: 5+ MIN - pre-saved below **

matchups_2000 = get_schedule(2000, playoffs=False)
matchups_2001 = get_schedule(2001, playoffs=False)
matchups_2002 = get_schedule(2002, playoffs=False)
matchups_2003 = get_schedule(2003, playoffs=False)
matchups_2004 = get_schedule(2004, playoffs=False)
matchups_2005 = get_schedule(2005, playoffs=False)
matchups_2006 = get_schedule(2006, playoffs=False)
matchups_2007 = get_schedule(2007, playoffs=False)
matchups_2008 = get_schedule(2008, playoffs=False)
matchups_2009 = get_schedule(2009, playoffs=False)
matchups_2010 = get_schedule(2010, playoffs=False)
matchups_2011 = get_schedule(2011, playoffs=False)
matchups_2012 = get_schedule(2012, playoffs=False)
matchups_2013 = get_schedule(2013, playoffs=False)
matchups_2014 = get_schedule(2014, playoffs=False)
matchups_2015 = get_schedule(2015, playoffs=False)
matchups_2016 = get_schedule(2016, playoffs=False)
matchups_2017 = get_schedule(2017, playoffs=False)
matchups_2018 = get_schedule(2018, playoffs=False)
matchups_2019 = get_schedule(2019, playoffs=False)
matchups_2020 = get_schedule(2020, playoffs=False)
matchups_2021 = get_schedule(2021, playoffs=False)
matchups_2022 = get_schedule(2022, playoffs=False)

print(f'DOWNLOAD SUCCESSFUL.')

## ODDLY 2020 data is missing ..... COVID

matchups_2000_2022 = pd.concat([matchups_2000, matchups_2001, matchups_2002, matchups_2003, matchups_2004, matchups_2005,
                             matchups_2006, matchups_2007, matchups_2008, matchups_2009, matchups_2010, matchups_2011,
                             matchups_2012, matchups_2013, matchups_2014, matchups_2015, matchups_2016, matchups_2017,
                             matchups_2018, matchups_2019, matchups_2021, matchups_2022], axis=0, join='outer') # how='outer', , on='DATE'
matchups_2000_2022.info()
#matchups_2000_2019.head()

"""### DATA EXPORT"""

matchup_filepath = 'drive/My Drive/GWU/TEAM-7/data/historical_matchups'

matchups_2000_2022.to_excel(matchup_filepath + '.xlsx', index=False)
matchups_2000_2022.to_csv(matchup_filepath + '.csv', index=False)

"""### DATA IMPORT"""

matchup_history = pd.read_excel(matchup_filepath + '.xlsx', index_col='DATE')
#matchup_history = pd.read_csv(matchup_filepath + '.csv', index_col='DATE')

"""### PRE-PROCESSING"""

print(matchup_history.columns)
print('-'*100)
print(matchup_history.info())
print('-'*100)
print(matchup_history.head())

matchup_history['HOME'].unique()

matchup_history['VISITOR_CODE'] = matchup_history['VISITOR'].map(team_code_dict)
matchup_history['HOME_CODE'] = matchup_history['HOME'].map(team_code_dict)
matchup_history['VISITOR_MARGIN'] = matchup_history['VISITOR_PTS'] - matchup_history['HOME_PTS']
matchup_history['HOME_MARGIN'] = matchup_history['HOME_PTS'] - matchup_history['VISITOR_PTS']

matchup_history.info()

matchup_history



matchup_history.dropna(inplace=True)
matchup_history.info()







"""### DATA VIZ"""

plt.figure(figsize=(12,10))
sns.histplot(matchup_history['HOME_CODE'], palette='mako') #, rug=True

#plt.title('', fontsize=16)
#plt.xlabel('', fontsize=16)
#plt.ylabel('', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show();

"""## LEAGUE STANDINGS"""

standings = get_standings(date='2022-03-06')
print(standings)

"""## BOX SCORES"""

s = get_box_scores('2020-01-13', 'CHI', 'BOS', period='GAME', stat_type='BASIC')
print(s)

s = get_box_scores('2020-01-13', 'CHI', 'BOS', period='GAME', stat_type='BASIC')
print(s)

"""## PLAY-BY-PLAY"""

s = get_pbp('2020-01-13', 'CHI', 'BOS')
print(s)

"""## SHOT CHARTS"""

s = get_shot_chart('2020-01-13', 'CHI', 'BOS')
print(s)

from basketball_reference_scraper.injury_report import get_injury_report

s = get_injury_report()
print(s)

"""## PLAYER GAME LOGS"""

#s = get_roster_stats('GSW', 2022, data_format='PER_GAME', playoffs=False)
#print(s)

#s = get_opp_stats('GSW', 2021, data_format='PER_GAME')
#print(s)

#s = get_team_misc('GSW', 2021)
#print(s)

from basketball_reference_scraper.players import get_game_logs #get_player_headshot, get_stats

df = get_game_logs('LeBron James', '2010-01-19', '2014-01-20', playoffs=False)
print(df)

#s = get_stats('Stephen Curry', stat_type='PER_GAME', playoffs=False, career=False)
#print(s)

#df = get_game_logs('Pau Gasol', '2010-01-12', '2010-01-20', playoffs=False)
#print(df)

# url = get_player_headshot('Kobe Bryant')
# print(url)

#df = get_game_logs('HOU', '2010-01-12', '2010-01-15', playoffs=False)
#print(df)

#from basketball_reference_scraper.drafts import get_draft_class

#df = get_draft_class(2003)
#print(df)



"""# MODELING PIPELINE

## IMPORTS
"""

# Commented out IPython magic to ensure Python compatibility.
## LIBRARY IMPORTS

import sys
#from google.colab import drive

import numpy as np
import pandas as pd

# %tensorflow_version 2.x 
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')


random_seed = 42
tf.random.set_seed(random_seed)
#import numpy as np
np.random.seed(random_seed)

# Absolute path of current folder
abspath_curr = '/content/drive/My Drive/SPORTS/NBA/'

# Absolute path of shallow utilities folder
abspath_util_shallow = '/content/drive/My Drive/Colab Notebooks/teaching/gwu/machine_learning_I/spring_2022/code/utilities/p2_shallow_learning/'

# IMPORT 

#reg_szn_detail = pd.read_csv('drive/My Drive/SPORTS/NCAAB/data/MRegularSeasonDetailedResults.csv')

#print(tourney_detail.head())
#print('-'*100)
#print(tourney_detail.info())
#print('-'*100)
#print(reg_szn_detail.head())
#print('-'*100)
#print(reg_szn_detail.info())

"""## FEATURE ENGINEERING"""

#reg_szn_detail['WRatio'] = (reg_szn_detail.WCount / (reg_szn_detail.WCount + reg_szn_detail.LCount))

#reg_szn_detail['WtdAvgMargin'] = ((reg_szn_detail['WCount'] * reg_szn_detail['AMarginAvg'] -
#                                reg_szn_detail['LCount'] * reg_szn_detail['BMarginAvg']) /
#                               (reg_szn_detail['WCount'] + reg_szn_detail['LCount'])
                               )
#reg_szn_detail.info()

"""## TARGET"""

# GENERATE COLUMNS FOR TARGET VARIABLE PREDICTION

#reg_szn_detail['ScoreMargin'] = reg_szn_detail['AScore'] - reg_szn_detail['BScore']
#reg_szn_detail['Win'] = (reg_szn_detail['ScoreMargin'] > 0).astype(int)

# DROP COLUMNS
#reg_szn_detail.drop(['NumOT', 'ALoc', 'WCount', 'LCount', 'AMargin', 'BMargin', 'ATeamID', 'BTeamID', 'TeamID', 'DayNum', 'Season'], axis=1, inplace=True)
#reg_szn_detail.info()



"""## TRAIN-TEST"""

# DEFINE TEST FEATURES
test_features = ['ScoreMargin',
                 'Win',
                 'AFGM',
     'BFGM',
     'AFGA',
     'BFGA', 
     'AFGM3',
     'BFGM3',
     'AFGA3',
     'BFGA3',
     'AFTM',
     'BFTM', 
     'AFTA',
     'BFTA',
    'AOR',
    'BOR',
    'ADR',
    'BDR',
     'AAst',
     'BAst',
     'ATO',
     'BTO',
     'AStl',
     'BStl', 
     'ABlk',
     'BBlk',
     'APF',
     'BPF',
     'WtdAvgMargin',
     'WRatio',
]

reg_szn_test = reg_szn_detail[test_features]
reg_szn_test.info()

reg_szn_test.head()

from sklearn.model_selection import train_test_split

train, test = train_test_split(reg_szn_test, train_size=0.8, random_state=42)

# Load the raw training data
df_raw_train = train

# Make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)

# Load the raw test data
df_raw_test = test

# Make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)

## TARGET VARIABLE ASSIGNMENT

target = 'Win'

# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

# Print the dimension of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

# Print the first 5 rows of df_train
df_train.head()

# Print the first 5 rows of df_test
df_test.head()

"""## SPLIT"""

from sklearn.model_selection import train_test_split

# Divide the training data into training (80%) and validation (20%)
df_train, df_val = train_test_split(df_train, train_size=0.8, random_state=random_seed)

# Reset the index
df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)

# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

# Print the dimension of df_val
pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns'])

# Combine df_train, df_val and df_test
df = pd.concat([df_train, df_val, df_test], sort=False)
df.head()



"""## LABEL ENCODER"""

from sklearn.preprocessing import LabelEncoder

# The LabelEncoder
le = LabelEncoder()

# Encode categorical target in the combined data
df[target] = le.fit_transform(df[target])

# Print the first 5 rows of df
df.head()

"""## PRE-PROCESSING"""

#df.drop(['id'], inplace=True,  axis=1)
#df.head()

# Separating the training data
df_train = df.iloc[:df_train.shape[0], :]

# Separating the validation data
df_val = df.iloc[df_train.shape[0]:df_train.shape[0] + df_val.shape[0], :]

# Separating the test data
df_test = df.iloc[df_train.shape[0] + df_val.shape[0]:, :]

# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

# Print the dimension of df_val
pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns'])

# Print the dimension of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

# Combine df_train, df_val and df_test
df = pd.concat([df_train, df_val, df_test], sort=False)

"""## FEATURE MATRIX"""

# Get the feature matrix
X_train = df_train[np.setdiff1d(df_train.columns, [target])].values
X_val = df_val[np.setdiff1d(df_val.columns, [target])].values
X_test = df_test[np.setdiff1d(df_test.columns, [target])].values

# Get the target vector
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

"""## SCALE"""

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler
mms = MinMaxScaler()

# Normalize the training data
X_train = mms.fit_transform(X_train)

# Normalize the validation data
X_val = mms.transform(X_val)

# Normalize the test data
X_test = mms.transform(X_test)

"""# HYPERPARAMETER TUNING"""

# Model / Package Imports
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.experimental import enable_hist_gradient_boosting
#from sklearn.ensemble import HistGradientBoostingClassifier

# Creating dictionary of the models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'mlpc': MLPClassifier(early_stopping=True, random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          #'hgbc': HistGradientBoostingClassifier(random_state=random_seed)
          }

# Creating the dictionary of the pipelines
from sklearn.pipeline import Pipeline

pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])

# Commented out IPython magic to ensure Python compatibility.
## SUPPLEMENTAL - NOT NECESSARY

# Change working directory to the absolute path of the shallow utilities folder
# %cd $abspath_util_shallow

# Import the shallow utitilities
# %run pmlm_utilities_shallow.ipynb

#Getting the predefined split cross-validator
  # feature matrix and target velctor in the combined training and validation data
  # target vector in the combined training and validation data
  # PredefinedSplit
  # See the implementation in pmlm_utilities.ipynb

X_train_val, y_train_val, ps = get_train_val_ps(X_train, y_train, X_val, y_val)

"""## GridSearch CV Param Grids"""

param_grids = {}

"""## Logistic Regression"""

# The parameter grid of tol
tol_grid = [10 ** -5, 10 ** -3] #10 ** -11, 
  # ORIGINAL: tol_grid = [10 ** -5, 10 ** -4, 10 ** -3]

# The parameter grid of C
C_grid = [10, 1, .1, 0.01] # 1 , .001
  # ORIGINAL: C_grid = [0.1, 1, 10]

# The parameter grid of penalty type
penalty_grid = ['none', 'l1', 'l2'] #, 'elasticnet'

# The parameter grid of solver type
solver_grid = ['newton-cg', 'lbfgs', 'liblinear'] # 'sag', 'saga'

# Update param_grids
param_grids['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid,
                      #'model__penalty': penalty_grid,
                      'model__solver': solver_grid,
                      }]

"""## MLP Classifier"""

# The grids for alpha
alpha_grids = [10 ** i for i in range(-5, -2)]
  # ORIGINAL: alpha_grids = [10 ** i for i in range(-5, -2)]

# The grids for learning_rate_init
learning_rate_init_grids = [9 ** i for i in range(-3, -1)]
  # ORIGINAL: learning_rate_init_grids = [10 ** i for i in range(-4, -1)]

# Update param_grids
param_grids['mlpc'] = [{'model__alpha': alpha_grids,
                        'model__learning_rate_init': learning_rate_init_grids}]

"""## Random Forest Classifier"""

# The grids for min_samples_split
min_samples_split_grids = [20, 100]
  # ORIGINAL: min_samples_split_grids = [2, 20, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1, 100]
  # ORIGINAL: min_samples_leaf_grids = [1, 20, 100]

# The grids for n_estimators
n_estimators_grids = [100] # 10, 50, 100

# The grids for max_depth
max_depth_grids = [1, 5, 10]

# The grids for max_features
max_features_grids = ['auto', 'sqrt']

# Update param_grids
param_grids['rfc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids,
                       'model__n_estimators': n_estimators_grids,
                       #'model__max_depth': max_depth_grids,
                       #'model__max_features': max_features_grids,
                       }]

                       # model__max_depth:

"""## GRIDSEARCH"""



import os
# Make directory
directory = os.path.dirname(abspath_curr + '/result/mm2022/GridSearchCV_results/')
if not os.path.exists(directory):
    os.makedirs(directory)

# HYPERPARAMETER TUNING

from sklearn.model_selection import GridSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []

# For each model
for acronym in pipes.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=2, #8
                      cv=ps, #5
                      return_train_score=True)
        
    # Fit the pipeline
    gs = gs.fit(X_train_val, y_train_val)
    
    # Update best_score_params_estimator_gs
    best_score_params_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(path_or_buf=abspath_curr + '/result/mm2022/GridSearchCV_results/' + acronym + '.csv', index=False) #'/result/mnist/cv_results/GridSearchCV/'

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])

"""## MODEL SELECTION"""

# Get the best_score, best_params and best_estimator obtained by GridSearchCV
best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]

print(best_score_gs)
print(best_params_gs)
print(best_estimator_gs)

"""# GENERATE SUBMISSION FILE
###### Use best model as selected above to generate submission file for Kaggle competition:

## Create Directory
"""

# Make directory
directory = os.path.dirname(abspath_curr + '/result/submission/')
if not os.path.exists(directory):
    os.makedirs(directory)

"""## Generate Submission"""

# Get the prediction on the testing data using best_model
y_test_pred = best_estimator_gs.predict(X_test)

# Transform y_test_pred back to the original class
y_test_pred = le.inverse_transform(y_test_pred)

# Get the submission dataframe
df_submit = pd.DataFrame(np.hstack((np.arange(1, y_test_pred.shape[0] + 1).reshape(-1, 1), y_test_pred.reshape(-1, 1))),
                         columns=['id', target]).astype({'id':int, target:int})                                                                                      

# Generate the submission file
df_submit.to_csv(abspath_curr + '/result/submission/submission.csv', index=False)

"""# INTERPRETATION

## Create Directory
"""

import os

# Make directory
directory = os.path.dirname(abspath_curr + '/result/figure/')
if not os.path.exists(directory):
    os.makedirs(directory)

"""## Feature Importance - Table"""

# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_rfc, best_param_rfc, best_estimator_rfc = best_score_params_estimator_gs[1]

# Get the dataframe of feature and importance
df_fi_rfc = pd.DataFrame(np.hstack((np.setdiff1d(df.columns, [target]).reshape(-1, 1), best_estimator_rfc.named_steps['model'].feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])

# Sort df_fi_rfc in descending order of the importance
df_fi_rfc = df_fi_rfc.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print the first 5 rows of df_fi_rfc
df_fi_rfc[:]



"""## Feature Importance - Plot"""

# Create a figure
fig = plt.figure(figsize=(10, 5))

# The bar plot of the top 5 feature importance
plt.bar(df_fi_rfc['Features'][:5], df_fi_rfc['Importance'][:5], color='green')

# Set x-axis
plt.xlabel('Features')
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance')

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath_curr + '/result/poker/figure/feature_importance_rfc.pdf')
plt.show()

"""# SCRATCH NOTES"""

# SCORING MARGIN / POSSESSIONS
tr_data_hub['net-avg-scoring-margin'] = tr_data_hub['average-scoring-margin'] - tr_data_hub['opponent-average-scoring-margin']
tr_data_hub['net-points-per-game'] = tr_data_hub['points-per-game'] - tr_data_hub['opponent-points-per-game']
tr_data_hub['net-effective-possession-ratio'] = tr_data_hub['effective-possession-ratio'] - tr_data_hub['opponent-effective-possession-ratio']
tr_data_hub['net-adj-efficiency'] = tr_data_hub['offensive-efficiency'] - tr_data_hub['defensive-efficiency']

# NET SHOOTING PERCENTAGES
tr_data_hub['net-effective-field-goal-pct'] = tr_data_hub['effective-field-goal-pct'] - tr_data_hub['opponent-effective-field-goal-pct']
tr_data_hub['net-true-shooting-percentage'] = tr_data_hub['true-shooting-percentage'] - tr_data_hub['opponent-true-shooting-percentage']

# STOCKS = STEALS + BLOCKS
tr_data_hub['stocks-per-game'] = tr_data_hub['steals-per-game'] + tr_data_hub['blocks-per-game']
tr_data_hub['opponent-stocks-per-game'] = tr_data_hub['opponent-steals-per-game'] + tr_data_hub['opponent-blocks-per-game']
tr_data_hub['net-stocks-per-game'] = tr_data_hub['stocks-per-game'] - tr_data_hub['opponent-stocks-per-game']

# AST/TO = TURNOVERS / ASSISTS
tr_data_hub['total-turnovers-per-game'] = tr_data_hub['turnovers-per-game'] + tr_data_hub['opponent-turnovers-per-game']
tr_data_hub['net-assist--per--turnover-ratio'] = tr_data_hub['assist--per--turnover-ratio'] - tr_data_hub['opponent-assist--per--turnover-ratio']

# REBOUNDS
tr_data_hub['net-total-rebounds-per-game'] = tr_data_hub['total-rebounds-per-game'] - tr_data_hub['opponent-total-rebounds-per-game']
tr_data_hub['net-off-rebound-pct'] = tr_data_hub['offensive-rebounding-pct'] - tr_data_hub['opponent-offensive-rebounding-pct']
tr_data_hub['net-def-rebound-pct'] = tr_data_hub['defensive-rebounding-pct'] - tr_data_hub['opponent-defensive-rebounding-pct']
  
  # ALTERNATE CALC - yields different performance than above
    #tr_data_hub['net-off-rebound-pct'] = tr_data_hub['offensive-rebounding-pct'] - tr_data_hub['opponent-defensive-rebounding-pct']
    #tr_data_hub['net-def-rebound-pct'] = tr_data_hub['defensive-rebounding-pct'] - tr_data_hub['opponent-offensive-rebounding-pct']

tr_data_hub.info()
#tr_data_hub.columns



"""## PCA"""

#%%

X = df[features].values
X = StandardScaler().fit_transform(X)

#%%
pca = PCA(n_components='mle', svd_solver='full') # 'mle'

pca.fit(X)
X_PCA = pca.transform(X)
print('ORIGINAL DIMENSIONS:', X.shape)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')

#%%
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)

plt.show()

"""## SVD"""

# SINGULAR VALUE DECOMPOSITION ANALYSIS [SVD]
# CONDITION NUMBER

# ORIGINAL DATA

from numpy import linalg as LA

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'ORIGINAL DATA: SINGULAR VALUES {d}')
print(f'ORIGINAL DATA: CONDITIONAL NUMBER {LA.cond(X)}')

# TRANSFORMED DATA
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'TRANSFORMED DATA: SINGULAR VALUES {d_PCA}')
print(f'TRANSFORMED DATA: CONDITIONAL NUMBER {LA.cond(X_PCA)}')
print('*'*58)

#%%
# CONSTRUCTION OF REDUCED DIMENSION DATASET

#pca_df = pca.explained_variance_ratio_

a, b = X_PCA.shape
column = []

for i in range(b):
    column.append(f'PRINCIPAL COLUMN {i+1}')

df_PCA = pd.DataFrame(data=X_PCA, columns=column)
df_PCA = pd.concat([df_PCA, Y], axis=1)

df_PCA.info()