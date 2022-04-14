#%%
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import dash as dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly as ply
import plotly.express as px

import datetime

print("\nIMPORT SUCCESS")

#%%
# LIVE MATCHUPS
matchup_filepath = 'dash_matchups-4-7-22-vF'
matchup_df = pd.read_excel(matchup_filepath + '.xlsx', index_col='TEAM_CODE')
#matchup_df = pd.read_csv(matchup_filepath + '.csv', index_col='TEAM_CODE')

#print(matchup_df.columns)
print(matchup_df)

#%%
print(matchup_df['TEAM'])

preds_dict = {'Magic':'LOSS', 'Hornets':'WIN', 'Celtics':'LOSS', 'Bucks':'WIN', '76ers':'WIN',
              'Raptors':'LOSS', 'Trail Blazers':'LOSS', 'Pelicans':'WIN', 'Spurs':'LOSS',
              'Timberwolves':'WIN', 'Grizzlies':'LOSS', 'Nuggets':'WIN', 'LA Lakers':'LOSS', 'Warriors':'WIN'}

matchup_df['MODEL_PREDICTION'] = matchup_df.TEAM.map(preds_dict)
print(matchup_df)

#%%
# STATS LIBRARY
#tr_filepath = '../cloud-computing-finale/data/tr_data_hub_4-13-22'
#tr_data_hub = pd.read_excel(tr_filepath + '.xlsx', index_col='TEAM_CODE')
#tr_data_hub = pd.read_csv(tr_filepath + '.csv', index_col='TEAM_CODE')
#tr_data_hub.info()


#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
matchup_app = dash.Dash('MATCHUP MACHINE', external_stylesheets=external_stylesheets)
application = matchup_app.server

# {current_time:%Y-%m-%d %H:%M}
#'backgroundColor': 'rgb(220, 220, 220)',

matchup_app.layout = html.Div([html.H1('NBA MATCHUP MACHINE', style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '3px solid black'}),
                     #html.Br(),
                               html.Div([html.H3((f'MATCHUP INFORMATION - 4/7/2022'),
                                                 style={'textAlign': 'center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '3px solid black'}), #'dodgerblue'

                               dash_table.DataTable(matchup_df.to_dict('records'), [{"name": i, "id": i} for i in matchup_df.columns],
                                                    style_data={'border': '2px solid black'},
                                                    style_cell={'padding': '5px'},   #324f6e - TBD  #B10DC9 - fuschia #7FDBFF - Aqua
                                                    style_header={'backgroundColor': '#7FDBFF', 'color': 'black', 'fontWeight': 'bold', 'border': '2px solid black'}, #'1px solid blue'
                                                    style_data_conditional = [{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(211, 211, 211)'},
                                                                              {'if': {'filter_query': '{MODEL_PREDICTION} = {LOSS}','column_id': 'MODEL_PREDICTION',
                                                                               'color': 'red', 'backgroundColor': 'red', 'fontWeight': 'bold'}},
                                                                              {'if': {'column_id': 'MODEL_PREDICTION'}, 'backgroundColor': '#01FF70', 'fontWeight': 'bold',},],

                                                    #  , 'color': 'white', rgb(231, 210, 192) // rgb(210, 192, 232)
                                                    ),
                               #html.H3('MODEL PREDICTS XX TO WIN TONIGHT', style={'textAlign': 'center'})
                               ]),
                               html.Br(),
                     html.P('METRIC COMPARISON'),
                     dcc.Graph(id='matchup-chart'),
                     html.Br(),
                     html.P('STAT A'),
                     dcc.Dropdown(id='stata',
                                  options=[{'label': 'effective-field-goal-pct', 'value': 'effective-field-goal-pct'},
                                           {'label': 'true-shooting-percentage', 'value': 'true-shooting-percentage'},
                                           {'label': 'offensive-efficiency', 'value': 'offensive-efficiency'},
                                           {'label': 'defensive-efficiency', 'value': 'defensive-efficiency'},
                                           {'label': 'net-adj-efficiency', 'value': 'net-adj-efficiency'},
                                           #{'label': 'three-pointers-made-per-game', 'value': 'three-pointers-made-per-game'},
                                           #{'label': 'three-pointers-attempted-per-game', 'value': 'three-pointers-attempted-per-game'},
                                           {'label': 'assist--per--turnover-ratio', 'value': 'assist--per--turnover-ratio'}], value='defensive-efficiency'),
                     html.Br(),
                     html.P('STAT B'),
                     dcc.Dropdown(id='statb',
                                  options=[{'label': 'effective-field-goal-pct', 'value': 'effective-field-goal-pct'},
                                           {'label': 'true-shooting-percentage', 'value': 'true-shooting-percentage'},
                                           {'label': 'offensive-efficiency', 'value': 'offensive-efficiency'},
                                           {'label': 'defensive-efficiency', 'value': 'defensive-efficiency'},
                                           {'label': 'net-adj-efficiency', 'value': 'net-adj-efficiency'},
                                           #{'label': 'three-pointers-made-per-game', 'value': 'three-pointers-made-per-game'},
                                           #{'label': 'three-pointers-attempted-per-game', 'value': 'three-pointers-attempted-per-game'},
                                           {'label': 'assist--per--turnover-ratio', 'value': 'assist--per--turnover-ratio'}], value='defensive-efficiency'),
                               html.Br(),
                               html.Div(id='layout')])

@matchup_app.callback(Output(component_id='layout', component_property='children'),
                      #Output(component_id='matchup-df', component_property='figure'),
                      Output(component_id='matchup-chart', component_property='figure'),
                      [Input(component_id='stata', component_property='value'),
                       Input(component_id='statb', component_property='value')])

def display_chart(stata, statb):
    fig = px.scatter(x=matchup_df[stata], y=matchup_df[statb])
    return fig

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8050)


#%%
#matchup_app.run_server(
#    port = 8040,
#    host = '0.0.0.0',
#    debug=True
#)

#%%

#%%
current_time = datetime.datetime.now()
#display_time = datetime.datetime.now()
print(f'{current_time:%Y-%m-%d %H:%M}')

#%%
#style_data_conditional = [{'if': {'row_index': 'odd'},
 #                          'backgroundColor': 'rgb(220, 220, 220)',
 #                          }],

#df['Rating'] = df['Humidity'].apply(lambda x:
#    '⭐⭐⭐' if x > 30 else (
#    '⭐⭐' if x > 20 else (
#    '⭐' if x > 10 else ''
#)))
#df['Growth'] = df['Temperature'].apply(lambda x: '↗️' if x > 0 else '↘️')
#df['Status'] = df['Temperature'].apply(lambda x: '🔥' if x > 0 else '🚒')




#%%
