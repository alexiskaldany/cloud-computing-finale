import dash
import dash_table
import numpy as np
import pandas as pd
import dash as dash
from dash import html
from dash import dcc
from dash.dependencies import Input,Output
import plotly.express as px
from scipy.fft import fft
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df_1 = pd.read_csv('matchup1.csv')

df_2 = pd.read_csv('matchup2.csv')

df_3 = pd.read_csv('matchup3.csv')

df_4 = pd.read_csv('matchup4.csv')

df_5 = pd.read_csv('matchup5.csv')

df_6 = pd.read_csv('matchup6.csv')

df_7 = pd.read_csv('matchup7.csv')

my_app = dash.Dash('Cloud Computing Project', external_stylesheets=external_stylesheets)

application = my_app.server

my_app.layout = html.Div([html.H1('NBA matchup win predictor', style={'textAlign': 'center', 'background': '#768ca3'}),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                                   children=[
                                       dcc.Tab(label='Matchup 1', value='Matchup 1'),
                                       dcc.Tab(label='Matchup 2', value='Matchup 2'),
                                       dcc.Tab(label='Matchup 3', value='Matchup 3'),
                                       dcc.Tab(label='Matchup 4', value='Matchup 4'),
                                       dcc.Tab(label='Matchup 5', value='Matchup 5'),
                                       dcc.Tab(label='Matchup 6', value='Matchup 6'),
                                       dcc.Tab(label='Matchup 7', value='Matchup 7')



                                   ]),
                          html.Div(id='layout')])

matchup1_layout = html.Div([
    html.H1('Charolette hornets vs Orlando Magic', style={'textAlign': 'center'}),
    dash_table.DataTable(df_1.to_dict('records'), [{"name": i, "id": i} for i in df_1.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the Charolette hornets to win tonight', style={'textAlign': 'center'})

    ])

matchup2_layout = html.Div([
    html.H1('Toronto Raptors vs Philadelphia 76ers', style={'textAlign': 'center'}),
    dash_table.DataTable(df_2.to_dict('records'), [{"name": i, "id": i} for i in df_2.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the Philadelphia 76ers to win tonight', style={'textAlign': 'center'})

    ])

matchup3_layout = html.Div([
    html.H1('Milwaukee Bucks vs Boston Celtics', style={'textAlign': 'center'}),
    dash_table.DataTable(df_3.to_dict('records'), [{"name": i, "id": i} for i in df_3.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the Milwaukee Bucks to win tonight', style={'textAlign': 'center'})

    ])

matchup4_layout = html.Div([
    html.H1('Minnesota Timberwolves vs San Antonio Spurs', style={'textAlign': 'center'}),
    dash_table.DataTable(df_4.to_dict('records'), [{"name": i, "id": i} for i in df_4.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the Minnesota Timberwolves to win tonight')

    ])

matchup5_layout = html.Div([
    html.H1('New Orleans Pelicans vs Portland Trailblazers', style={'textAlign': 'center'}),
    dash_table.DataTable(df_5.to_dict('records'), [{"name": i, "id": i} for i in df_5.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the New Orleans Pelicans to win tonight', style={'textAlign': 'center'})

    ])

matchup6_layout = html.Div([
    html.H1('Denver Nuggets vs Memphis Grizzlies', style={'textAlign': 'center'}),
    dash_table.DataTable(df_6.to_dict('records'), [{"name": i, "id": i} for i in df_6.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the Denver Nuggets to win tonight', style={'textAlign': 'center'})

    ])

matchup7_layout = html.Div([
    html.H1('Golden State Warriors vs Los Angeles Lakers', style={'textAlign': 'center'}),
    dash_table.DataTable(df_7.to_dict('records'), [{"name": i, "id": i} for i in df_7.columns], style_data={ 'border': '#324f6e' }),
    html.H3('We predict the Golden State Warriors to win tonight', style={'textAlign': 'center'})

    ])

@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])
def update_layout(ques):
    if ques == 'Matchup 1':
        return matchup1_layout
    elif ques == 'Matchup 2':
        return matchup2_layout
    elif ques == 'Matchup 3':
        return matchup3_layout
    elif ques == 'Matchup 4':
        return matchup4_layout
    elif ques == 'Matchup 5':
        return matchup5_layout
    elif ques == 'Matchup 6':
        return matchup6_layout
    elif ques == 'Matchup 7':
        return matchup7_layout


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8050)

