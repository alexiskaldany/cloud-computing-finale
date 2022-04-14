import dash
from dash import dcc
from dash import html
from dash import Dash, dash_table, html, dcc, Input, Output
import dash_bootstrap_components as dbc
#import boto3
import pandas as pd


# reading data
df = pd.read_csv('prediction.csv')
# App instantiation
app = dash.Dash(__name__)
application = app.server
#  external_stylesheets=[
#     dbc.themes.CYBORG,
# ],
# # Layout
# style= {'background-image':
#                                                                            'url("/assets/background.png")',
#                                                                            'background-repeat': 'no-repeat',
#                                                                            'background-position': 'center',
#                                                                            'background-size': 'auto'}
app.layout = html.Div(children=[
html.H1(
        id='Header', children=['Difference Between Home and Away'], style={'textAlign': 'center',}),
    dash_table.DataTable
    (id='Predictions DataTable',
     data=df.to_dict('records'),
     columns=[{"name": i, "id": i}
              for i in df.columns],
     style_cell={'textAlign': 'center', 'width': '{}%'.format(len(
         df.columns))},
     page_size=33,
     style_header={
         'backgroundColor': 'rgb(30, 30, 30)',
         'color': 'white',
     },
     fixed_columns={'headers': True, 'data': 1},
     sort_action='native',
     style_data={'border': '1px solid grey',
                 'whiteSpace': 'normal',
                 'height': 'auto',
                 'lineHeight': '15px',
                #  'backgroundColor': 'rgb(500, 0, 0)',
                 'color': 'black',
                 },
     style_table={'minWidth': '100%'},
     merge_duplicate_headers=True,
     )
    #,html.Img(id='image',src="/assets/background.png")
])

# You will need to put this line at the bottom of your code to run #the application.

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8050)
