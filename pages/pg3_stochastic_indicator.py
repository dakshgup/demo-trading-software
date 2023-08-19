import dash
import plotly.express as px
from dash import callback, Output, Input,dcc,html
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from stoch_div_ui_connection import runStochDivergance
from stocks_list import dropdown_options
# import pandas as pd
# import numpy as np
# import pandas_ta as ta
import datetime 
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/interstat',
                   name='Stochastic Indicator',
)
colors = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'button': '#87CEFA',
}
layout = html.Div(
    style={'backgroundColor': colors['background'],'textAlign': 'center'},
    children=[
        html.H1(
            children='Stochastic Divergence',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'paddingTop': '20px'
            }
        ),
        html.Div(
            className='row',
            style={'padding': '20px'},
            children=[
                html.Div(
                    className='six columns',
                    children=[
                        html.H3('Input', style={'color': colors['text']}),
                        html.Div(
                            className='input-container',
                            children=[
                                html.Div(
                                    className='center-aligned',
                                    children=[
                                        html.Label('Stock Symbol', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.Dropdown(id='stock-input', options=dropdown_options, placeholder='Enter stock symbol'),
                                        html.Label('Start Date', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.DatePickerSingle(id='start-date', placeholder='Select start date'),

                                        html.Label('End Date', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.DatePickerSingle(id='end-date', placeholder='Select end date'),


                                        html.Button(
                                            'Submit',
                                            id='submit-button',
                                            n_clicks=0,
                                            style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'}
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        html.H3('Output', style={'color': colors['text']}),
                        dcc.Graph(id='fig4', figure=go.Figure())
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                )
            ]
        )
    ]
)

@callback(
    Output('fig4', 'figure'),
    Input('submit-button', 'n_clicks'),
    State('stock-input', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date')
)
def update_figures(n_clicks, stock_symbol, start_date, end_date):
    if n_clicks is None or stock_symbol is None or start_date is None or end_date is None:
        # Return empty figures if any of the inputs are not provided
        return go.Figure()

    figures = {}
    figures['fig4'],signals = runStochDivergance(stock_symbol, startDate_=start_date, R_=1.02, endDate_=end_date)
    return figures['fig4']
