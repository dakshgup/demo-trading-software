import dash
from dash import dcc, html,Input,Output
import plotly.express as px
import plotly.graph_objects as go
from point2pointDivergence2 import IntermarketDivergenceLows,IntermarketDivergenceHighs
import dash
from stocks_list import dropdown_options
# from stochastic_dash import runStochDivergence
# import dash_core_components as dcc
# import dash_html_components as html
from dash import callback, Output, Input
import datetime 
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/intermarket',
                   name='Inter-Divergence',
)
colors = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'button': '#87CEFA',
}
layout = html.Div(
    style={'backgroundColor': colors['background'], 'textAlign': 'center'},
    children=[
        html.H1(
            children='Intermarket Divergence',
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
                                dcc.Dropdown(id='input-s1', placeholder='Enter s1', options=dropdown_options, style={'width': '100%', 'padding': '10px'}),
                                dcc.Dropdown(id='input-s2', placeholder='Enter s2', options=dropdown_options, style={'width': '100%', 'padding': '10px'}),
                                dcc.Input(id='input-start-date', placeholder='Enter start date (YYYY-MM-DD)', type='text', style={'width': '100%', 'padding': '10px'}),
                                dcc.Input(id='input-end-date', placeholder='Enter end date (YYYY-MM-DD)', type='text', style={'width': '100%', 'padding': '10px'},value=today),
                                html.Button('Submit', id='submit-button', n_clicks=0, style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'})
                            ]
                        )
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                )
            ]
        ),
        html.Div(
            className='row',
            style={'padding': '20px'},
            children=[
                html.Div(
                    className='six columns',
                    children=[
                        html.H3('Output', style={'color': colors['text']}),
                        html.Div(id='output-low', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig1')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                
            ]
        ),
        html.Div(
            className='row',
            style={'padding': '20px'},
            children=[
                
                html.Div(
                    className='six columns',
                    children=[
                        html.Div(id='output-high', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig2')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                )
            ]
        )
    ]
)



@callback(
    Output('output-low', 'children'),
    Output('fig1', 'figure'),
    Output('output-high', 'children'),
    Output('fig2', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-s1', 'value'),
     dash.dependencies.State('input-s2', 'value'),
     dash.dependencies.State('input-start-date', 'value'),
     dash.dependencies.State('input-end-date', 'value')]
)
def update_output(n_clicks, s1, s2, start_date, end_date):
    if n_clicks is None or s1 is None or s2 is None or start_date is None or end_date is None:
        # Return empty figures if any of the inputs are not provided
        return " ",go.Figure()," " ,go.Figure()
    if n_clicks > 0:
        fighigh = go.Figure()
        figlow =go.Figure()
        found1, start1, end1, figlow = IntermarketDivergenceLows(s1, s2,start_date,end_date)
        if found1:
            outputlow = f"Bullish Divergence Observed between {s1} and {s2}. Observed on weekly charts from {str(start1).split(' ')[0]} to {str(end1).split(' ')[0]}."
        else:
            outputlow = f'No Bullish divergence found between the provided inputs.'
        found, start, end, fighigh = IntermarketDivergenceHighs(s1, s2,start_date,end_date)
        if found:
            outputhigh = f"Bearish Divergence Observed between {s1} and {s2}. Observed on weekly charts from {str(start).split(' ')[0]} to {str(end).split(' ')[0]}."
        else:
            outputhigh = f'No Bearish divergence found between the provided inputs.'
   
        return outputlow,figlow,outputhigh,fighigh

