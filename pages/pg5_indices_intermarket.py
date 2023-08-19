import dash
from dash import dcc, html,Input,Output
import plotly.express as px
import plotly.graph_objects as go
from point2pointDivergence2 import IntermarketDivergenceLows,IntermarketDivergenceHighs
import dash
# from stochastic_dash import runStochDivergence
from dash import callback, Output, Input,State
import datetime 
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/intermarketstat',
                   name='DJI GSPC NDX Divergence',
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
            children='DJI GSPC NDX Divergence',
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
                        html.H3('^NDX ^GSPC', style={'color': colors['text']}),
                        html.Div(id='output-low1', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig11')
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
                        html.Div(id='output-high1', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig21')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
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
                        html.H3('^GSPC ^DJI', style={'color': colors['text']}),
                        html.Div(id='output-low2', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig12')
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
                        html.Div(id='output-high2', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig22')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
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
                        html.H3('^NDX ^DJI', style={'color': colors['text']}),
                        html.Div(id='output-low3', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig13')
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
                        html.Div(id='output-high3', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='fig23')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                )
            ]
        )
    ]
)



@callback(
    [
        Output('output-low1', 'children'),
        Output('fig11', 'figure'),
        Output('output-high1', 'children'),
        Output('fig21', 'figure'),
        Output('output-low2', 'children'),
        Output('fig12', 'figure'),
        Output('output-high2', 'children'),
        Output('fig22', 'figure'),
        Output('output-low3', 'children'),
        Output('fig13', 'figure'),
        Output('output-high3', 'children'),
        Output('fig23', 'figure'),
    ],
    [
        Input('submit-button', 'n_clicks')
    ],
    [
        State('input-start-date', 'value'),
        State('input-end-date', 'value')
    ]
)
def update_output(n_clicks, start_date, end_date):
    # print(start_date,end_date)
    if n_clicks is None or start_date is None or end_date is None:
        # Return empty figures if any of the inputs are not provided
        return [" ",go.Figure()," " ,go.Figure()]*3
    if n_clicks > 0:
        pairs = [
            {'s1': '^NDX', 's2': '^GSPC'},
            {'s1': '^GSPC', 's2': '^DJI'},
            {'s1': '^NDX', 's2': '^DJI'}
        ]
        outputs = []
        for pair in pairs:
            s1 = pair['s1']
            s2 = pair['s2']
            fighigh = go.Figure()
            figlow = go.Figure()
            found1, start1, end1, figlow = IntermarketDivergenceLows(s1, s2,start_date,end_date)
            if found1:
                outputlow = f'Bullish Divergence Observed between {s1} and {s2}. Observed on weekly charts from {str(start1).split(" ")[0]} to {str(end1).split(" ")[0]}.'
            else:
                outputlow = f'No Bullish divergence found between the provided inputs.'
            found, start, end, fighigh = IntermarketDivergenceHighs(s1, s2,start_date,end_date)
            if found:
                outputhigh = f'Bearish Divergence Observed between {s1} and {s2}. Observed on weekly charts from {str(start).split(" ")[0]} to {str(end).split(" ")[0]}.'
            else:
                outputhigh = f'No Bearish divergence found between the provided inputs.'
            outputs.extend([outputlow, figlow, outputhigh, fighigh])
        return outputs


