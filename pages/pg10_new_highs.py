import dash
import plotly.express as px
import dash
from dash import dcc, html,Input,Output
import plotly.express as px
import plotly.graph_objects as go
# from point2pointDivergence2 import IntermarketDivergenceLows,IntermarketDivergenceHighs
import dash
# from stochastic_dash import runStochDivergence
from dash import callback, Output, Input,State
from new_all_time_high import check_for_all_time_high
from stocks_list import dropdown_options
import datetime 
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/newhighs',
                   name='New Highs',
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
            children='New All time High',
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
                                dcc.Dropdown(id='input-s11', placeholder='Enter s1', options=dropdown_options, style={'width': '100%', 'padding': '10px'}),
                                dcc.Dropdown(id='input-int11', placeholder='Choose frequency', options=[{'label':'daily', 'value':'1d'},{'label':'weekly','value':'1wk'}], style={'width': '100%', 'padding': '10px'}),
                                html.Button('Submit', id='submit-button11', n_clicks=0, style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'})
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
                        html.Div(id='output-ans', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        dcc.Graph(id='figx')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
                
            ]
        ),
    ]
)



@callback(
    Output('output-ans', 'children'),
    Output('figx', 'figure'),
    [Input('submit-button11', 'n_clicks')],
    [dash.dependencies.State('input-s11', 'value'),
    dash.dependencies.State('input-int11','value'),
    ]
)
def update_output(n_clicks, s1,interval):
    if s1 is None or n_clicks is None or n_clicks<=0 or interval is None:
        return '',go.Figure()
    ans,fig = check_for_all_time_high(s1,interval)
    if ans==False:
       return 'False', fig
    else:
       return 'True', fig
    return '',go.Figure()
