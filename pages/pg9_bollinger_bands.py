import dash
# import dash_core_components as dcc
from dash import dcc as dcc
from dash import html as html
# import dash_html_components as html
from dash import callback, Output, Input
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import pandas_ta as ta
import datetime 
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/bollinger',
                   name='Bollinger Bands',
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
            children='Stock Candlestick Chart with Bollinger Bands',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'paddingTop': '20px'
            }
        ),
        html.Div(
            style={'padding': '20px'},
            children=[
                html.Div(
                    children=[
                        html.Label('Ticker Symbol:', style={'color': colors['text'], 'padding': '10px'}),
                        dcc.Input(id='input-ticker', value='AAPL', type='text'),
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                ),
                html.Div(
                    children=[
                        html.Label('Start Date:', style={'color': colors['text'], 'padding': '10px'}),
                        dcc.Input(id='input-start-date', value='2020-01-01', type='text'),
                    
                
                        html.Label('End Date:', style={'color': colors['text'], 'padding': '10px'}),
                        dcc.Input(id='input-end-date', value=today, type='text'),
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                ),
                html.Div(
                    children=[
                        html.Label('Period (e.g., 20):', style={'color': colors['text'], 'padding': '10px'}),
                        dcc.Input(id='input-period', value='20', type='number'),
                 
                        html.Label('Interval (e.g., 1d):', style={'color': colors['text'], 'padding': '10px'}),
                        dcc.Input(id='input-interval', value='1d', type='text'),
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                ),
                html.Button(
                    'Submit',
                    id='submit-button',
                    n_clicks=0,
                    style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'}
                ),
                dcc.Graph(id='stock-chart')
            ],
            className='row'
        )
    ]
)


@callback(
    dash.dependencies.Output('stock-chart', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-ticker', 'value'),
     dash.dependencies.State('input-start-date', 'value'),
     dash.dependencies.State('input-end-date', 'value'),
     dash.dependencies.State('input-period', 'value'),
     dash.dependencies.State('input-interval', 'value')]
)
def update_chart(n_clicks, ticker, start_date, end_date, period, interval):
    # Retrieve stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Calculate Bollinger Bands using pandas_ta
    stock_data.ta.bbands(length=int(period), append=True)

    # Create Candlestick chart with Bollinger Bands
    figure = go.Figure(data=[
        go.Candlestick(x=stock_data.index,
                       open=stock_data['Open'],
                       high=stock_data['High'],
                       low=stock_data['Low'],
                       close=stock_data['Close'],
                       name='Candlestick'),
        go.Scatter(x=stock_data.index, y=stock_data[f'BBL_{period}_2.0'], name='Lower Band'),
        go.Scatter(x=stock_data.index, y=stock_data[f'BBM_{period}_2.0'], name='Moving Average'),
        go.Scatter(x=stock_data.index, y=stock_data[f'BBU_{period}_2.0'], name='Upper Band')
    ])
    figure.update_layout(
      autosize=False,
      width=1200,
      height=600,)
    figure.update_layout(
        title_text=interval,
        xaxis_range=[stock_data.index.min(), stock_data.index.max()],
        yaxis_range=[stock_data['Low'].min(), stock_data['High'].max()],
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(rangemode='tozero'),
    )
    return figure

