import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.subplots 
from plotly.subplots import make_subplots
# from scipy import stats
import yfinance as yf
from plotly.subplots import make_subplots
import dash
from dash.dependencies import Input, Output, State
from datetime import date
from trendline import calculate_combined_trades, is_pivot, is_breakout, exportcsv ,plot_stock_data,calculate_breakpoint_pos,calculate_exit_date,calculate_point_pos,calculate_profit_or_stopped,collect_channel
from stocks_list import dropdown_options
import datetime 
from datetime import timedelta
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/trendlines',  # represents the url text
                   name='Trendlines_1',  # name of page, commonly used as name of link
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
            children='Trendlines',
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
                                        html.Label('Ticker', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.Dropdown(id='ticker-input',options=dropdown_options),

                                        html.Label('Start Date', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.Input(id='start-date-input', type='text', value='2022-01-01'),

                                        html.Label('End Date', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.Input(id='end-date-input', type='text', value=today),
                                    ]
                                ),
                                html.Div(
                                    className='center-aligned',
                                    children=[
                                        
                                        html.Label('Interval', style={'color': colors['text'], 'padding': '20px'}),
                                        dcc.Dropdown(
                                                id='interval-input',
                                                options=[
                                                    {'label': 'daily', 'value': '1D'},
                                                    {'label': 'Weekly', 'value': '1wk'},
                                                    {'label': 'Monthly', 'value': '1mo'},
                                                ],
                                                placeholder="Interval"
                                            ),

                                        dcc.Dropdown(
                                                id='level-input',
                                                options=[
                                                    {'label': '2', 'value': 2},
                                                    {'label': '4', 'value': 4},
                                                    {'label': '6', 'value': 6},
                                                    {'label': '8', 'value': 8},
                                                    {'label': '10', 'value': 10}
                                                ],
                                                placeholder="Level"
                                            ),
                                
                                        html.Button(
                                            'Plot',
                                            id='plot-button',
                                            n_clicks=0,
                                            style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'}
                                        ),
                                        html.Button(
                                            'Export CSV',
                                            id='export-csv-button',
                                            n_clicks=0,
                                            style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'}
                                        ),
                                        html.Button(
                                            'Alerts',
                                            id='alerts-button',
                                            n_clicks=0,
                                            style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'}
                                        ),
                                    ]
                                    
                                ),
                            ]
                        ),
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        html.Div(
                           className='six columns',
                           children=[
                              html.Div(id='alerts-message', style={'color': colors['text'], 'paddingTop': '10px'})
                           ],
                           style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                           ),
                        html.H3('Output', style={'color': colors['text']}),
                        dcc.Graph(id='plot-output')
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
                        html.Div(id='export-csv-message', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                ),
            ]
        )
    ]
)

@callback(
            Output('export-csv-message', 'children'),
            Input('export-csv-button', 'n_clicks'),
            State('ticker-input', 'value'),
            State('start-date-input', 'value'),
            State('end-date-input', 'value'),
            State('interval-input', 'value'),
            State('level-input', 'value')
        )
def export_csv(n_clicks, ticker, start_date, end_date, interval, level):
 if n_clicks > 0:
    combined_trades, success_rate, overall_return = calculate_combined_trades(ticker, start_date, end_date, interval)
    filename = f"{ticker}_{interval}_{success_rate:.2f}_{overall_return*100:.1f}%.csv"
    combined_trades.to_csv(filename, index=False)
    return html.P(f'CSV exported successfully! Success Rate: {success_rate:.2f}, Overall Return: {overall_return*100:.1f}%.')

@callback(
    Output('plot-output', 'figure'),
    Input('plot-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('start-date-input', 'value'),
    State('end-date-input', 'value'),
    State('interval-input', 'value'),
    State('level-input', 'value')
)
def update_plot(n_clicks, ticker, start_date, end_date, interval, level):
    if n_clicks is None or ticker is None or start_date is None or end_date is None or interval is None or level is None or n_clicks <= 0:
        # Return empty figures if the button has not been clicked or stock symbol is not provided
        return go.Figure()
    
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    from datetime import timedelta
    if interval=='1wk':
      if data.iloc[-1].name - data.iloc[-2].name < timedelta(7):
        data.drop(data.tail(1).index,inplace=True)
    data.to_csv("Edata.csv")
    df = pd.read_csv("Edata.csv")
    df_pl = df
    window = 3 * level
    backcandles = 10 * window

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Calculate ATR using pandas_ta
    atr = ta.atr(high=filtered_df['High'], low=filtered_df['Low'], close=filtered_df['Close'], length=14)

    atr_multiplier = 2
    stop_percentage = atr.iloc[-1] * atr_multiplier / filtered_df['Close'].iloc[-1]
    profit_percentage = (1 + (level - 1) / 4) * stop_percentage
    print(f"stoploss: {stop_percentage:.2f}%_takeprofit: {profit_percentage:.2f}%")

    df_pl['Date'] = pd.to_datetime(df_pl['Date'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(ticker, 'Volume'), row_width=[0.1, 0.7])

    fig.add_trace(
        go.Candlestick(
            hovertext=df_pl['Date'].dt.strftime('%Y-%m-%d'),
            x=df_pl.index,
            open=df_pl['Open'],
            high=df_pl['High'],
            low=df_pl['Low'],
            close=df_pl['Close'],
            name="Candlestick"
        ),
        row=1, col=1
    )

    df_pl['isPivot'] = df_pl.apply(lambda row: is_pivot(row.name, window, df_pl), axis=1)
    df_pl['pointpos'] = df_pl.apply(calculate_point_pos, axis=1)

    fig.add_trace(
        go.Scatter(
            x=df_pl.index,
            y=df_pl['pointpos'],
            mode="markers",
            marker=dict(size=10, color="MediumPurple"),
            name="Pivot"
        ),
        row=1, col=1
    )

    df['isBreakOut'] = 0
    for i in range(backcandles + window, len(df)):
        df.loc[i, 'isBreakOut'] = is_breakout(i, backcandles, window, df, stop_percentage)

    df['breakpointpos'] = df.apply(calculate_breakpoint_pos, axis=1)
    df_breakout = df[df['isBreakOut'] != 0]

    for candle in range(backcandles + window, len(df)):
        if df.iloc[candle].isBreakOut != 0:
            best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(
                candle, backcandles, window, df)
            extended_x = np.array(range(candle + 1, candle + 15))
            x1 = np.array(range(candle - best_back_l - window, candle + 1))
            x2 = np.array(range(candle - best_back_h - window, candle + 1))

            if r_sq_l >= 0.80 and df.iloc[candle].isBreakOut == 1:
                extended_y_lows = sl_lows * extended_x + interc_lows
                fig.add_trace(
                    go.Scatter(
                        x=extended_x,
                        y=extended_y_lows,
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Lower Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x1,
                        y=sl_lows * x1 + interc_lows,
                        mode='lines',
                        name='Lower Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )

            if r_sq_h >= 0.80 and df.iloc[candle].isBreakOut == 2:
                extended_y_highs = sl_highs * extended_x + interc_highs
                fig.add_trace(
                    go.Scatter(
                        x=extended_x,
                        y=extended_y_highs,
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Max Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x2,
                        y=sl_highs * x2 + interc_highs,
                        mode='lines',
                        name='Max Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )

    colors = ['black', 'blue']
    for color in colors:
        mask = df_breakout['isBreakOut'].map({1: 'black', 2: 'blue'}) == color
        fig.add_trace(
            go.Scatter(
                x=df_breakout.index[mask],
                y=df_breakout['breakpointpos'][mask],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color
                ),
                marker_symbol="hexagram",
                name="Breakout",
                legendgroup=color
            ),
            row=1, col=1
        )

    fig.update_traces(
        name="Break Below",
        selector=dict(legendgroup="black")
    )

    fig.update_traces(
        name="Break Above",
        selector=dict(legendgroup="blue")
    )

    fig.add_trace(
        go.Bar(
            x=df_pl['Date'],
            y=df_pl['Volume'],
            marker_color='red',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.update_layout(
      autosize=False,
      width=1200,
      height=800,)

    fig.update_layout(
        title_text=interval,
        xaxis_range=[df_pl.index.min(), df_pl.index.max()],
        yaxis_range=[df_pl['Low'].min(), df_pl['High'].max()],
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(rangemode='tozero'),
    )

    return fig

@callback(
            Output('alerts-message', 'children'),
            Input('alerts-button', 'n_clicks'),
            State('ticker-input', 'value'),
            State('interval-input', 'value'),
        )
def alerts(n_clicks, ticker, interval):
 if n_clicks > 0:
    todays_date = date.today()
    todays_date=str(todays_date)
    data = yf.download(ticker, start='2000-12-27', end=todays_date, interval=interval, progress=False)
    data.to_csv("Edata.csv")
    df = pd.read_csv("Edata.csv")
    df_pl = df
    output=f'No Signal Today for {ticker} on interval {interval}'
    for level in (2,11,2):
        window = 3 * level
        backcandles = 10 * window

        filtered_df = df[(df['Date'] >= '2000-12-27') & (df['Date'] <= todays_date)]

        # Calculate ATR using pandas_ta
        atr = ta.atr(high=filtered_df['High'], low=filtered_df['Low'], close=filtered_df['Close'], length=14)

        atr_multiplier = 2
        stop_percentage = atr.iloc[-1] * atr_multiplier / filtered_df['Close'].iloc[-1]
        profit_percentage = (1 + (level - 1) / 4) * stop_percentage
        print(f"stoploss: {stop_percentage:.2f}%_takeprofit: {profit_percentage:.2f}%")

        df_pl['Date'] = pd.to_datetime(df_pl['Date'])
        df_pl['isPivot'] = df_pl.apply(lambda row: is_pivot(row.name, window, df_pl), axis=1)
        signal= is_breakout(len(df)-1, backcandles, window, df, stop_percentage)
        if signal==1:
           output=f'Short Signal Today for {ticker} on interval {interval} with take profit= {profit_percentage:.2f} and stoploss= {stop_percentage:.2f} level of signal is {level}'
        elif signal==2:
           output=f'Long Signal Today for {ticker} on interval {interval} with take profit= {profit_percentage:.2f} and stoploss= {stop_percentage:.2f} level of signal is {level}'
    return html.Div([
            html.Pre(output),
        ])
