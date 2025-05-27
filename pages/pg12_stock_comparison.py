import dash
from dash import dcc, html, callback, Output, Input
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import dash_bootstrap_components as dbc
from stocks_list import dropdown_options

dash.register_page(__name__,
                   path='/stock-comparison',
                   name='Stock Comparison',
)

colors = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'button': '#87CEFA',
}

layout = html.Div(
    style={'backgroundColor': colors['background'], 'padding': '20px'},
    children=[
        html.H1(
            children='Stock Performance Comparison',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'marginBottom': '30px'
            }
        ),
        
        dbc.Row([
            dbc.Col([
                html.Label('Select First Stock:', style={'color': colors['text'], 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='stock1-dropdown',
                    options=dropdown_options,
                    value='AAPL',
                    style={'marginBottom': '20px'}
                )
            ], width=4),
            
            dbc.Col([
                html.Label('Select Second Stock:', style={'color': colors['text'], 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='stock2-dropdown',
                    options=dropdown_options,
                    value='MSFT',
                    style={'marginBottom': '20px'}
                )
            ], width=4),
            
            dbc.Col([
                html.Label('Time Period:', style={'color': colors['text'], 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='period-dropdown',
                    options=[
                        {'label': '1 Month', 'value': '1mo'},
                        {'label': '3 Months', 'value': '3mo'},
                        {'label': '6 Months', 'value': '6mo'},
                        {'label': '1 Year', 'value': '1y'},
                        {'label': '2 Years', 'value': '2y'}
                    ],
                    value='6mo',
                    style={'marginBottom': '20px'}
                )
            ], width=4)
        ]),
        
        html.Div(id='comparison-chart'),
        
        html.Div(id='comparison-stats', style={'marginTop': '20px'})
    ]
)

@callback(
    [Output('comparison-chart', 'children'),
     Output('comparison-stats', 'children')],
    [Input('stock1-dropdown', 'value'),
     Input('stock2-dropdown', 'value'),
     Input('period-dropdown', 'value')]
)
def update_comparison(stock1, stock2, period):
    if not stock1 or not stock2:
        return html.Div(), html.Div()
    
    try:
        # Fetch data for both stocks
        data1 = yf.download(stock1, period=period)
        data2 = yf.download(stock2, period=period)
        
        if data1.empty or data2.empty:
            return html.Div("Error fetching data", style={'color': colors['text']}), html.Div()
        
        # Calculate normalized performance (percentage change from start)
        data1_norm = ((data1['Close'] / data1['Close'].iloc[0]) - 1) * 100
        data2_norm = ((data2['Close'] / data2['Close'].iloc[0]) - 1) * 100
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data1_norm.index,
            y=data1_norm.values,
            mode='lines',
            name=f'{stock1}',
            line=dict(color='#00CC96', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data2_norm.index,
            y=data2_norm.values,
            mode='lines',
            name=f'{stock2}',
            line=dict(color='#EF553B', width=2)
        ))
        
        fig.update_layout(
            title=f'{stock1} vs {stock2} - Normalized Performance (%)',
            xaxis_title='Date',
            yaxis_title='Performance (%)',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
            height=500
        )
        
        chart = dcc.Graph(figure=fig)
        
        # Calculate statistics
        stock1_return = data1_norm.iloc[-1]
        stock2_return = data2_norm.iloc[-1]
        stock1_volatility = data1_norm.std()
        stock2_volatility = data2_norm.std()
        
        stats = dbc.Row([
            dbc.Col([
                html.H4(f'{stock1} Stats', style={'color': '#00CC96'}),
                html.P(f'Total Return: {stock1_return:.2f}%', style={'color': colors['text']}),
                html.P(f'Volatility: {stock1_volatility:.2f}%', style={'color': colors['text']})
            ], width=6),
            dbc.Col([
                html.H4(f'{stock2} Stats', style={'color': '#EF553B'}),
                html.P(f'Total Return: {stock2_return:.2f}%', style={'color': colors['text']}),
                html.P(f'Volatility: {stock2_volatility:.2f}%', style={'color': colors['text']})
            ], width=6)
        ])
        
        return chart, stats
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['text']}), html.Div()