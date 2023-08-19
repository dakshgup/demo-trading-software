import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
from stocks_list import dropdown_options
from new_all_time_high import check_for_all_time_high

# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Home',  # name of page, commonly used as name of link
)

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(options=dropdown_options)
                    ], xs=10, sm=10, md=8, lg=4, xl=4, xxl=4
                )
            ]
        ),
    ]
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
            children='Check for New All time Highs',
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
                        html.Div(
                            className='input-container',
                            children=[
                                html.Button('Check for all new time highs', id='submit-button1', n_clicks=0, style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'})
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
                        html.Div(id='output-list', style={'color': colors['text'], 'paddingTop': '10px'})
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                )
                
            ]
        ),
    ]
)

@callback(
    Output('output-list', 'children'),
    [Input('submit-button1', 'n_clicks')]
)
def update_output(n_clicks):
    if n_clicks is None or n_clicks <=0:
        return ''
    lst_daily = []
    lst_weekly = []
    for stock in dropdown_options:
        print(stock)
        ans,fig = check_for_all_time_high(stock['value'],interval='1d')
        if ans:
            lst_daily.append(stock['value'])
        ans,fig = check_for_all_time_high(stock['value'],interval='1wk')
        if ans:
            lst_weekly.append(stock['value'])
    if(len(lst_daily) == 0) and len(lst_weekly) == 0:
        x = ''
        for stock in dropdown_options:
            x += stock['label']
            x += ', '
        return f"No new weekly or daily high found in the stocks {x}"
    return [html.Div(f'New daily high found for {x}') for x in lst_daily] + [html.Div(f'New high found on weekly basis for {x}') for x in lst_weekly]
