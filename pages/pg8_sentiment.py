import dash
from dash import dcc, html
import plotly.express as px
import datetime 
today = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dash.register_page(__name__,
                   path='/sentiment',
                   name='Sentiment',
)

layout = html.Div(
    [
        dcc.Markdown('Sentiment Analysis', style={'textAlign':'center'}),
    ]
)
