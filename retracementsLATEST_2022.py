from turtle import color
from getPointsFile import getPointsBest
import matplotlib.pyplot as plt
from loess.loess_1d import loess_1d
import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from matplotlib import style
import matplotlib.dates as mdates
from CorrectR import getPoints
from matplotlib.ticker import ScalarFormatter
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib import ticker
from Utilities import Scaler
import plotly.graph_objects as go
import plotly.io as pio
import dash
from dash import dcc,html
from dash.dependencies import Input, Output,State

def diffrenciate(xout, yout,xx,yy):
    h = xout[1] - xout[0]
    yPrime = np.diff(yout)
    yPrime = yPrime/h
    xPrime = xx[1:]
    xPs, yPs,  _ = loess_1d(xPrime, yPrime, xnew=None, degree=1, frac=0.05, npoints=None, rotate=False, sigy=None)
    return xPs, yPs, xPrime, yPrime

def getInbetween(pair, minIndexes, xx,yy):
    allIndex = minIndexes
    if yy[pair[1]] < yy[pair[0]]:
        min_ = yy[pair[1]]
        max_ = yy[pair[0]]
    else:
        min_ = yy[pair[0]]
        max_ = yy[pair[1]]
    c = 0
    for a in allIndex:
        if min_ < yy[a] < max_:
            print(min_, yy[a], max_)
            c += 1
    if c < 1:
        return False
    return True

def getPairs(xx, yy, loesFraction, primeLoessFraction=0.05):
    xout, yout, _ = loess_1d(xx, yy, xnew=None, degree=1, frac=loesFraction, npoints=None, rotate=False, sigy=None)

    xPs, yPs, xPrime, yPrime = diffrenciate(xout, yout,xx,yy)
    h = xout[1] - xout[0]
    yPrimePrime = np.diff(yPs)
    yPrimePrime = yPrimePrime/h
    xPrimePrime = xx[2:]
    xPPs, yPPs, _ = loess_1d(xPrimePrime, yPrimePrime, xnew=None, degree=1, frac=primeLoessFraction, npoints=None, rotate=False, sigy=None)
    minIndexes = []
    maxIndexes = []
    i = 1
    while i < len(yPs):
        if yPs[i-1] < 0 and yPs[i] > 0:
            minIndexes.append(i)
        elif yPs[i-1] > 0 and yPs[i] < 0:
            maxIndexes.append(i)
        i += 1

    minIndexesCheck = []
    maxIndexesCheck = []
    i = 1
    while i < len(yPs):
        if yPs[i-1] < 0 and yPs[i] > 0:
            minIndexesCheck.append(i)
        elif yPs[i-1] > 0 and yPs[i] < 0:
            maxIndexesCheck.append(i)
        i += 1

    allIndexCheck = sorted(minIndexes+maxIndexes)

    allIndex = sorted(minIndexes+maxIndexes)
    allDiffs = []
    i = 0
    while i+1 < len(allIndex):
        a = xout[allIndex[i]+1]
        b = xout[allIndex[i+1]]
        allDiffs.append( [abs(a-b), (allIndex[i], allIndex[i+1])] )
        i += 1

    pairsInnit = [(minIndex, maxIndex) for minIndex in minIndexes for maxIndex in maxIndexes]

    pairs = []
    for diff in allDiffs:
        if getInbetween(diff[1], allIndexCheck, xx,yy) and abs(diff[1][0] - diff[1][1]) > 1:
            pairs.append(diff[1])

    print(len(pairs), len(allDiffs))
    return pairs

def plotter(pairs,xx,yy,xScaler,yScaler,intervalSET,data,y,x):
    cols = ['black', 'blue', 'green', "yellow"]
    currentPrice = yScaler.getScaledvalue(y[-1])
    print(currentPrice)
    data = data.copy()

    plotted = []
    data = data.reset_index()
    data['Date'] = data['Date'].map(mdates.date2num)
    j = 0
    for pp in pairs:
        print(pp, "uy4rwegk")

        if yy[pp[0]] < yy[pp[1]]:
            min_chosen = yy[pp[0]]
            max_chosen = yy[pp[1]]  
            p = (pp[0], pp[1])  
        else:
            max_chosen = yy[pp[0]]
            min_chosen = yy[pp[1]]  
            p = (pp[1], pp[0])  

        min_ = min(yy[ max(0,p[0]-7):p[0]+7])
        max_ = max(yy[max(0,p[1]-7):p[1]+7])

        max_index = np.argmax(np.asarray(yy[ max(0,p[0]-7):p[0]+7]))
        min_index = np.argmin(np.asarray(yy[ max(0,p[0]-7):p[0]+7]))

        actualMinINdex = min_index-7+max(0,p[0]-7)
        actualMaxINdex = max_index-7+max(0,p[1]-7)

        firstDate = data['Date'][0]
        if intervalSET == '1wk':
            plottedPoint = [(xScaler.getUnscaledValue(min(xx[p[0]],xx[p[0]]))*7)+firstDate, (xScaler.getUnscaledValue(xx[p[1]])*7)+firstDate], [10**yScaler.getUnscaledValue(min_), 10**yScaler.getUnscaledValue(min_)]
        if intervalSET == '1mo':
            plottedPoint = [ (xScaler.getUnscaledValue(min(xx[p[0]],xx[p[0]]))*30.5)+firstDate, (xScaler.getUnscaledValue(min(xx[p[1]],xx[p[1]]))*30.5)+firstDate], [10**yScaler.getUnscaledValue(min_), 10**yScaler.getUnscaledValue(min_)]
        print(min_chosen <= max_chosen, "jhbjhbn")

        if not plottedPoint in plotted:
            plotted.append(plottedPoint)
            tickerEvery = (max(data["high"]) - min(data["low"]))/10
            ax1 = plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)
            ax1.xaxis_date()
            ax1.set_yscale('log')
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(tickerEvery)) 
            ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
            ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3,6,9)))
            ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            for label in ax1.get_xticklabels(which='major'):
                label.set(rotation=60)

            ax1.yaxis.set_minor_locator(ticker.NullLocator()) 
            ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())  
            ax1.grid(True)

            candlestick_ohlc(ax1, data.values, width=0.7)

            from datetime import datetime
            date_format = "%m/%d/%Y"
            a = datetime.strptime('8/18/2008', date_format)
            b = datetime.strptime('9/26/2008', date_format)
            delta = b - a
            print(delta.days)

            firstDate = data['Date'][0]
            print(firstDate)

            if intervalSET == '1wk':
                ax1.plot([ (xScaler.getUnscaledValue(min(xx[p[0]],xx[p[0]]))*7)+firstDate, (xScaler.getUnscaledValue(1)*7)+firstDate], [10**yScaler.getUnscaledValue(min_), 10**yScaler.getUnscaledValue(min_)])
                ax1.plot([ (xScaler.getUnscaledValue(min(xx[p[1]],xx[p[1]]))*7)+firstDate, (xScaler.getUnscaledValue(1)*7)+firstDate], [10**yScaler.getUnscaledValue(max_), 10**yScaler.getUnscaledValue(max_)])
            elif intervalSET == '1mo':
                ax1.plot([ (xScaler.getUnscaledValue(min(xx[actualMinINdex],xx[actualMinINdex]))*30.5)+firstDate, (xScaler.getUnscaledValue(1)*30.5)+firstDate], [10**yScaler.getUnscaledValue(min_), 10**yScaler.getUnscaledValue(min_)])
                ax1.plot([ (xScaler.getUnscaledValue(min(xx[actualMaxINdex],xx[actualMaxINdex]))*30.5)+firstDate, (xScaler.getUnscaledValue(1)*30.5)+firstDate], [10**yScaler.getUnscaledValue(max_), 10**yScaler.getUnscaledValue(max_)])

            nums = [0.2366, 0.382, 0.5, 0.618, 0.764, 0.786, 0.886]
            for num in nums:
                if intervalSET == '1wk':
                    y = 10**yScaler.getUnscaledValue(max_) - ((10**yScaler.getUnscaledValue(max_) - 10**yScaler.getUnscaledValue(min_))*num)
                    ax1.plot([ (xScaler.getUnscaledValue(min(xx[p[1]],xx[p[1]]))*7)+firstDate, (xScaler.getUnscaledValue(1)*7)+firstDate], [y, y],  '-', label=str(num))
                elif intervalSET == '1mo':
                    y = 10**yScaler.getUnscaledValue(max_) - ((10**yScaler.getUnscaledValue(max_) - 10**yScaler.getUnscaledValue(min_))*num)
                    ax1.plot([ (xScaler.getUnscaledValue(min(xx[p[1]],xx[p[1]]))*30.5)+firstDate, (xScaler.getUnscaledValue(1)*30.5)+firstDate], [y, y],  '-', label=str(num))
    
            j += 1
            plt.legend()
        plt.show()

def plotter1(pairs, xx, yy, xScaler, yScaler, intervalSET, data, y):
    figures=[]
    cols = ['black', 'blue', 'green', "yellow"]
    currentPrice = yScaler.getScaledvalue(y[-1])
    print(currentPrice)
    data = data.copy()

    plotted = []
    data = data.reset_index()
    data['Date1'] = data['Date'].map(mdates.date2num)
    print("date",data)
    j = 0
     # Declare figures as a global variable

    for pp in pairs:
        print(pp, "uy4rwegk")

        if yy[pp[0]] < yy[pp[1]]:
            min_chosen = yy[pp[0]]
            max_chosen = yy[pp[1]]
            p = (pp[0], pp[1])
        else:
            max_chosen = yy[pp[0]]
            min_chosen = yy[pp[1]]
            p = (pp[1], pp[0])

        min_ = min(yy[max(0, p[0] - 7):p[0] + 7])
        max_ = max(yy[max(0, p[1] - 7):p[1] + 7])

        max_index = np.argmax(np.asarray(yy[max(0, p[0] - 7):p[0] + 7]))
        min_index = np.argmin(np.asarray(yy[max(0, p[0] - 7):p[0] + 7]))

        actualMinINdex = min_index - 7 + max(0, p[0] - 7)
        actualMaxINdex = max_index - 7 + max(0, p[1] - 7)

        firstDate = data['Date1'][0]
        f1=data['Date'][0]
        l1=data['Date'][len(data)-1]
        if intervalSET == '1wk':
            plottedPoint = [
                (xScaler.getUnscaledValue(min(xx[p[0]], xx[p[0]])) * 7) + firstDate,
                (xScaler.getUnscaledValue(xx[p[1]]) * 7) + firstDate
            ], [
                10 ** yScaler.getUnscaledValue(min_),
                10 ** yScaler.getUnscaledValue(min_)
            ]
        if intervalSET == '1mo':
            plottedPoint = [
                (xScaler.getUnscaledValue(min(xx[p[0]], xx[p[0]])) * 30.5) + firstDate,
                (xScaler.getUnscaledValue(min(xx[p[1]], xx[p[1]])) * 30.5) + firstDate
            ], [
                10 ** yScaler.getUnscaledValue(min_),
                10 ** yScaler.getUnscaledValue(min_)
            ]
        
        print(min_chosen <= max_chosen, "jhbjhbn")

        if not plottedPoint in plotted:
            plotted.append(plottedPoint)

            candlestick = go.Candlestick(
                x=data['Date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )

            layout = go.Layout(
                title='Candlestick Chart',
                yaxis=dict(type='log', autorange=True),
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d',  
                    rangeslider=dict(visible=False)
                )
            )

            fig = go.Figure(data=[candlestick], layout=layout)

            fig.update_xaxes(
                rangeslider=dict(visible=False)
            )
            fig.update_layout(
            autosize=False,
            width=1200,
            height=800,)

            fig.add_trace(go.Scatter(
                x=[f1,
                l1],
                y=[10 ** yScaler.getUnscaledValue(min_), 10 ** yScaler.getUnscaledValue(min_)],
                mode='lines'
            ))

            fig.add_trace(go.Scatter(
                     x=[f1,
                l1],
                y=[10 ** yScaler.getUnscaledValue(max_), 10 ** yScaler.getUnscaledValue(max_)],
                mode='lines'
            ))

            nums = [0.2366, 0.382, 0.5, 0.618, 0.764, 0.786, 0.886]
            for num in nums:
                if intervalSET == '1wk':
                    y = 10 ** yScaler.getUnscaledValue(max_) - (
                            (10 ** yScaler.getUnscaledValue(max_) - 10 ** yScaler.getUnscaledValue(min_)) * num)
                    fig.add_trace(go.Scatter(
                        x=[f1,l1],
                        y=[y, y],
                        mode='lines',
                        name=str(num)
                    ))
                elif intervalSET == '1mo':
                    y = 10 ** yScaler.getUnscaledValue(max_) - (
                            (10 ** yScaler.getUnscaledValue(max_) - 10 ** yScaler.getUnscaledValue(min_)) * num)
                    fig.add_trace(go.Scatter(
                        x=[f1,l1],
                        y=[y, y],
                        mode='lines',
                        name=str(num)
                    ))
                

            figures.append(fig)

        j += 1

    return figures
figures=[]
colors = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'button': '#87CEFA',
}
app = dash.Dash(__name__)
app.layout = html.Div(
    style={'backgroundColor': colors['background'],'textAlign': 'center'},
    children=[
        html.H1(
            children='Plotly Dash - Multiple Figures',
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
                                html.Label('Stock Symbol', style={'color': colors['text'], 'padding': '20px'}),
                                dcc.Input(id='symbol-input', type='text', value='amzn'),

                                html.Label('Start Date', style={'color': colors['text'], 'padding': '20px'}),
                                dcc.Input(id='date-input', type='text', value='1990-01-01'),

                                html.Label('Interval', style={'color': colors['text'], 'padding': '20px'}),
                                dcc.Dropdown(
                                    id='interval-dropdown',
                                    options=[
                                        {'label': '1 Week', 'value': '1wk'},
                                        {'label': '1 Month', 'value': '1mo'}
                                    ],
                                    value='1mo'
                                ),

                                html.Button(
                                    'Submit',
                                    id='submit-button',
                                    n_clicks=0,
                                    style={'backgroundColor': colors['button'], 'borderRadius': '20px', 'padding': '10px'}
                                ),
                            ]
                        ),
                    ],
                    style={'backgroundColor': '#424242', 'padding': '10px', 'borderRadius': '10px'}
                ),
                html.Div(
                    className='six columns',
                    children=[
                        html.H3('Output', style={'color': colors['text']}),
                        html.Div(id='graphs-container')
                    ],
                    style={'backgroundColor': '#424242', 'padding': '20px', 'borderRadius': '10px'}
                )
            ]
        )
    ]
)


@app.callback(Output('graphs-container', 'children'), [Input('submit-button', 'n_clicks')], [State('symbol-input', 'value'), State('date-input', 'value'), State('interval-dropdown', 'value')])
def run_retrace(n_clicks, symbol, start_date, interval):
  if n_clicks > 0:
    intervalSET = '1mo'
    data, highs, lows = getPointsBest('amzn', startDate='1990-01-01', interval=intervalSET, min_=0.5)
    print(data)
    y = np.log10(data["close"])
    yh = np.log10(data["high"])
    yl = np.log10(data["low"])
    x = np.linspace(0, len(data)-1, len(data))

    xScaler = Scaler(x)
    yScaler = Scaler(y)
    yy = yScaler.getScaled()
    xx = xScaler.getScaled()

    xScalerHigh = Scaler(x)
    yScalerHigh = Scaler(yh)
    yyh = yScaler.getScaled()
    xxh = xScaler.getScaled()

    xScalerLow = Scaler(x)
    yScalerLow = Scaler(yl)
    yyl = yScaler.getScaled()
    xxl = xScaler.getScaled()

    xx_array = np.array(xx)
    yy_array = np.array(yy)

    pairs0 = getPairs(xx_array, yy_array, 0.05)
    pairs1 = getPairs(xx_array, yy_array, 0.08)
    pairs2 = getPairs(xx_array, yy_array, 0.1)
    pairs11 = list(set(pairs1)|set(pairs2))

    pairs = []
    min__ = 100000000000000
    min_idx = 0
    for pp in pairs11:
        if yy[pp[0]] < yy[pp[1]]:
            min_chosen = yy[pp[0]]
            min_index_chosen = pp[0]
            max_chosen = yy[pp[1]]  
            max_index_chosen = pp[1]
            p = (pp[0], pp[1])  
        else:
            max_chosen = yy[pp[0]]
            max_index_chosen = pp[0]
            min_chosen = yy[pp[1]] 
            min_index_chosen = pp[1] 
            p = (pp[1], pp[0]) 
        if min__ > min_chosen:
            min__ = min_chosen 
            min_idx = min_index_chosen 

    for pp in pairs11:
        pairs.append( (min_idx, pp[1]) )

    figures = plotter1(pairs, xx, yy, xScaler, yScaler, intervalSET, data, y)

    return [dcc.Graph(figure=fig) for fig in figures]
    

if __name__ == '__main__':
    app.run_server(debug=True)
