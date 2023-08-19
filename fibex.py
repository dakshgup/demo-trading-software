import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from CorrectR import getTuringPointsOnYear, getPointsforArray
import math
from loess.loess_1d import loess_1d
import finplot as fplt
from getPointsFile import getPointsBest, getPointsGivenR
from datetime import datetime

class Scaler:
    def __init__(self, series):
        self.s = series
        self.max = np.max(series)
        self.min = np.min(series)

    def getScaled(self):
        return np.subtract(self.s,self.min)/(self.max-self.min)
    
    def getUnscaled(self):
        return self.s
        

class Linear:
    def __init__(self, x1, y1, x2, y2, startIndex, endIndex):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.startIndex = startIndex
        self.endIndex = endIndex

    def getAngle(self, inDegress=True):
        cosTheta = (self.y2 - self.y1)/(math.sqrt((self.y2 - self.y1)*(self.y2 - self.y1) + (self.x2 - self.x1)*(self.x2 - self.x1)))
        theta = math.acos(cosTheta)
        if not inDegress:
            return theta
        else:
            return theta * 180/math.pi

    def getMangnitude(self):
        return math.sqrt((self.y2 - self.y1)*(self.y2 - self.y1) + (self.x2 - self.x1)*(self.x2 - self.x1))
        

# Returns the closest idex in list of turning points for a given index
def getClosestIndexinStock(Tp,list_):
    dist = 1000000000
    point = None
    for t in list_:
        d = abs(t-Tp)
        if d <= dist:
            dist = d
            point = t
    return point


# Returns a list of linear lines that approximate the stock, which can then be
# used to judge for a rise or a fall
def getLinearModel(x, data, highs, stockHighs):
    linears = []
    i = 0
    while i+1 < len(highs):
        linears.append(Linear(x[highs[i]], data[highs[i]], x[highs[i+1]], data[highs[i+1]], getClosestIndexinStock(highs[i],stockHighs) , getClosestIndexinStock(highs[i+1],stockHighs)))
        i += 1
    return linears

global df

def isThereOverlap(s1, s2, e1, e2):
    # if e2 > s1 or e1 > s2:     remove if any overlap
    if s1 > s2 > e1 and s1 > e2 > e1:  
        return True
    if s2 > s1 > e2 and s2 > e1 > e2:  
        return True
    if e1 > s2 > s1 and e1 > e2 > s1:  
        return True
    if e2 > s1 > s2 and e2 > e1 > s2:  
        return True
    else:
        return False

def removeConflicts(selectedLins1, data):
    selectedLins = []
    for s in selectedLins1:
        selectedLins.append(s)
    i = 0
    while i < len(selectedLins):
        s = selectedLins[i]
        startPrice = data.iloc[s.startIndex].high
        endPrice = data.iloc[s.endIndex].low
        range_ = startPrice - endPrice
        j = i + 1
        while j<len(selectedLins):
            s1 = selectedLins[j]
            startPrice1 = data.iloc[s1.startIndex].high
            endPrice1 = data.iloc[s1.endIndex].low
            range1 = startPrice1 - endPrice1
            if isThereOverlap(startPrice, startPrice1, endPrice, endPrice1):
                if range_ >= range1:
                    selectedLins.remove(selectedLins[j])
                    j = 0
                    i = -1
                else:
                    selectedLins.remove(selectedLins[i])
                    j = 0
                    i = -1
            j += 1
        i += 1
    return selectedLins

def update_crosshair_text(x, y, xtext, ytext):
        ytext = '%s \n C: %.2f\n O: %.2f\n H: %.2f\n L: %.2f\n ' % (ytext, (df.iloc[x].close), (df.iloc[x].open), (df.iloc[x].high), (df.iloc[x].low))
        return xtext, ytext
def runExtentions(STOCK, startDate_='1800-01-01', endDate_='2121-01-01', R_=None, returnData=False, showPlot=True, intervalGiven=False):
    global df

    # date_format = "%Y-%m-%d"
    # a = datetime.strptime(startDate_, date_format)
    # b = datetime.strptime(endDate_, date_format)
    # delta = b - a
    # days = delta.days
    # years = days/255

    data, highs, lows = getPointsGivenR(STOCK, 1.2, startDate=startDate_, endDate=endDate_, interval='1d')
    days = len(data)
    years = days/255

    if years < 3:
        interval_ = '1d'
    elif 3 < years < 8:
        interval_ = '1wk'
    else:
        interval_ = '1mo'
    
    print(interval_)

    if intervalGiven is False:
        if R_ == None:
            if interval_ == '1wk':
                data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.5, interval=interval_)
            elif interval_ == '1mo':
                data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.5, interval=interval_, max_=1)
            else:
                data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.4, interval=interval_)
        else:
            R = R_
            data, highs, lows = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_, interval=interval_)

        data = data.dropna()

        # get another copy to get the logData
        logData, highs, lows = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_, interval=interval_)
        logData = logData.dropna()
        logData = np.log10(logData)

        
    else:
        if R_ == None:
            if intervalGiven == '1wk':
                data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.5, interval=intervalGiven)
            elif intervalGiven == '1mo':
                data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.5, interval=intervalGiven, max_=1)
            else:
                data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.4, interval=intervalGiven)
        
        else:
            R = R_
            data, highs, lows = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_, interval=intervalGiven)
        data = data.dropna()
        # get another copy to get the logData
        logData, highs, lows = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_, interval=intervalGiven)
        logData = logData.dropna()
        logData = np.log10(logData)

    # Scale the logData to be b/w 0 and 1 for x and y axis
    x = np.linspace(1, len(logData), len(logData))
    y = np.asarray(logData["close"])
    ScalerX = Scaler(x)
    ScalerY = Scaler(y)
    xs = ScalerX.getScaled()
    ys = ScalerY.getScaled()

    xo, yo = xs, ys

    hi, lo = highs, lows


    Tps = sorted(highs+lows)
    print("points are ", len(Tps))

    # Get a list of lines to approximate the loess smooothned data
    lins = getLinearModel(xo, yo, sorted(hi+lo), sorted(highs+lows))

    print(len(lins), len(Tps))


    # plt.plot((logData['close']),'-o', markevery=highs+lows, markersize=5, fillstyle='none')
    # plt.show()

    print(data.head(50))

    fallingLins = [l for l in lins if l.getAngle() > 90]

    # sorted array of biggest falls 
    sortedLins = sorted(fallingLins, key=lambda l: l.getMangnitude(), reverse=True)

    for l in sortedLins:
        print(l.getAngle(), '   M: ', l.getMangnitude(), '  s: ', l.x1, '  e: ', l.x2, '\n')


    currentLins = [l for l in sortedLins if l.getMangnitude() > 0.05]
    relevantLins = currentLins


    SelectedLins__ =  relevantLins

    SelectedLins = removeConflicts(SelectedLins__, data)

    print("______---------_________---------___________---------_________---------_________")
    for l in SelectedLins:
        print(l.getAngle(), '   M: ', l.getMangnitude(), '  s: ', l.x1, '  e: ', l.x2, '\n')


    # plt.plot(xs, np.asarray(ys),'-o', markevery=highs+lows, markersize=5, fillstyle='none')
    # plt.plot(xo, yo, '-o', markevery=hi+lo, markersize=5, fillstyle='none')
    # plt.show()

    # Now plotting the data nicely
    df = data

    # ax = fplt.create_plot(yscale='log', title=STOCK)
    # fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']])

    # axo = ax.overlay(scale=0.2)
    # fplt.volume_ocv(df[['open','close','volume']], ax=axo)
    # # hover_label = fplt.add_legend('',ax=ax)


    # fplt.plot(df['close'].rolling(50).mean(), legend='ma-50D', ax=ax)
    # fplt.plot(df['close'].rolling(200).mean(), legend='ma-200D', ax=ax)

    # # # fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    # fplt.add_crosshair_info(update_crosshair_text, ax=ax)

    # # # ax.set_visible(crosshair=True, xaxis=True, yaxis=True, xgrid=True, ygrid=True)
    # # fplt.volume_ocv(df['Volume'], ax=ax.overlay())

    nums = [1.618, 2.618, 4.236, 6.854, 11.09, 17.944, 29.034, 46.978, 76.012]

    cols = ['r', 'k', 'b', 'y', 'm', 'r', 'g', 'k', 'b', 'y', 'm', 'g', 'r', 'k', 'b', 'y', 'm', 'g','r', 'k', 'b', 'y', 'm', 'g']
    c = 0
    # for s in SelectedLins:
    #     z = data.iloc[s.endIndex].low
    #     h = data.iloc[s.startIndex].high
    #     # fplt.add_line((s.startIndex, data.iloc[s.startIndex].high), (len(data),data.iloc[s.startIndex].high), ax=ax, width=3, interactive=True, color=cols[c])
    #     # fplt.add_line((s.startIndex, data.iloc[s.endIndex].low), (len(data),data.iloc[s.endIndex].low), ax=ax, width=3, interactive=True, color=cols[c])
    #     if 0.4*data['close'][-1] < (z + (h-z)*nums[-1]):
    #         for num in nums:
    #             # fplt.add_line((s.startIndex, z + (h-z)*num), (len(data), z+(h-z)*num), ax=ax, width=1, interactive=False, color=cols[c])
    #     c += 1

    # # if showPlot:
    #     fplt.show()
    
        
    if returnData:
        return data, SelectedLins
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Analysis"),
    html.Div([
        html.Label("Stock Symbol:"),
        dcc.Input(id="stock-input", type="text", value="AAPL"),
    ]),
    html.Div([
        html.Label("Start Date:"),
        dcc.Input(id="start-date-input", type="text", value="2020-01-01"),
    ]),
    html.Div([
        html.Label("End Date:"),
        dcc.Input(id="end-date-input", type="text", value="2022-01-01"),
    ]),
    html.Button(id="submit-button", children="Submit"),
    html.Div(id="output-plots"),
])

@app.callback(
    Output("output-plots", "children"),
    Input("submit-button", "n_clicks"),
    Input("stock-input", "value"),
    Input("start-date-input", "value"),
    Input("end-date-input", "value"),
)
def update_plots(n_clicks, stock, start_date, end_date):
    if n_clicks is None:
        return None
    
    # Call the runExtentions function to get data and selected lines
    data, selected_lines = runExtentions(stock, start_date, end_date, returnData=True, showPlot=False)
    
    # Create the Plotly figure with adjusted row heights
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            marker_color=data['close'].diff().apply(lambda x: 'green' if x >= 0 else 'red')
        ),
        row=2, col=1
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['close'].rolling(50).mean(),
            name='ma-50D',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['close'].rolling(200).mean(),
            name='ma-200D',
            line=dict(color='orange')
        ),
        row=1, col=1
    )

    # Selected Lines
    nums = [1.618, 2.618, 4.236, 6.854, 11.09, 17.944, 29.034, 46.978, 76.012]
    cols = ['red', 'black', 'blue', 'yellow', 'magenta', 'red', 'green', 'black', 'blue', 'yellow', 'magenta', 'green', 'red', 'black', 'blue', 'yellow', 'magenta', 'green','red', 'black', 'blue', 'yellow', 'magenta', 'green']
    c = 0
    for s in selected_lines:
        z = data.iloc[s.endIndex].low
        h = data.iloc[s.startIndex].high

        fig.add_shape(
            type="line",
            x0=data.index[s.startIndex],
            y0=h,
            x1=data.index[-1],
            y1=h,
            line=dict(color=cols[c], width=1),
            row=1, col=1
        )
        fig.add_shape(
            type="line",
            x0=data.index[s.startIndex],
            y0=z,
            x1=data.index[-1],
            y1=z,
            line=dict(color=cols[c], width=1),
            row=1, col=1
        )
        if 0.4 * data['close'][-1] < (z + (h - z) * nums[-1]):
            for num in nums:
                fig.add_shape(
                    type="line",
                    x0=data.index[s.startIndex],
                    y0=z + (h - z) * num,
                    x1=data.index[-1],
                    y1=z + (h - z) * num,
                    line=dict(color=cols[c], width=1, dash='dash'),
                    row=1, col=1
                )
        c += 1
        
    # Update x-axis and y-axis properties
    fig.update_xaxes(
        rangeslider_visible=False,
        range=[data.index[0], data.index[-1]],
        row=1, col=1
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        range=[data.index[0], data.index[-1]],
        row=2, col=1
    )
    fig.update_yaxes(type="log", title_text="Price", row=1, col=1)  # Set y-axis to logarithmic scale
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(
      autosize=False,
      width=1500,
      height=1200,)
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
