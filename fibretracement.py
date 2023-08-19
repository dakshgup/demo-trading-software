import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from turtle import color
from getPointsFile import getPointsBest
import matplotlib.pyplot as plt
from loess.loess_1d import loess_1d
import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import finplot as fplt
from matplotlib import style
import matplotlib.dates as mdates
from CorrectR import getPoints
from matplotlib.ticker import ScalarFormatter
from mpl_finance import candlestick_ohlc
from matplotlib import ticker
from Utilities import Scaler

def diffrenciate(xout, yout,xx):
    # diffrenciate
    h = xout[1] - xout[0]
    yPrime = np.diff(yout)
    yPrime = yPrime/h
    xPrime = xx[1:]
    # losss smoothining the derivative
    xPs, yPs,  _ = loess_1d(xPrime, yPrime, xnew=None, degree=1, frac=0.05, npoints=None, rotate=False, sigy=None)
    return xPs, yPs, xPrime, yPrime

# return false if there is min and miax between them (and they are not inflexion points)
def getInbetween(pair, minIndexes, yy):
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
    # xoutSmooth, youtSmooth, _ = loess_1d(xx, yy, xnew=None, degree=1, frac=0.05, npoints=None, rotate=False, sigy=None)
    # loess smoothning - 0.13
    xout, yout, _ = loess_1d(xx, yy, xnew=None, degree=1, frac=loesFraction, npoints=None, rotate=False, sigy=None)

    xPs, yPs, xPrime, yPrime = diffrenciate(xout, yout,xx)
    # xPsSmooth, yPsSmooth, xPrimeSmooth, yPrimeSmooth = diffrenciate(xoutSmooth, youtSmooth)

    # second derivative
    h = xout[1] - xout[0]
    yPrimePrime = np.diff(yPs)
    yPrimePrime = yPrimePrime/h
    xPrimePrime = xx[2:]
    # losss smoothining the derivative
    xPPs, yPPs, _ = loess_1d(xPrimePrime, yPrimePrime, xnew=None, degree=1, frac=primeLoessFraction, npoints=None, rotate=False, sigy=None)


    # plotting
    # plt.plot(xx, yy, '-o' ,markevery=highs+lows)
    # plt.plot(xout, yout, '-o' ,markevery=highs+lows)
    # plt.show()

    # detecting the points where it crosses zero
    minIndexes = []
    maxIndexes = []
    i = 1
    while i < len(yPs):
        if yPs[i-1] < 0 and yPs[i] > 0:
            # crossover has occored for minimum
            minIndexes.append(i)
            # minIndexes.append(i+1)
        elif yPs[i-1] > 0 and yPs[i] < 0:
            # crossover has occored for maxiumum
            maxIndexes.append(i)
            # maxIndexes.append(i+1)
        i += 1

    # detecting the points where it crosses zero
    minIndexesCheck = []
    maxIndexesCheck = []
    i = 1
    while i < len(yPs):
        if yPs[i-1] < 0 and yPs[i] > 0:
            # crossover has occored for minimum
            minIndexesCheck.append(i)
            # minIndexes.append(i+1)
        elif yPs[i-1] > 0 and yPs[i] < 0:
            # crossover has occored for maxiumum
            maxIndexesCheck.append(i)
            # maxIndexes.append(i+1)
        i += 1

    allIndexCheck = sorted(minIndexes+maxIndexes)

    # getting the are between consicitive maximas
    allIndex = sorted(minIndexes+maxIndexes)
    allDiffs = []
    i = 0
    while i+1 < len(allIndex):
        a = xout[allIndex[i]+1]
        b = xout[allIndex[i+1]]
        allDiffs.append( [abs(a-b), (allIndex[i], allIndex[i+1])] )
        i += 1

    # plotting the stcok and its derivative together 
    # fig, axs = plt.subplots(3)
    # fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(xx[1:], yy[1:], '-^', markevery=highs+lows)
    # axs[0].plot(xout[1:], yout[1:], '-o', markevery=minIndexes)
    # axs[0].plot(xout[1:], yout[1:], '-x', markevery=maxIndexes, color='orange')
    # axs[1].plot(xx[1:], yPrime)
    # axs[1].plot(xPs, yPs, '-o', markevery=minIndexes)
    # axs[1].plot(xPs, yPs, '-o', markevery=maxIndexes, color='orange')
    # # axs[1].plot(xPsSmooth, yPsSmooth)
    # axs[1].plot(xx[1:], np.zeros((len(xx[1:]))))
    # axs[2].plot(xx[2:], yPrimePrime)
    # axs[2].plot(xPPs, yPPs, '-o', markevery=minIndexes)
    # axs[2].plot(xPPs, yPPs, '-o', markevery=maxIndexes, color='orange')
    # axs[2].plot(xx[2:], np.zeros((len(xx[2:]))))
    # plt.show()

    # getting all possible comibinations of (min, max) indexes as zero hundred pairs
    pairsInnit = [(minIndex, maxIndex) for minIndex in minIndexes for maxIndex in maxIndexes]

    pairs = []
    for diff in allDiffs:
        if getInbetween(diff[1], allIndexCheck, yy) and abs(diff[1][0] - diff[1][1]) > 1:
            pairs.append(diff[1])

    # pairs = []
    # for pp in pairs1:
    #     if yy[pp[0]] < yy[pp[1]]:
    #         min_chosen = yy[pp[0]]
    #         max_chosen = yy[pp[1]]  
    #         p = (pp[0], pp[1])  
    #     else:
    #         max_chosen = yy[pp[0]]
    #         min_chosen = yy[pp[1]]  
    #         p = (pp[1], pp[0])  

    #     min_ = min(yy[ max(0,p[0]-7):p[0]+7])
    #     max_ = max(yy[max(0,p[1]-7):p[1]+7])

    #     max_index = np.argmax(np.asarray(yy[ max(0,p[0]-7):p[0]+7]))
    #     min_index = np.argmin(np.asarray(yy[ max(0,p[0]-7):p[0]+7]))

    #     pairs.append((min_index, max_index))

    print(len(pairs), len(allDiffs))

    return pairs
def retracementlatest(STOCK='amzn'):
    intervalSET = '1mo'

    data, highs, lows = getPointsBest(STOCK, startDate='1990-01-01', interval=intervalSET, min_=0.5)
    y = np.log10(data["close"])
    yh = np.log10(data["high"])
    yl = np.log10(data["low"])
    x = np.linspace(0, len(data)-1, len(data))

    # scaling between 0 and 1 
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
        if min__ > min_chosen:
            min__ = min_chosen
            min_idx = min_index_chosen

    for pp in pairs11:
        pairs.append((min_idx, pp[1]))

    currentPrice = yScaler.getScaledvalue(y[-1])
    print(currentPrice)
    data = data.copy()

    plotted = []
    data = data.reset_index()
    data['Date'] = data['Date'].map(mdates.date2num)

    for pp in pairs:
        if yy[pp[0]] < yy[pp[1]]:
            min_chosen = yy[pp[0]]
            max_chosen = yy[pp[1]]  
            p = (pp[0], pp[1])  
        else:
            max_chosen = yy[pp[0]]
            min_chosen = yy[pp[1]]  
            p = (pp[1], pp[0])  

        min_ = min(yy[max(0,p[0]-7):p[0]+7])
        max_ = max(yy[max(0,p[1]-7):p[1]+7])
        max_index = np.argmax(np.asarray(yy[max(0,p[0]-7):p[0]+7]))
        min_index = np.argmin(np.asarray(yy[max(0,p[0]-7):p[0.+7]))

        min_index += max(0,p[0]-7)
        max_index += max(0,p[1]-7)

        if p[0] in plotted or p[1] in plotted:
            continue
        plotted.append(p[0])
        plotted.append(p[1])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xScalerHigh.getUnscaled(), yScalerHigh.getUnscaled(), 'gray')
        ax.plot(xScalerLow.getUnscaled(), yScalerLow.getUnscaled(), 'gray')
        ax.plot(xScaler.getUnscaled(), yScaler.getUnscaled(), 'blue')
        ax.plot(xScaler.getUnscaled()[p[0]:p[1]], yScaler.getUnscaled()[p[0]:p[1]], 'red')
        ax.annotate(str(round(100*(max_-min_)/min_, 2)), (xScaler.getUnscaled()[min_index], yScaler.getUnscaled()[min_index]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax.annotate(str(round(100*(max_-min_)/min_, 2)), (xScaler.getUnscaled()[max_index], yScaler.getUnscaled()[max_index]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8)

        # Convert matplotlib figure to Plotly figure
        fig.update_layout(template='plotly_white')
        fig.update_traces(marker=dict(color='blue'))
        fig.add_trace(go.Scatter(
            x=xScaler.getUnscaled(),
            y=yScaler.getUnscaled(),
            mode='lines',
            line=dict(color='blue'),
            name='Price'
        ))
        fig.add_trace(go.Scatter(
            x=xScaler.getUnscaled()[p[0]:p[1]],
            y=yScaler.getUnscaled()[p[0]:p[1]],
            mode='lines',
            line=dict(color='red'),
            name='Retracement'
        ))
        fig.add_annotation(
            x=xScaler.getUnscaled()[min_index],
            y=yScaler.getUnscaled()[min_index],
            text=str(round(100*(max_-min_)/min_, 2)),
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=20,
            font=dict(size=8),
        )
        fig.add_annotation(
            x=xScaler.getUnscaled()[max_index],
            y=yScaler.getUnscaled()[max_index],
            text=str(round(100*(max_-min_)/min_, 2)),
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-20,
            font=dict(size=8),
        )

        fig.show()

retracementlatest()