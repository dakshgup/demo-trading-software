import yfinance_dup as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finplot_mod import _roihandle_move_snap
from getPointsFile import getPointsBest
import math
from scipy.optimize import minimize
# from thetaTrendlines import run_
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import finplot_mod as fplt
from matplotlib import style
import matplotlib.dates as mdates
from CorrectR import getPoints
from matplotlib.ticker import ScalarFormatter
from mpl_finance import candlestick_ohlc
from matplotlib import ticker
from Utilities import Scaler


class Line:
    def __init__(self, x1, y1, x2, y2, type=None) :
        self.xMin = x1
        self.xMax = x2
        self.y1 = y1
        self.y2 = y2

        self.m = (y2-y1)/(x2-x1)
        self.c = y1 - self.m*x1

        self.type=type

        self.length = ((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))**(1/2)
    
    def inRange(self, x):
        return self.xMin <= x <= self.xMax

    def getY(self, x, ignoreRange=False):
        if ignoreRange:
            return self.m*x + self.c
        else:
            if self.inRange(x):
                return self.m*x + self.c
            else:
                return None
            

class Scorer:
    def __init__(self, data, lows, dataOrignal, type) -> None:
        self.data = data
        if type == 'low':
            self.actualData = 10 ** data["low"]
        elif type == 'high':
            self.actualData = 10 ** data["high"]
        else:
            self.actualData = 10 ** data["close"]
        self.lowsUnscaled = lows
        self.dataOrignal = dataOrignal
        self.type = type

        xxx = np.linspace(0, len(data)-1, len(data))
        if type == 'low':
            self.yScalar = Scaler(data["low"])
        elif type == 'high':
            self.yScalar = Scaler(data["high"])
        self.xScalar = Scaler(xxx)

        self.yScalarHighs = Scaler(data["high"])

        self.yScaled = self.yScalar.getScaled()
        self.xScaled = self.xScalar.getScaled()

        self.lowsScaled = self.xScalar.getScaledArray(lows)

        self.Points = self.getPoints(self.lowsScaled)

    def getPoints(self, lows):
        P = []
        for l in lows:
            P.append((l, self.Y(l)))
        return P

    def Y(self, x):
        i, = np.where(np.isclose(self.xScaled, x))
        return self.yScaled[i][0]

    def getArea(self, line: Line):
        a = line.xMin
        b = line.xMax
        dx = self.xScaled[2] - self.xScaled[1]
        S_plus = 0
        S_minus = 0
        i = a
        while i <= b:
            yA = self.Y(i)
            yP = line.getY(i)
            d = yA - yP
            if d > 0:
                # if self.type == "low":
                if d < 0.1:
                    S_plus += d*dx
                elif d > 0.2:
                    S_plus -= d*dx 
                # else:
                #     S_plus += d*dx
            else:
                # if self.type=='high':
                #     if abs(d) < 0.1:
                #         S_minus += d*dx
                #     else:
                #         S_minus -= d*dx
                # else:
                S_minus += d*dx
            i += dx
        return S_plus, S_minus

    def getIfClose(self, p, line:Line, e=0.05):
        yP = line.getY(p[0])
        yA = p[1]
        if not yP is None:
            d = ((yP-yA)*(yP-yA))**(1/2)
            return d <= e
        else:
            return False

    def getCloseTPsum(self, line:Line, e=0.05):
        s = 0
        for p in self.Points:
            if self.getIfClose(p, line, e):
                s += 1
        return s-2

    # self, line:Line, alpha_plus=100, alpha_minus=10000, beta=2, gamma=2
    # self, line:Line, alpha_plus=100, alpha_minus=10000, beta=20, gamma=2 # best till now
    def getScore(self, line:Line, alpha_plus=100, alpha_minus=100000, beta=5, gamma=5):
            # self, line:Line, alpha_plus=10, alpha_minus=1000, beta=2, gamma=2
            # self, line:Line, alpha_plus=100, alpha_minus=100000, beta=2, gamma=2 lasr

            # self, line:Line, alpha_plus=100, alpha_minus=10000, beta=20, gamma=5 current on may 11
        
        area_plus, area_minus = self.getArea(line)
        length = line.length
        phi = self.getCloseTPsum(line)
        score = alpha_plus*area_plus + alpha_minus*area_minus + beta*length + gamma*phi
        return score

    def plot(self, showLows=True):
        if showLows:
            plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled)
            plt.show()
        else:
            plt.plot(self.xScaled, self.yScaled)
            plt.show()

    def plotLine(self, line:Line, showLows=True, tillEnd=False):
        if not tillEnd:
            if showLows:
                plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled)
                plt.plot([line.xMin, line.xMax], [line.y1, line.y2])
                plt.show()
            else:
                plt.plot(self.xScaled, self.yScaled)
                plt.plot([line.xMin, line.xMax], [line.y1, line.y2])
                plt.show()
        else:
            if showLows:
                plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled)
                plt.plot([line.xMin, 1], [line.y1, line.getY(1, ignoreRange=True)])
                plt.show()
            else:
                plt.plot(self.xScaled, self.yScaled)
                plt.plot([line.xMin, 1], [line.y1, line.getY(line.getY(1), ignoreRange=True)])
                plt.show()

    def plotMultipleLines(self, lines, showLows=True, tillEnd=False, showHighs=False, highs=None):
        if tillEnd:
            for line in lines:
                plt.plot([line.xMin, 1], [line.y1, line.getY(1, ignoreRange=True)])
        else:
            for line in lines:
                plt.plot([line.xMin, line.xMax], [line.y1, line.y2])
        
        # if showLows:
        #     plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled)
        # else:
        #     plt.plot(self.xScaled, self.yScaled)

        if showLows and showHighs:
            plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled+highs)
        elif showLows:
            plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled)
        else:
            plt.plot(self.xScaled, self.yScaled)

        ax = plt.gca()
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        plt.show()

    def plotMultipleLinesUnscaled(self, lines, showLows=True, tillEnd=False, showHighs=False, highs=None):
        if tillEnd:
            print(self.xScalar.getUnscaledValue(1), "rekusekdbfeqbfuebfuebfnusfkwnfekdf")
            for line in lines:
                # dateDate = self.data.iloc[int(self.xScalar.getUnscaledValue(line.xMin))].name
                xx = np.linspace(self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1))
                xxx = np.linspace(line.xMin, 1)
                yy = []
                for x in xxx:
                    yy.append(10**self.yScalar.getUnscaledValue(line.getY(x, ignoreRange=True)))

                # plt.plot([self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1)], [10**self.yScalar.getUnscaledValue(line.y1), 10**self.yScalar.getUnscaledValue(line.getY(1, ignoreRange=True))])
                plt.plot(xx, yy)
        else:
            for line in lines:
                dateDate = self.data[self.xScalar.getUnscaledValue(line.xMin)]
                xx = np.linspace(self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(line.xMax))
                xxx = np.linspace(line.xMin, line.xMax)
                yy = []
                for x in xxx:
                    yy.append(10**self.yScalar.getUnscaledValue(line.getY(x, ignoreRange=True)))

                # plt.plot([self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1)], [10**self.yScalar.getUnscaledValue(line.y1), 10**self.yScalar.getUnscaledValue(line.getY(1, ignoreRange=True))])
                plt.plot(xx, yy)
        
        # if showLows:
        #     plt.plot(self.xScaled, self.yScaled, '-o', markevery=self.lowsUnscaled)
        # else:
        #     plt.plot(self.xScaled, self.yScaled)
        data = self.dataOrignal.copy()      
        data = data.reset_index()
        data['Date'] = data['Date'].map(mdates.date2num)

        if showLows and showHighs:
            plt.plot(self.xScalar.getUnscaled(), 10**self.yScalar.getUnscaled(), '-o', markevery=self.lowsUnscaled+highs)
        elif showLows:
            plt.plot(self.xScalar.getUnscaled(), 10**self.yScalar.getUnscaled(), '-o', markevery=self.lowsUnscaled)
        else:
            plt.plot(self.xScalar.getUnscaled(), 10**self.yScalar.getUnscaled())

        ax = plt.gca()
        # ax.set_xlim([-0.1, 1.1])
        ax.set_yscale('log')
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        yy = min(self.data)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        print(10**min(self.data["close"])-10, "eirushfirwnwefjhqbrev")
        ax.set_ylim([min(10**min(self.data["close"]), 1), 10**max(self.data["close"])+10])
        ax.xaxis.axis_date()
        # plt.xticks(data['Date'].dt.strftime('%b-%d-%y'))
        plt.show()

    def plotFinal(self, lines, showLows=True, tillEnd=False, showHighs=False, highs=None):
        # fplt.candlestick_ochl(self.dataOrignal[['open', 'close', 'high', 'low']])
        # for line in lines:

        # fplt.show()
        data = self.dataOrignal.copy()

        # df_volume = data['volume']

        data = data.reset_index()
        data['Date'] = data['Date'].map(mdates.date2num)
        # data = data.drop(['Adj Close', 'volume'], axis=1)
        # print(data.head())

        tickerEvery = (max(data["high"]) - min(data["low"]))/10
        # fig.tight_layout()
        ax1 = plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)
        # ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
        ax1.xaxis_date()
        ax1.set_yscale('log')
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(tickerEvery))  # major y tick positions every 100
        
        

        # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))  # major x tick positions every 7
        # ax1.xaxis.set_major_formatter(ticker.)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        if not intervalSET == '1mo':
            ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3,6,9)))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        for label in ax1.get_xticklabels(which='major'):
            label.set(rotation=60)


        ax1.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting
        ax1.grid(True)

        candlestick_ohlc(ax1, data.values, width=1)

        from datetime import datetime
        date_format = "%m/%d/%Y"
        a = datetime.strptime('8/18/2008', date_format)
        b = datetime.strptime('9/26/2008', date_format)
        delta = b - a
        print(delta.days)

        firstDate = data['Date'][0]
        print(firstDate)

        for line in lines:
            if line.type == 'high':
                if intervalSET == '1wk':
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7)+firstDate, (self.xScalar.getUnscaledValue(1)*7)+firstDate)
                elif intervalSET == '1d':
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*365/253)+firstDate, (self.xScalar.getUnscaledValue(1)*365/253)+firstDate)
                else:
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*30.5)+firstDate, (self.xScalar.getUnscaledValue(1)*30.5)+firstDate)
                xxx = np.linspace(line.xMin, 1)
                yy = []
                for x in xxx:
                    yy.append(10**self.yScalarHighs.getUnscaledValue(line.getY(x, ignoreRange=True)))

                    # plt.plot([self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1)], [10**self.yScalar.getUnscaledValue(line.y1), 10**self.yScalar.getUnscaledValue(line.getY(1, ignoreRange=True))])
                ax1.plot(xx, yy)
            else:
                # xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7)+firstDate, (self.xScalar.getUnscaledValue(1)*7)+firstDate)
                if intervalSET == '1wk':
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7)+firstDate, (self.xScalar.getUnscaledValue(1)*7)+firstDate)
                elif intervalSET == '1d':
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7/5)+firstDate, (self.xScalar.getUnscaledValue(1)*7/5)+firstDate)
                else:
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*30.5)+firstDate, (self.xScalar.getUnscaledValue(1)*30.5)+firstDate)
                xxx = np.linspace(line.xMin, 1)
                yy = []
                for x in xxx:
                    yy.append(10**self.yScalar.getUnscaledValue(line.getY(x, ignoreRange=True)))

                    # plt.plot([self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1)], [10**self.yScalar.getUnscaledValue(line.y1), 10**self.yScalar.getUnscaledValue(line.getY(1, ignoreRange=True))])
                ax1.plot(xx, yy)

        # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

        # plt.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.05, wspace=0.2, hspace=0.37)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        ax1.set_ylim([0.8*min(data["low"]), 1.1*max(data["high"])])
        plt.show()

    def plotFinalWithDotted(self, lines, showLows=True, tillEnd=False, showHighs=False, highs=None):
        # fplt.candlestick_ochl(self.dataOrignal[['open', 'close', 'high', 'low']])
        # for line in lines:

        # fplt.show()
        data = self.dataOrignal.copy()

        # df_volume = data['volume']

        data = data.reset_index()
        data['Date'] = data['Date'].map(mdates.date2num)
        # data = data.drop(['Adj Close', 'volume'], axis=1)
        # print(data.head())

        tickerEvery = (max(data["high"]) - min(data["low"]))/10
        # fig.tight_layout()
        ax1 = plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)
        # ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
        ax1.xaxis_date()
        ax1.set_yscale('log')
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(tickerEvery))  # major y tick positions every 100
        
        

        # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))  # major x tick positions every 7
        # ax1.xaxis.set_major_formatter(ticker.)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
        if not intervalSET == '1mo':
            ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3,6,9)))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        for label in ax1.get_xticklabels(which='major'):
            label.set(rotation=60)


        ax1.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting
        ax1.grid(True)

        candlestick_ohlc(ax1, data.values, width=1)

        from datetime import datetime
        date_format = "%m/%d/%Y"
        a = datetime.strptime('8/18/2008', date_format)
        b = datetime.strptime('9/26/2008', date_format)
        delta = b - a
        print(delta.days)

        firstDate = data['Date'][0]
        print(firstDate)

        cols = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        i = 0
        for line in lines:
            if line.type == 'high':
                if intervalSET == '1wk':
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7)+firstDate, (self.xScalar.getUnscaledValue(line.xMax)*7)+firstDate, 50)
                    xx1 = np.linspace((self.xScalar.getUnscaledValue(line.xMax)*7)+firstDate, (self.xScalar.getUnscaledValue(1)*7)+firstDate, 50)
                elif intervalSET == '1d':
                    xx1 = np.linspace((self.xScalar.getUnscaledValue(line.xMax)*365/253)+firstDate, (self.xScalar.getUnscaledValue(1)*365/253)+firstDate, 50)
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*365/253)+firstDate, (self.xScalar.getUnscaledValue(line.xMax)*365/253)+firstDate, 50)
                else:
                    xx1 = np.linspace((self.xScalar.getUnscaledValue(line.xMax)*30.5)+firstDate, (self.xScalar.getUnscaledValue(1)*30.5)+firstDate, 50)
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*30.5)+firstDate, (self.xScalar.getUnscaledValue(line.xMax)*30.5)+firstDate, 50)
                xxx = np.linspace(line.xMin, line.xMax, 50)
                xxx1 = np.linspace(line.xMax, 1, 50)
                yy = []
                yy1 = []
                for x in xxx:
                    yy.append(10**self.yScalarHighs.getUnscaledValue(line.getY(x, ignoreRange=True)))
                for x in xxx1:
                    yy1.append(10**self.yScalarHighs.getUnscaledValue(line.getY(x, ignoreRange=True)))
                    # plt.plot([self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1)], [10**self.yScalar.getUnscaledValue(line.y1), 10**self.yScalar.getUnscaledValue(line.getY(1, ignoreRange=True))])
                ax1.plot(xx, yy, color=cols[i])
                ax1.plot(xx1, yy1, '--', color=cols[i])
                i += 1
                if i >= len(cols):
                    i = 0
            else:
                # xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7)+firstDate, (self.xScalar.getUnscaledValue(1)*7)+firstDate)
                if intervalSET == '1wk':
                    xx1 = np.linspace((self.xScalar.getUnscaledValue(line.xMax)*7)+firstDate, (self.xScalar.getUnscaledValue(1)*7)+firstDate, 50)
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*7)+firstDate, (self.xScalar.getUnscaledValue(line.xMax)*7)+firstDate, 50)
                elif intervalSET == '1d':
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*365/253)+firstDate, (self.xScalar.getUnscaledValue(line.xMax)*365/253)+firstDate, 50)
                    xx1 = np.linspace((self.xScalar.getUnscaledValue(line.xMax)*365/253)+firstDate, (self.xScalar.getUnscaledValue(1)*365/253)+firstDate, 50)
                else:
                    xx = np.linspace((self.xScalar.getUnscaledValue(line.xMin)*30.5)+firstDate, (self.xScalar.getUnscaledValue(line.xMax)*30.5)+firstDate, 50)
                    xx1 = np.linspace((self.xScalar.getUnscaledValue(line.xMax)*30.5)+firstDate, (self.xScalar.getUnscaledValue(1)*30.5)+firstDate, 50)
                xxx = np.linspace(line.xMin, line.xMax, 50)
                xxx1 = np.linspace(line.xMax, 1, 50)
                yy = []
                yy1 = []
                for x in xxx:
                    yy.append(10**self.yScalar.getUnscaledValue(line.getY(x, ignoreRange=True)))
                for x in xxx1:
                    yy1.append(10**self.yScalar.getUnscaledValue(line.getY(x, ignoreRange=True)))

                    # plt.plot([self.xScalar.getUnscaledValue(line.xMin), self.xScalar.getUnscaledValue(1)], [10**self.yScalar.getUnscaledValue(line.y1), 10**self.yScalar.getUnscaledValue(line.getY(1, ignoreRange=True))])
                ax1.plot(xx, yy, color=cols[i])
                ax1.plot(xx1, yy1, '--', color=cols[i])
                i += 1
                if i >= len(cols):
                    i = 0
                

        # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

        # plt.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.05, wspace=0.2, hspace=0.37)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        ax1.set_ylim([0.8*min(data["low"]), 1.1*max(data["high"])])
        plt.show()
              

def run(i, j, scorer, plot=False, alpha_plus_=100, alpha_minus_=100000, beta_=5, gamma_=5):
    if i < len(scorer.lowsScaled) and j < len(scorer.lowsScaled):
        p1 = ( float(scorer.lowsScaled[i]), float(scorer.Y(scorer.lowsScaled[i])) )
        p2 = ( float(scorer.lowsScaled[j]), float(scorer.Y(scorer.lowsScaled[j]) ))
        line = Line(p1[0], p1[1], p2[0], p2[1])
        score = scorer.getScore(line, alpha_plus=alpha_plus_, alpha_minus=alpha_minus_, beta=beta_, gamma=gamma_)
        print(score)
        if plot:
            scorer.plotLine(line)
        return score, line
    else:
        return -100000, None

def go(i, scorer, type, alpha_plus=100, alpha_minus=100000, beta=5, gamma=5):
    if type == "low":
        maxScore = -10000000000000000000
        i_s = None
        j_s = None
        # i = 4
        j = i + 1
        while j < len(lows):
            score, line = run(i, j, scorer, alpha_plus_=alpha_plus, alpha_minus_=alpha_minus, beta_=beta, gamma_=gamma)
            if score > maxScore:
                maxScore = score
                i_s = i
                j_s = j
            print(score, i, j)
            j += 1
        print(i_s, j_s)
        if not i_s is None and j_s is not None and i_s <len(scorer.lowsScaled) and j_s < len(scorer.lowsScaled):
            p1 = ( float(scorer.lowsScaled[i_s]), float(scorer.Y(scorer.lowsScaled[i_s])) )
            p2 = ( float(scorer.lowsScaled[j_s]), float(scorer.Y(scorer.lowsScaled[j_s]) ))
            line = Line(p1[0], p1[1], p2[0], p2[1])
            run(i_s, j_s, scorer, plot=False, alpha_plus_=alpha_plus, alpha_minus_=alpha_minus, beta_=beta, gamma_=gamma)
            return i_s, j_s, line, score
        else:
            return None, j, None, 0
    else:
        maxScore = -10000000000000000000
        i_s = None
        j_s = None
        # i = 4
        j = i + 1
        while j < len(highs):
            score, line = run(i, j, scorer, alpha_plus_=alpha_plus, alpha_minus_=alpha_minus, beta_=beta, gamma_=gamma)
            if score > maxScore:
                maxScore = score
                i_s = i
                j_s = j
            print(score, i, j)
            j += 1
        print(i_s, j_s)
        if not i_s is None and j_s is not None and i_s <len(scorer.lowsScaled) and j_s < len(scorer.lowsScaled):
            p1 = ( float(scorer.lowsScaled[i_s]), float(scorer.Y(scorer.lowsScaled[i_s])) )
            p2 = ( float(scorer.lowsScaled[j_s]), float(scorer.Y(scorer.lowsScaled[j_s]) ))
            line = Line(p1[0], p1[1], p2[0], p2[1])
            run(i_s, j_s, scorer, plot=False, alpha_plus_=alpha_plus, alpha_minus_=alpha_minus, beta_=beta, gamma_=gamma)
            return i_s, j_s, line, score
        else:
            return None, j, None, 0

intervalSET = '1wk'
# google apple

if __name__ == "__main__":
    data, highs, lows = getPointsBest('aapl', max_=0.25, limit=70, min_=0.15, startDate='2000-01-01', interval=intervalSET, returnData='lows')
    # data = -1*data + 500
    print(data.head())

    plt.plot(data["close"], '-o', markevery=lows)
    plt.yscale('log')
    plt.show()

    dataOrignal = data.copy(deep=True)
    data = np.log10(data)
    scorer = Scorer(data, lows, dataOrignal, 'low')

    # For shorter term
    i = 0
    lines = []
    while i < len(lows):
        _, i, line, score = go(i, scorer, 'low',  alpha_plus=100, alpha_minus=100000, beta=5, gamma=5)
        if line is not None:
            lines.append(line)

    # For longer term
    i = 0
    while i < len(lows):
        _, i, line, score = go(i, scorer, 'low',  alpha_plus=100, alpha_minus=10000, beta=20, gamma=5)
        if line is not None:
            lines.append(line)

    # scorer.plotMultipleLines(lines, tillEnd=True)


    data, highs, lows = getPointsBest('jpm', max_=0.25, limit=70, min_=0.15, startDate='2000-01-01', interval=intervalSET, returnData='highs')
    # data = -1*data + 500
    print(data.head())

    dataOrignal = data.copy(deep=True)
    data = np.log10(data)

    data2 = -1*data
    scorerHighs = Scorer(data2, highs, dataOrignal, 'high')

    # For shorter term
    i = 0
    linesH = []
    while i < len(highs):
        _, i, line, score = go(i, scorerHighs, 'high', alpha_plus=100, alpha_minus=100000, beta=5, gamma=5)
        if line is not None:
            linesH.append(line)
        # i += 1
    # _, i, line = go(0, scorer)
    for line in linesH:
        x1 = line.xMin
        x2 = line.xMax
        y1p = line.y1
        y2p = line.y2
        y1 = (y1p * -1)+1
        y2 = (y2p* -1)+1
        # yy1 = (y1 - min(data["close"])) / (max(data["close"]) - min(data["close"]))
        # yy2 = (y2- min(data["close"])) / (max(data["close"]) - min(data["close"]))
        ll = Line(x1, y1, x2, y2, 'high')
        lines.append(ll)

    # For longer term
    i = 0
    while i < len(highs):
        _, i, line, score = go(i, scorerHighs, 'high', alpha_plus=100, alpha_minus=10000, beta=20, gamma=5)
        if line is not None:
            linesH.append(line)
        # i += 1
    # _, i, line = go(0, scorer)
    for line in linesH:
        x1 = line.xMin
        x2 = line.xMax
        y1p = line.y1
        y2p = line.y2
        y1 = (y1p * -1)+1
        y2 = (y2p* -1)+1
        # yy1 = (y1 - min(data["close"])) / (max(data["close"]) - min(data["close"]))
        # yy2 = (y2- min(data["close"])) / (max(data["close"]) - min(data["close"]))
        ll = Line(x1, y1, x2, y2, 'high')
        lines.append(ll)

    # scorer.plotMultipleLinesUnscaled(lines, tillEnd=True, showHighs=True, highs=highs)
    scorer.plotFinalWithDotted(lines, tillEnd=True, showHighs=True, highs=highs)
    # scorer.plotMultipleLines(lines, tillEnd=False, showHighs=True, highs=highs)
