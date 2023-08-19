import yfinance_dup as yf
import numpy as np
import pandas as pd
from finta import TA
from CorrectR import getPoints, getPointsforArray, getTuringPointsOnYear
import matplotlib.pyplot as plt
import finplot as fplt
from getPointsFile import getPointsBest, getPointsGivenR

# 1 = increasing, -1 = decreasing, 0 = flat

class Reigon:
    def __init__(self, s, e, c):
        self.start = s
        self.end = e
        self.class_ = c

def getReigons(highs, data):
    reigons = []
    i = 0
    while i+1 < len(highs):
        h1 = highs[i]
        h2 = highs[i+1]
        p1 = data[h1]
        p2 = data[h2]
        if p2 > p1 and (p2-p1)/p2 > 0.025:
            reigons.append(Reigon(h1, h2, 1))
        elif p2 < p1 and (p1-p2)/p1 > 0.025:
            reigons.append(Reigon(h1, h2, -1))
        else:
            reigons.append(Reigon(h1, h2, 0))
        i += 1
    return reigons

def getFinalReigons(reigons):
    rr = reigons.copy()
    i = 0
    while i+1 < len(rr):
        r1 = rr[i]
        r2 = rr[i+1]

        if not r1.class_ == r2.class_:
            i += 1
        else:
            rr[i].end = r2.end
            rr.remove(r2)
    return rr

# yet to test if it works for all cases 
def getOverlap(r1, r2, percent=0.3):
    s1 = r1.start
    s2 = r2.start
    e1 = r1.end
    e2 = r2.end

    if s2<=s1 and e2<=s1:
        return False
    elif s2 >= e1 and e2 >= e1:
        return False
    elif s2<s1 and e2>e1:
        return True
    elif s2>s1 and e2<e1:
        return True
    elif s1<s2 and e2>s1:
        p = (e2-s1)/(e1-s1)
        if p > percent:
            return True
        else:
            return False
    elif s2<e1 and e2>e1:
        p = (e1-s2)/(e1-s1)
        if p > percent:
            return True
        else:
            return False

def getDivergance_LL_HL(r, rS):
    divs = []
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc!=dc:
                    divs.append((max(rrs.start, rr.start), min(rrs.end, rr.end)))

                # if sc == -1 or sc == 0:
                #     if dc == 1:
                #         divs.append((max(rrs.start, rr.start), min(rrs.end, rr.end)))
    return divs

def getDivergance_HH_LH(r, rS):
    divs = []
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc!=dc:
                    divs.append((max(rrs.start, rr.start), min(rrs.end, rr.end)))

                # if sc == 1 or sc == 0:
                #     if dc == -1:
                #         divs.append((max(rrs.start, rr.start), min(rrs.end, rr.end)))
    return divs

global df
global df1

def update_legend_text(x, y):
    global df
    global df1
    df = df.reset_index()
    # dfd = dfd.reset_index()

    row = df.loc[df.Date==x]
    # format html with the candle and set legend
    fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.open<row.close).all() else 'a00')
    rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
    hover_label.setText(rawtxt % (symbol, interval.upper(), row.open, row.close, row.high, row.low))

def update_crosshair_text(x, y, xtext, ytext):
    global df
    global df1
    ytext = '%s \n C: %.2f\n O: %.2f\n H: %.2f\n L: %.2f\n ' % (ytext, (df.iloc[x].close), (df.iloc[x].open), (df.iloc[x].high), (df.iloc[x].low))
    return xtext, ytext


# print('started')


def runDiverganceGeneral2(STOCK, STOCK1, StartDate_='2000-01-01', endDate_='2100-01-01'):
    global df
    global df1

    # data, highs, lows = getTuringPointsOnYear(STOCK, TpsPerYear, getData=True, startDate=StartDate_)
    # data1, highs1, lows1 = getTuringPointsOnYear(STOCK1, TpsPerYear, getData=True, startDate=StartDate_)

    data, highs, lows = getPointsBest(STOCK, startDate=StartDate_, endDate=endDate_, max_=0.2)
    data1, highs1, lows1 = getPointsBest(STOCK1, startDate=StartDate_, endDate=endDate_, max_=0.2)

    # print(data.info())
    # print(highs, lows)
    # plt.plot(np.asarray(data["close"]),'-o', markevery=highs+lows, markersize=5, fillstyle='none')
    # plt.yscale('log')
    # plt.show()


    highsStoch, lowsStoch = highs1, lows1


    rr = getReigons(lows, data['close'])

    for r in rr:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    fr = getFinalReigons(rr)

    for r in fr:
        print(r.start, r.end, r.class_)


    rrS = getReigons(lowsStoch, data1["close"])

    for r in rrS:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    frS = getFinalReigons(rrS)

    for r in frS:
        print(r.start, r.end, r.class_)



    #####################

    rr1 = getReigons(highs, data['close'])

    for r in rr1:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    fr1 = getFinalReigons(rr1)

    for r in fr1:
        print(r.start, r.end, r.class_)


    rrS1 = getReigons(highsStoch, data1["close"])

    for r in rrS1:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    frS1 = getFinalReigons(rrS1)

    for r in frS1:
        print(r.start, r.end, r.class_)




    type1 = getDivergance_LL_HL(fr, frS)
    type2 = getDivergance_HH_LH(fr1, frS1)


    print('\n\nLL Stock, HL D\n')
    print(type1)
    print('\nHH Stock, LH D\n')
    print(type2)


    print(highsStoch)

    df = data
    df1 = data1

    try:
        ax, ax1 = fplt.create_plot(yscale='log', title=STOCK, rows=2, top_graph_scale=1)
    except:
        ax, ax1 = fplt.create_plot(yscale='log', title=STOCK, rows=2)
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=ax)

    # ax1.scale = 1

    # axo = ax.overlay(scale=0.2)
    # fplt.volume_ocv(df[['open','close','volume']], ax=axo)
    # hover_label = fplt.add_legend('',ax=ax)
    # fplt.plot(D, ax=ax1)
    fplt.candlestick_ochl(df1[['open', 'close', 'high', 'low']], ax=ax1)
    # fplt.plot(K, ax=ax1)

    # fplt.plot(df['close'].rolling(10).mean(), legend='ma-10W', ax=ax)
    # fplt.plot(df['close'].rolling(40).mean(), legend='ma-40W', ax=ax)



    fplt.set_y_scale('log', ax=ax1)

    # fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.add_crosshair_info(update_crosshair_text, ax=ax1)
    # fplt.add_band(20, 80, ax=ax1)

    ax.set_visible(crosshair=True, xaxis=True, yaxis=True, xgrid=True, ygrid=True)
    ax1.set_visible(crosshair=True, xaxis=True, yaxis=True, xgrid=True, ygrid=True)
    # fplt.volume_ocv(df['Volume'], ax=ax.overlay())

    for t in type1:
        s, e = t[0], t[1]
        stockS = data.iloc[t[0]].low
        stockE = data.iloc[t[1]].low

        Ds, De = data1.iloc[t[0]].low, data1.iloc[t[1]].low

        StockM = (stockE - stockS)/(e-s)
        Dm = (De - Ds)/(e-s)

        if StockM > 0 and Dm > 0:
            continue
            type1.remove(t)
        elif StockM < 0 and Dm < 0:
            continue
            type1.remove(t)
        else:
            fplt.add_line((t[0], data1.iloc[t[0]].low), (t[1],data1.iloc[t[1]].low), ax=ax1, width=3)
            fplt.add_line((t[0], data.iloc[t[0]].low), (t[1],data.iloc[t[1]].low), ax=ax, width=3)

    for t in type2:
        s, e = t[0], t[1]
        stockS = data.iloc[t[0]].high
        stockE = data.iloc[t[1]].high

        Ds, De = data1.iloc[t[0]].high, data1.iloc[t[1]].high

        StockM = (stockE - stockS)/(e-s)
        Dm = (De - Ds)/(e-s)

        if StockM > 0 and Dm > 0:
            continue
            type2.remove(t)
        elif StockM < 0 and Dm < 0:
            continue
            type2.remove(t)
        else:
            fplt.add_line((t[0], data1.iloc[t[0]].high), (t[1],data1.iloc[t[1]].high), ax=ax1, width=3)
            fplt.add_line((t[0], data.iloc[t[0]].high), (t[1],data.iloc[t[1]].high), ax=ax, width=3)


    # ax1.se
    fplt.show()

if __name__ == "__main__":
    STOCK =  'djia'
    STOCK1 = '^ndx'

    # StartDate = '2021-03-10'
    StartDate_ = '2022-09-01'
    RStart = 1.2
    endDate_ = '2023-04-30'

    TpsPerYear = 20

    runDiverganceGeneral2(STOCK1, STOCK, StartDate_, endDate_)
