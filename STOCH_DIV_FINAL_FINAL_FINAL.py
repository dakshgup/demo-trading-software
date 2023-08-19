from turtle import st
import yfinance as yf
import numpy as np
import pandas as pd
from finta import TA
from CorrectR import getPoints, getPointsforArray
import matplotlib.pyplot as plt
import finplot as fplt
from getPointsFile import getPointsBest, getPointsGivenR


# 1 = increasing, -1 = decreasing, 0 = flat

class Reigon:
    def __init__(self, s, e, c):
        self.start = s
        self.end = e
        self.class_ = c

def getReigons(highs, data, stoch=False):
    reigons = []
    if stoch:
        i = 15
    else:
        i = 0

    while i+1 < len(highs):
        h1 = highs[i]
        h2 = highs[i+1]
        p1 = data[h1]
        p2 = data[h2]
        if p2 > p1 and (p2-p1)/p2 > 0.025: #.05
            reigons.append(Reigon(h1, h2, 1))
        elif p2 < p1 and (p1-p2)/p1 > 0.025: #.05
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

                if sc!= dc:
                    divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))

                # if sc == -1 or sc == 0:
                #     if dc == 1:
                #         if not rr.start == rr.end and not rrs.start == rrs.end:
                #             divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs

def getDivergance_HH_LH(r, rS):
    divs = []
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc != dc:
                    divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
                # if sc == 1 or sc == 0:
                #     if dc == -1:
                #         if not rr.start == rr.end and not rrs.start == rrs.end:
                #             divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs

global df

def update_legend_text(x, y):
    global df
    df = df.reset_index()
    # dfd = dfd.reset_index()

    row = df.loc[df.Date==x]
    # format html with the candle and set legend
    fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.open<row.close).all() else 'a00')
    rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
    hover_label.setText(rawtxt % (symbol, interval.upper(), row.open, row.close, row.high, row.low))

def update_crosshair_text(x, y, xtext, ytext):
    global df
    ytext = '%s \n C: %.2f\n O: %.2f\n H: %.2f\n L: %.2f\n ' % (ytext, (df.iloc[x].close), (df.iloc[x].open), (df.iloc[x].high), (df.iloc[x].low))
    return xtext, ytext

def getClosestPrevIndex(start, data, type):
    min_ = 10000000000
    selected = None
    if type == 'start':
        for d in data:
            if start - d > 0:
                if start - d < min_:
                    min_ = start - d
                    selected = d
        return selected
    else:
        for d in data:
            if start - d < 0:
                if -start + d < min_:
                    min_ = -start + d
                    selected = d
        return selected

def runStochDivergance(STOCK, startDate_='2000-01-01', endDate_='2022-08-07', R_=None):
    global df

    if R_ == None:
        data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, interval='1d',getR=True, min_=0.15, max_=0.25) 
    else:
        R = R_
        data, _, _ = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_)
        _, lows = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_, type_='lows')
        _, highs = getPointsGivenR(STOCK, R, startDate=startDate_, endDate=endDate_, type_='highs')

        _, lowsSensitive = getPointsGivenR(STOCK, 1.02, startDate=startDate_, endDate=endDate_, type_='lows')
        _, highsSensitive = getPointsGivenR(STOCK, 1.02, startDate=startDate_, endDate=endDate_, type_='highs')
        highs.append(highsSensitive[-1])
        highs.append(highsSensitive[-2])
        highs.append(highsSensitive[-3])
        lows.append(lowsSensitive[-1])
        lows.append(lowsSensitive[-2])
        lows.append(lowsSensitive[-3])
    lows.append(len(data)-1)
#############################
    highs.append(len(data)-1)
###############################

    lows = np.asarray(lows)
    lows -= 15
    lows = lows[lows >= 0]
    lows = lows.tolist()

    highs = np.asarray(highs)
    highs -= 15
    highs = highs[highs >= 0]
    highs = highs.tolist()

    # print(data.info())
    # print(highs, lows)
    plt.plot(np.asarray(data["low"]),'-o', markevery=lows, markersize=5, fillstyle='none')
    plt.yscale('log')
    plt.show()

    plt.plot(np.asarray(data["high"]),'-o', markevery=highs, markersize=5, fillstyle='none')
    plt.yscale('log')
    plt.show()

    ################################### CHANGE THIS FOR MULTIPLE INDICATORS
    K = TA.STOCH(data, 14)
    D = TA.STOCHD(data)
    data = data[15:]
    D = D[15:]
    print(D.iloc[0])

    x = D.to_numpy()
    print(x[0])
    highsStoch, lowsStoch = getPointsforArray(x, 1.05)
    highsStoch.append(len(D)-1)

    plt.plot(x,'-o', markevery=lowsStoch+highsStoch)
    plt.show()


    rr = getReigons(lows, data['low'])

    for r in rr:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    fr = getFinalReigons(rr)

    for r in fr:
        print(r.start, r.end, r.class_)


    rrS = getReigons(lowsStoch, D, stoch =True)

    for r in rrS:
        print(r.start, r.end, r.class_)

    # print("hejrgsfbhjkds")
    # frS = getFinalReigons(rrS)

    # for r in frS:
        # print(r.start, r.end, r.class_)


    rr1 = getReigons(highs, data['high'])

    for r in rr1:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    fr1 = getFinalReigons(rr1)

    for r in fr1:
        print(r.start, r.end, r.class_)

    # for h in highsStoch:
    #     h += 26

    rrS1 = getReigons(highsStoch, D)

    print("hejrgsfbh")
    frS1 = getFinalReigons(rrS1)

    # for l in lowsStoch:
    #     l += 26

    rrS1 = getReigons(lowsStoch, D)

    for r in rrS1:
        print(r.start, r.end, r.class_)

    print("hejrgsfbhjkds")
    frS2 = getFinalReigons(rrS1)

    for r in frS1:
        print(r.start, r.end, r.class_)

    type1 = getDivergance_LL_HL(fr, frS2)
    type2 = getDivergance_HH_LH(fr1, frS1)


    print('\n\nLL Stock, HL D\n')
    print(type1)
    print('\nHH Stock, LH D\n')
    print(type2)

    ma = TA.WMA(data, 14)

    df = data

    ax, ax1 = fplt.create_plot(yscale='log', title=STOCK, rows=2)
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']])


    axo = ax.overlay(scale=0.2)
    fplt.volume_ocv(df[['open','close','volume']], ax=axo)
    # hover_label = fplt.add_legend('',ax=ax)
    fplt.plot(D, ax=ax1)
    # fplt.plot(K, ax=ax1)

    fplt.plot(df['close'].rolling(10).mean(), legend='ma-10W', ax=ax)
    fplt.plot(df['close'].rolling(40).mean(), legend='ma-40W', ax=ax)

    fplt.set_y_scale('linear', ax=ax1)

    # fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.add_band(20, 80, ax=ax1)

    # ax.set_visible(crosshair=True, xaxis=True, yaxis=True, xgrid=True, ygrid=True)
    # fplt.volume_ocv(df['Volume'], ax=ax.overlay())

    for t in type1:
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        stockS = data.iloc[t[0][0]].high
        stockE = data.iloc[t[1][0]].high

        Ds, De = D.iloc[t[0][1]], D.iloc[t[1][1]]

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS)/(eS-sS)
            Dm = (eD - sS)/(eD-sD)
        
            print("YYOY")
##########################################################################
            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending

                # fplt.add_line((t[0][1], D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=7, color='r')
                # fplt.add_line((t[0][0], data.iloc[t[0][0]].low), (t[1][0],data.iloc[t[1][0]].low), ax=ax, width=7, color='r')
                
                fplt.add_line((dStart, D.iloc[dStart]), (dEnd,D.iloc[dEnd]), ax=ax1, width=7)
                fplt.add_line((stockStart, data.iloc[stockStart].low), (stockEnd,data.iloc[stockEnd].low), ax=ax, width=7)


    for t in type2:
        print(t)
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        ss = max(sS, sD)
        ee = min(eS, eD)
        stockS = data.iloc[ss].high
        stockE = data.iloc[ee].high
        dds = D.iloc[ss]
        dde = D.iloc[ee]

        Ds, De = D.iloc[t[0][1]], D.iloc[t[1][1]]

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS)/(eS-sS)
            Dm = (dde - dds)/(eS-sS)

            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                # fplt.add_line((t[0][1], D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=7, color='r')
                # fplt.add_line((t[0][0], data.iloc[t[0][0]].high), (t[1][0],data.iloc[t[1][0]].high), ax=ax, width=7, color='r')
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending

                # fplt.add_line((t[0][1], D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=7, color='g')
                # fplt.add_line((t[0][0], data.iloc[t[0][0]].high), (t[1][0],data.iloc[t[1][0]].high), ax=ax, width=7, color='g')
            
                fplt.add_line((dStart, D.iloc[dStart]), (dEnd,D.iloc[dEnd]), ax=ax1, width=7)
                fplt.add_line((stockStart, data.iloc[stockStart].high), (stockEnd,data.iloc[stockEnd].high), ax=ax, width=7)


    # for r in fr:
    #     # fplt.add_line((t[0, D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=3)
    #     # if r.class_ == 1:
    #     #     fplt.add_line((r.start, data.iloc[r.start].low), (r.end,data.iloc[r.end].low), ax=ax, width=3, color='r')
    #     # elif r.class_ == -1:
    #     #     fplt.add_line((r.start, data.iloc[r.start].low), (r.end,data.iloc[r.end].low), ax=ax, width=3, color='g')
    #     # else:
    #     fplt.add_line((r.start, data.iloc[r.start].low), (r.end,data.iloc[r.end].low), ax=ax, width=3)
    # for r in fr1:
    #     # # fplt.add_line((t[0, D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=3)
    #     # if r.class_ == 1:
    #     #     fplt.add_line((r.start, data.iloc[r.start].high), (r.end,data.iloc[r.end].high), ax=ax, width=3, color='r')
    #     # elif r.class_ == -1:
    #     #     fplt.add_line((r.start, data.iloc[r.start].high), (r.end,data.iloc[r.end].high), ax=ax, width=3, color='g')
    #     # else:
    #     fplt.add_line((r.start, data.iloc[r.start].high), (r.end,data.iloc[r.end].high), ax=ax, width=3)
    # for h in highs:
    #     fplt.add_point((h, data.iloc[h].high), ax=ax, width=5, color='b')
    # for l in lows:
    #     fplt.add_point((l, data.iloc[l].low-1), ax=ax, width=5, color='y')
    # for r in frS1:
    #     # fplt.add_line((t[0, D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=3)
    #     # if r.class_ == 1:
    #     #     fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3, color='r')
    #     # elif r.class_ == -1:
    #     #     fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3, color='g')
    #     # else:
    #     fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3)
    # for r in frS2:
    #     # fplt.add_line((t[0, D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=3)
    #     # fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3)
    #     # if r.class_ == 1:
    #     #     fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3, color='r')
    #     # elif r.class_ == -1:
    #     #     fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3, color='g')
    #     # else:
    #     fplt.add_line((r.start, D.iloc[r.start]), (r.end,D.iloc[r.end]), ax=ax1, width=3)
    # for h in highsStoch:
    #     fplt.add_point((h, D.iloc[h]), ax=ax1, width=5, color='b')
    # for l in lowsStoch:
    #     fplt.add_point((l, D.iloc[l]), ax=ax1, width=5, color='y')
    fplt.show()
# Maybe to 1.05 and 1.03 and combine both the results
if __name__ == "__main__":
    runStochDivergance('eth-usd', '2022-09-01', endDate_='2023-04-30')
