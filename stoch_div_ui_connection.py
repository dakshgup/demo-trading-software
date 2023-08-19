from turtle import st
import yfinance_dup as yf
import numpy as np
import pandas as pd
from finta import TA
from CorrectR import getPoints, getPointsforArray
import matplotlib.pyplot as plt
# import finplot_mod as fplt
from getPointsFile import getPointsBest, getPointsGivenR
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def getDivergance_LL_HL(r, rS):
    divs = []
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc == -1 or sc == 0:
                    if dc == 1:
                        if not rr.start == rr.end and not rrs.start == rrs.end:
                            divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs

def getDivergance_HH_LH(r, rS):
    divs = []
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc == 1 or sc == 0:
                    if dc == -1:
                        if not rr.start == rr.end and not rrs.start == rrs.end:
                            divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs

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

global df

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
        data, highs, lows, R = getPointsBest(STOCK, startDate=startDate_, endDate=endDate_, getR=True, min_=0.15, max_=0.25) 
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

    ################################### CHANGE THIS FOR MULTIPLE INDICATORS
    K = TA.STOCH(data, 14)
    D = TA.STOCHD(data)
    data = data[15:]
    D = D[15:]
    # print(D.iloc[0])

    x = D.to_numpy()
    # print(x[0])
    highsStoch, lowsStoch = getPointsforArray(x, 1.05)
    highsStoch.append(len(D)-1)

    # plt.plot(x,'-o', markevery=lowsStoch+highsStoch)
    # plt.show()


    rr = getReigons(lows, data['low'])

    # for r in rr:
    #     print(r.start, r.end, r.class_)

    # print("hejrgsfbhjkds")
    fr = getFinalReigons(rr)

    # for r in fr:
    #     print(r.start, r.end, r.class_)


    rrS = getReigons(lowsStoch, D, stoch =True)



    rr1 = getReigons(highs, data['high'])
    fr1 = getFinalReigons(rr1)


    rrS1 = getReigons(highsStoch, D)

    # print("hejrgsfbh")
    frS1 = getFinalReigons(rrS1)

    rrS1 = getReigons(lowsStoch, D)

    frS2 = getFinalReigons(rrS1)


    type1 = getDivergance_LL_HL(fr, frS2)
    type2 = getDivergance_HH_LH(fr1, frS1)


    ma = TA.WMA(data, 14)

    df = data

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.01,subplot_titles=('Stock prices','Stochastic Indicator'),row_width=[0.29,0.7])
    fig.update_yaxes(type='log',row = 1,col = 1)
    fig.add_trace(go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close']),row=1,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)


    fig.add_trace(go.Scatter(x=D.index,y=D),row=2,col=1)

    fig.add_trace(go.Scatter(x=df.index,y=df['close'].rolling(10).mean(),name='ma-10W'))
    fig.add_trace(go.Scatter(x=df.index,y=df['close'].rolling(40).mean(),name='ma-40W'))
    lines_to_draw = []
    for t in type1:
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        stockS = data.iloc[t[0][0]].high
        stockE = data.iloc[t[1][0]].high

        Ds, De = D.iloc[t[0][1]], D.iloc[t[1][1]]

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS)/(eS-sS)
            Dm = (eD - sS)/(eD-sD)
        
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

                
                lines_to_draw.append(dict(x0=data.iloc[dStart].name,y0=D.iloc[dStart],x1=data.iloc[dEnd].name,y1=D.iloc[dEnd],type='line',xref='x2',yref='y2',line_width=7))
                lines_to_draw.append(dict(x0=data.iloc[stockStart].name,y0=data.iloc[stockStart].low,x1=data.iloc[stockEnd].name,y1=data.iloc[stockEnd].low,type='line',xref='x',yref='y',line_width=7))
                


    for t in type2:
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
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending

            
                lines_to_draw.append(dict(x0=data.iloc[dStart].name,y0=D.iloc[dStart],x1=data.iloc[dEnd].name,y1=D.iloc[dEnd],type='line',xref='x2',yref='y2',line_width=7))
                lines_to_draw.append(dict(x0=data.iloc[stockStart].name,y0=data.iloc[stockStart].high,x1=data.iloc[stockEnd].name,y1=data.iloc[stockEnd].high,type='line',xref='x',yref='y',line_width=7))

    fig.update_layout(shapes = lines_to_draw)
    # fig.show()
    signal_related_vars_dict={}
    return fig,signal_related_vars_dict

if __name__ == "__main__":
    runStochDivergance('^DJI', '2021-11-01', R_=1.02, endDate_='2023-03-03')
