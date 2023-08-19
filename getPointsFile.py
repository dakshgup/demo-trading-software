from CorrectR import getPointsforArray
import yfinance_dup as yf
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime
from datetime import datetime, timedelta
from Utilities import Linear


def getScaledY(data):
    return (np.asarray(data) - min(data))/(max(data) - min(data))

def getScaledX(x, data):
    return np.asarray(x) / len(data)

def getUnscaledX(x, data):
    p =  x * len(data)
    return int(p)

def getLinears(data, Tps):
    linears = []
    i = 0
    while i+1 < len(Tps):
        l = Linear(getScaledX(Tps[i], data), data[Tps[i]], getScaledX(Tps[i+1], data), data[Tps[i+1]])
        linears.append(l)
        i += 1
    return linears

def getLinearForX(lins, x):
    for l in lins:
        if l.isInRange(x):
            return l
    return None

def getMSE(data, lins):
    # We start from where the first linear approximation begins 
    i = lins[0].x1
    E = 0
    while i<lins[-1].x2:
        l = getLinearForX(lins, i)
        p = data[getUnscaledX(i, data)]
        pHat = l.getY(i)
        E += abs((p - pHat))*1/len(data)
        i += 1/len(data)
    return E*10
    # N = getUnscaledX(lins[-1].x2 - lins[0].x1, data)
    # return (1/N)*(E**(1/2))*1000

def getPointsBest(STOCK, min_=0.23, max_=0.8, getR=False, startDate='2000-01-01', endDate='2121-01-01', increment=0.005, limit=100, interval=None, returnData='close'):
    # , start='2020-01-01'
    if interval is None:
        data = yf.download(STOCK, start=startDate, end=endDate)
    else:
        data = yf.download(STOCK, start=startDate, end=endDate, interval=interval)
    
    data = data.dropna()
    data = data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Volume":"volume", "Close":"close"})
    
    date_format = "%Y-%m-%d"
    # a = datetime.strptime(startDate, date_format)
    end_ = datetime.strptime(endDate, date_format)
    today = datetime.today()

    d = today - end_
    # print(d.days)

    # if d.days <= 0:
    #     lastDay = datetime.today() - timedelta(days=1)
    #     ld = lastDay.strftime("%Y-%m-%d")
    #     latest =  yf.download(STOCK, start=ld, end=endDate, interval='1m')
    #     print(latest.tail())
    #     print(data.tail())

    #     print(data.iloc[-1])
    #     # data.__setitem__(data.iloc[-1].close , 2)
    #     data["close"] = data["close"].replace([data["close"][-1]], latest["Close"][-1])
    #     print("______________")
    #     print(data.iloc[-1])
    #     print(data.tail())

    if returnData == 'close':
        Sdata = getScaledY(data["close"])
    elif returnData == 'lows':
        Sdata = getScaledY(data["low"])
    elif returnData == 'highs':
        Sdata = getScaledY(data["high"])
    else:
        print("Wrong data for argument returnData")
        return None
    
    R = 1.1
    satisfied = False
    c = 0
    while not satisfied and c < limit and R > 1:
        if returnData == 'close':
            highs, lows = getPointsforArray(data["close"], R)
        elif returnData == 'lows':
            highs, lows = getPointsforArray(data["low"], R)
        else:
            highs, lows = getPointsforArray(data["high"], R)
        # highs, lows = getPointsforArray(data["close"], R)
        # plt.plot(data["close"], '-o', markevery=highs+lows)
        # plt.show()
        if not len(highs) <2 and not len(lows) < 2:
            linears = getLinears(Sdata, sorted(highs+lows))
            MSE = getMSE(Sdata, linears)
            c += 1
            if min_ < MSE and MSE < max_:
                satisfied = True
            elif MSE > min_:
                R -= increment
            else:
                R += increment
            print(c, R, MSE)
        else:
            R -= increment
    if R > 1:
        # h, l = getPointsforArray(data["close"], R)

        if returnData == 'close':
            h, l = getPointsforArray(data["close"], R)
        elif returnData == 'lows':
            h, l = getPointsforArray(data["low"], R)
        else:
            h, l = getPointsforArray(data["close"], R)

    else:
        # print("HERE")
        # h, l = getPointsforArray(data["close"], 1.001)

        if returnData == 'close':
            h, l = getPointsforArray(data["close"], 1.001)
        elif returnData == 'lows':
            h, l = getPointsforArray(data["low"], 1.001)
        else:
            h, l = getPointsforArray(data["close"], 1.001)

    if getR:
        return data, h, l, R
    else:
        return data, h, l

def getPointsGivenR(STOCK, R, startDate='2000-01-01', endDate='2121-01-01', interval=None, type_=None):
    if interval is None:
        data = yf.download(STOCK, start=startDate, end=endDate)
    else:
        data = yf.download(STOCK, start=startDate, end=endDate, interval=interval)
    data = data.dropna()
    data = data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Volume":"volume", "Close":"close"})

    date_format = "%Y-%m-%d"
    end_ = datetime.strptime(endDate, date_format)
    today = datetime.today()

    d = today - end_
    # print(d.days)

    # if d.days <= 0:
    #     lastDay = datetime.today() - timedelta(days=1)
    #     ld = lastDay.strftime("%Y-%m-%d")
    #     latest =  yf.download(STOCK, start=ld, end=endDate, interval='1m')
    #     print(latest.tail())
    #     print(data.tail())

    #     print(data.iloc[-1])
    #     # data.__setitem__(data.iloc[-1].close , 2)
    #     data["close"] = data["close"].replace([data["close"][-1]], latest["Close"][-1])
    #     print("______________")
    #     print(data.iloc[-1])
    #     print(data.tail())

    if type_ is None:
        highs, lows = getPointsforArray(data["close"], R)
        return data, highs, lows
    elif type_== 'lows':
        _, lows = getPointsforArray(data["low"], R)
        return data, lows
    elif type_== 'highs':
        highs, _ = getPointsforArray(data["high"], R)
        return data, highs
    else:
        return None, None

if __name__ == "__main__":
    dataB, highsB, lowsB = getPointsBest('arkk')

    plt.plot(dataB["close"], '-o', markevery=highsB+lowsB, markersize=5, fillstyle='none')
    plt.yscale('log')
    plt.show()
