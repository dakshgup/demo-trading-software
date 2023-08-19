import matplotlib.pyplot as plt
import yfinance_dup as yf
import numpy as np
from finta import TA
import random
import math

def findFirst(a, n, R, markers_on):
    iMin = 1
    iMax = 1
    i = 2
    while i<n and a[i]/a[iMin]< R and a[iMax]/a[i]< R:
        if a[i] < a[iMin]:
            iMin = i
        if a[i] > a[iMax]:
            iMax = i
        i += 1
    if iMin < iMax:
        markers_on.append(iMin)
    else:
        markers_on.append(iMax)
    return i, markers_on

def finMin(i, a, n, R, markers_on):
    iMin = i
    while i < n and a[i]/a[iMin]< R:
        if a[i] < a[iMin]:
            iMin = i
        i += 1
    if i < n or a[iMin] < a[i]:
        markers_on.append(iMin)
    return i, markers_on

def finMax(i, a, n, R, markers_on):
    iMax = i
    while i < n and a[iMax]/a[i] < R:
        if a[i] > a[iMax]:
            iMax = i
        i += 1
    if i < n or a[iMax] > a[i]:
        markers_on.append(iMax)
    return i, markers_on

# Higher R results in less points and lower R results in identifying more peaks (R > 1)
def getTurningPoints(closeSmall, R, combined=True):
    
    markers_on = []
    highs = []
    lows = []
    i, markers_on = findFirst(closeSmall, len(closeSmall), R, markers_on)
    if i < len(closeSmall) and closeSmall[i] > closeSmall[0]:
        i, highs = finMax(i, closeSmall, len(closeSmall)-1, R, highs)
    while i < len(closeSmall)-1 and not math.isnan(closeSmall[i]):
        i, lows = finMin(i, closeSmall, len(closeSmall)-1, R, lows)
        i, highs = finMax(i, closeSmall, len(closeSmall)-1, R, highs)
        # print("here", i, R, closeSmall[i])
    if combined:
        return highs + lows
    else:
        return highs, lows

# Finction that returns the revenue gievn then highs, lows and closing price. 
# highs and lows are array that have the indeices of highs and lows given by the above algorithm 
# closing is an array of the closing prices
def getRevenue(highs, lows, closing):
    inHand = 0
    numShares = 0
    # lastDay  =min(highs[-1], lows[-1])
    lastDay = highs[-1]
    day = lows[0]
    while day <= lastDay:
        # if its a low you buy tghe stock so money in hand goes down
        if day in lows:
            inHand -= closing[day]
            numShares += 1
            # print("Profit is (after buying at day", day, "):", inHand)
            if numShares > 1:
                print("SOMETHING WRONG")
        # if its a high you sell the stock and the money in hand goes up
        elif day in highs:
            if numShares > 0:
                inHand += closing[day]
                numShares -= 1
                # print("Profit is (after selling at day", day, "): ", inHand)
            if numShares > 1:
                print("SOMETHING WRONG")
 
        day += 1

    return inHand

# print(getRevenue(highs, lows, closing))

# Finction that returns the penalty for the highs and lows chosen by the algorithm 
# highs and lows are array that have the indeices of highs and lows given by the above algorithm 
# closing is an array of the closing prices
def getPenalty(beta, highs, lows, closing, pricePenalty=True, precentage=0):
    high = np.asarray(highs)
    low = np.asarray(lows)
    TPs = np.concatenate([high, low])
    TPs = np.sort(TPs)

    penalty = 0
    i = 1
    while i < len(TPs):
        s = beta - (TPs[i] - TPs[i-1])
        penalty += max(s, 0)
        i += 1
    
    if pricePenalty:
        penaltyPrice = 0
        i = 1
        while i < len(TPs):
            s = precentage - abs((closing[TPs[i]] - closing[TPs[i-1]]/ closing[TPs[i-1]]))
            penalty += max(s, 0)
            i += 1
        return penalty + penaltyPrice
    else:
        return penalty

# Finction that returns the score for the highs and lows chosen by the algorithm 
# highs and lows are array that have the indeices of highs and lows given by the above algorithm 
# closing is an array of the closing prices
# alpha and beta are parametrs (alpha is to make sure how severe the penalty is very close TPs)
# (beta is used to define how close and points too close - in terms of days)
def getScore(alpha, beta, percentage, highs, lows, closing, pricePenalty=True):
    scaler = None
    data = scaler.fit_transform(closing.reshape(-1, 1))
    revenue = getRevenue(highs, lows, closing)
    penalty = getPenalty(beta, highs, lows, closing, pricePenalty=pricePenalty, precentage=percentage)

    score = revenue - (alpha * penalty)
    return score


def getPoints(stock, R_=None, alpha=2, beta=10, percentage=0.2, startDate='2000-01-01', combined=True, showPlot=False, getData=False, interval_='1d', endDate="2100-10-01"):
    data = yf.download(stock, start=startDate, end=endDate, interval = interval_)
    data = data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Volume":"volume", "Close":"close"})
    data = data.dropna()
    closing = np.asarray(data["close"])

    if R_ == None:
        Rs = np.linspace(1.0001, 3, num=2000)
        maxScore = 0
        bestR = Rs[0]
        for R in Rs:
            highs, lows = getTurningPoints(closing, R, combined=False)
            if len(highs) > 0 and len(lows) > 0:
                score = getScore(alpha, beta, percentage, highs, lows, closing)
                # print("Score is:", score, "With R being:", R)
                if score > maxScore:
                    maxScore = score
                    bestR = R

        print(maxScore, bestR, "THERIS")
    else:
        bestR = R_

    if combined:
        TPs = getTurningPoints(closing, bestR)
    else:
        highs, lows = getTurningPoints(closing, bestR, combined=False)
    if showPlot:
        plt.plot(data["close"],'-o', markevery=TPs, markersize=5, fillstyle='none')
        plt.yscale("log")
        plt.show()
    
    if combined:
        return TPs
    if not combined and getData:
        return data, highs, lows
    else:
        return highs, lows

def getPointsforArray(series, R=1.1):
    highs, lows = getTurningPoints(series, R, combined=False)
    return highs, lows

def getTuringPointsOnYear(STOCK, pointsPerYear=5, startDate='2000-01-01', endDate='2100-01-01', combined_=False, getData=False, interval_='1d', getR=False):
    data = yf.download(STOCK, start=startDate, end=endDate, interval = interval_)
    data = data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Volume":"volume", "Close":"close"})
    data = data.dropna()
    if not pointsPerYear == None:
        RStart = 1.2
        TpsPerYear = pointsPerYear
        satisfied = False
        c = 0
        while not satisfied and c<30:
            highs, lows = getTurningPoints(data["close"], RStart, combined=False)
            years = len(data)/260
            # plt.close()
            TpPerYearWeGot = len(highs+lows)/years
            if TpsPerYear - 0.5 <= TpPerYearWeGot <= TpsPerYear + 0.5:
                satisfied = True
            elif TpPerYearWeGot > TpsPerYear + 0.5:
                RStart += 0.01
            else:
                RStart -= 0.01
            c+= 1
        highs, lows = getTurningPoints(data["close"], RStart, combined=False)
        # print(len(highs+lows), len(highs+lows)/years)

        if not getR:
            if getData and not combined_:
                return data, highs, lows
            elif getData and combined_:
                return data, highs+lows
            elif not getData and not combined_:
                return highs, lows
            else:
                return highs+lows
        else:
            if getData and not combined_:
                return data, highs, lows, RStart
            elif getData and combined_:
                return data, highs+lows, RStart
            elif not getData and not combined_:
                return highs, lows, RStart
            else:
                return highs+lows, RStart
    else:
        return data

def getPointsGeneral(data, R, combined=True, showPlot=False, getData=False):
    closing = data
    if combined:
        TPs = getTurningPoints(closing, R)
    else:
        highs, lows = getTurningPoints(closing, R, combined=False)
    if showPlot:
        plt.plot(data,'-o', markevery=TPs, markersize=5, fillstyle='none')
        plt.show()
    
    if combined:
        return TPs
    if not combined and getData:
        return data, highs, lows
    else:
        return highs, lows


if __name__ == "__main__":
    # getPoints('^GSPC')
    # getPoints('GOOGL')
    data, highs, lows = getPoints('baba', R_=1.1, alpha=1, beta=20, combined=False, showPlot=False, getData=True, startDate='2000-01-01', interval_="1d", endDate="2021-10-01")

    print(highs)
    print(lows)
    

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["close"],
        mode='lines',
        line=dict(
            width=1,
            color='black'
        ),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=data.index[highs + lows],
        y=data["close"].iloc[highs + lows],
        mode='markers',
        marker=dict(
            symbol='circle-open',
            size=5,
            line=dict(
                width=1,
                color='black'
            ),
        ),
        name='Highs and Lows'
    ))

    fig.update_layout(
        yaxis_type="log"
    )

    fig.show()



    
