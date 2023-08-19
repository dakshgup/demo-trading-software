from turtle import st
import yfinance_dup as yf
import numpy as np
import pandas as pd
from finta import TA
from CorrectR import getPoints, getPointsforArray
import matplotlib.pyplot as plt
import finplot_mod as fplt
from getPointsFile import getPointsBest, getPointsGivenR
import pandas_market_calendars as mcal
import os
from datetime import date, timedelta, datetime
import pickle

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

    # print(data.info())
    # print(highs, lows)
    # plt.plot(np.asarray(data["low"]),'-o', markevery=lows, markersize=5, fillstyle='none')
    # plt.yscale('log')
    # plt.show()

    # plt.plot(np.asarray(data["high"]),'-o', markevery=highs, markersize=5, fillstyle='none')
    # plt.yscale('log')
    # plt.show()

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

    # for r in rrS:
    #     print(r.start, r.end, r.class_)

    # print("hejrgsfbhjkds")
    # frS = getFinalReigons(rrS)

    # for r in frS:
        # print(r.start, r.end, r.class_)


    rr1 = getReigons(highs, data['high'])

    # for r in rr1:
    #     print(r.start, r.end, r.class_)

    # print("hejrgsfbhjkds")
    fr1 = getFinalReigons(rr1)

    # for r in fr1:
    #     print(r.start, r.end, r.class_)

    # for h in highsStoch:
    #     h += 26

    rrS1 = getReigons(highsStoch, D)

    # print("hejrgsfbh")
    frS1 = getFinalReigons(rrS1)

    # for l in lowsStoch:
    #     l += 26

    rrS1 = getReigons(lowsStoch, D)

    # for r in rrS1:
    #     print(r.start, r.end, r.class_)

    # print("hejrgsfbhjkds")
    frS2 = getFinalReigons(rrS1)

    # for r in frS1:
    #     print(r.start, r.end, r.class_)

    type1 = getDivergance_LL_HL(fr, frS2)
    type2 = getDivergance_HH_LH(fr1, frS1)


    # print('\n\nLL Stock, HL D\n')
    # print(type1)
    # print('\nHH Stock, LH D\n')
    # print(type2)

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
        
            # print("YYOY")
##########################################################################
            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = sS
                stockEnd = eS

                dStart = sD
                dEnd = eD

                # fplt.add_line((t[0][1], D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=7, color='r')
                # fplt.add_line((t[0][0], data.iloc[t[0][0]].low), (t[1][0],data.iloc[t[1][0]].low), ax=ax, width=7, color='r')
                
                # fplt.add_line((dStart, D.iloc[dStart]), (dEnd,D.iloc[dEnd]), ax=ax1, width=7)
                # fplt.add_line((stockStart, data.iloc[stockStart].low), (stockEnd,data.iloc[stockEnd].low), ax=ax, width=7)
                # if pd.to_datetime(data.iloc[stockEnd].name).strftime('%Y-%m-%d') == endDate_:
                for i in range(3,31):
                    if (stockEnd > len(df)-6 or dEnd > len(df)-6) and D.iloc[stockStart]<i:
                        if Dm > 0.2 or StockM < -0.2:
                            output_file = open(f'results/{STOCK}/data_{i}.csv','a')
                            output_file.write(str(data.iloc[-1].name).split(' ')[0])
                            output_file.write(',buy,'+str(D.iloc[dStart])+','+str(D.iloc[dEnd]) + ',' +str(data.iloc[stockStart].name).split(' ')[0] + ','+str(data.iloc[stockEnd].name).split(' ')[0] )
                            output_file.write('\n')
                            output_file.close()

    # print(type2)

    for t in type2:
        # print(t)
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
                stockStart = sS
                stockEnd = eS

                dStart = sD
                dEnd = eD

                # fplt.add_line((t[0][1], D.iloc[t[0][1]]), (t[1][1],D.iloc[t[1][1]]), ax=ax1, width=7, color='g')
                # fplt.add_line((t[0][0], data.iloc[t[0][0]].high), (t[1][0],data.iloc[t[1][0]].high), ax=ax, width=7, color='g')
            
                fplt.add_line((dStart, D.iloc[dStart]), (dEnd,D.iloc[dEnd]), ax=ax1, width=7)
                fplt.add_line((stockStart, data.iloc[stockStart].high), (stockEnd,data.iloc[stockEnd].high), ax=ax, width=7)
                # for i in range(5,30):
                #     if (stockEnd > len(df)-6 or dEnd > len(df)-6) and D.iloc[stockStart]<i:
                #         if Dm > 0.2 or StockM < -0.2:
                #             output_file = open(f'results/{STOCK}/data_{i}.csv','a')
                #             output_file.write(str(data.iloc[-1].name).split(' ')[0])
                #             output_file.write(',buy,'+str(D.iloc[dStart])+','+str(D.iloc[dEnd]) + ',' +str(data.iloc[stockStart].name).split(' ')[0] + ','+str(data.iloc[stockEnd].name).split(' ')[0] )
                #             output_file.write('\n')
                #             output_file.close()
                for i in range(70,98):
                    # print(stockStart,stockEnd,dStart,dEnd,Dm,StockM,D.iloc[stockStart])
                    if (stockEnd > len(df)-6 or dEnd > len(df)-6) and D.iloc[stockStart]>i:
                        if Dm < -0.2 or StockM > 0.2:
                            output_file = open(f'results/{STOCK}/data_{i}.csv','a')
                            output_file.write(str(data.iloc[-1].name).split(' ')[0])
                            output_file.write(',sell,'+str(D.iloc[dStart])+','+str(D.iloc[dEnd]) + ',' +str(data.iloc[stockStart].name).split(' ')[0] + ','+str(data.iloc[stockEnd].name).split(' ')[0] )
                            output_file.write('\n')
                            output_file.close()
                    
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def collect_data(stock,start_date,end_date):
    try:
        os.mkdir(f'results/{stock}')
    except:
        os.system(f'rm -r results/{stock}/*')
    for single_date in mcal.get_calendar('NYSE').valid_days(start_date = start_date,end_date = end_date):
        runStochDivergance(stock, single_date-timedelta(365*2), R_=1.02, endDate_=single_date.strftime("%Y-%m-%d"))

def get_signals_from_data(stock,data_of_stock,j,k):
    D = TA.STOCHD(data_of_stock)
    D = D[15:]
    data = data_of_stock[15:]
    buy_signal_dates = []
    sell_signal_dates = []
    dates_pair = []
    begin_dates = []
    try:
        data_k_csv = open(f'results/{stock}/data_{k}.csv','r')
    except:
        return buy_signal_dates,sell_signal_dates
    for line in data_k_csv.readlines():
        if(len(line.split(',')) == 1):   continue
        curr_date1 = line.split(',')[4]
        curr_date2 = line.strip().split(',')[5]
        if len(begin_dates) != 0 and curr_date1 in begin_dates:
            continue
        begin_dates.append(curr_date1)
        idx = 0
        signal = line.split(',')[1]
        if(signal== 'buy'):
            if D.loc[curr_date2] >= j:
                if(line.split(',')[0] not in buy_signal_dates):
                    buy_signal_dates.append(line.split(',')[0])
                continue
            for i in range(len(data)):
                if(datetime.strftime(data.iloc[i].name,'%Y-%m-%d')==curr_date2):
                    idx = i
                    break
            for i in range(idx,len(data)):
                if(D.iloc[i] >= j):
                    # outfile.write(str(data.iloc[i].name).split(' ')[0] + ',buy,'+curr_date1 + ',' + curr_date2+'\n')
                    if str(data.iloc[i].name).split(' ')[0] not in buy_signal_dates:
                        buy_signal_dates.append(str(data.iloc[i].name).split(' ')[0])
                    break

        if(signal== 'sell'):
            if D.loc[curr_date2] <= j:
                # outfile.write(line.split(',')[0] + ',sell,'+curr_date1+','+curr_date2+'\n')
                if line.split(',')[0] not in sell_signal_dates:
                    sell_signal_dates.append(line.split(',')[0])
                continue
            for i in range(len(data)):
                if(datetime.strftime(data.iloc[i].name,'%Y-%m-%d')==curr_date2):
                    idx = i
                    break
            for i in range(idx,len(data)):
                if(D.iloc[i] <= j):
                    # outfile.write(str(data.iloc[i].name).split(' ')[0] + ',sell,'+curr_date1 + ',' + curr_date2+'\n')
                    if str(data.iloc[i].name).split(' ')[0] not in sell_signal_dates:
                        # print(data.iloc[i].name)
                        sell_signal_dates.append(str(data.iloc[i].name).split(' ')[0])
                    break
    data_k_csv.close()
    # outfile.close()
    return buy_signal_dates,sell_signal_dates

def calculate_score(param1,param2,param3,k_days,data,stock,buy_signals,sell_signals,D):
    number_of_signals = len(buy_signals) + len(sell_signals)
    number_of_profitable_signals = 0
    number_of_lossy_signals = 0
    profit = 0
    loss = 0
    # print(buy_signals,sell_signals)
    for curr_date in (buy_signals):
        buy_date_idx = list(data.index).index(datetime.strptime(curr_date,'%Y-%m-%d')) + 1
        buy_price = data.iloc[buy_date_idx].Open
        idx = buy_date_idx + 1
        rose_above_param3 = False
        while idx < buy_date_idx + k_days:
            if not rose_above_param3:
                if D[idx] > param3:
                    rose_above_param3 = True
                idx+=1
                continue
            if D[idx] < param3:
                break
            idx+=1
        sell_date_idx = idx+1
        sell_price = data.iloc[sell_date_idx].Open
        if sell_price > buy_price:
            print(f'Profit={(sell_price-buy_price)/buy_price}\tholding_period={sell_date_idx-buy_date_idx}')
            profit += ((sell_price - buy_price)/buy_price)
            number_of_profitable_signals += 1
        elif buy_price >= sell_price:
            print(f'Profit={(sell_price-buy_price)/buy_price}\tholding_period={sell_date_idx-buy_date_idx}')
            loss += ((buy_price - sell_price)/buy_price)
            number_of_lossy_signals += 1
        print(data.iloc[buy_date_idx].name,data.iloc[sell_date_idx].name,sell_price-buy_price)
    for curr_date in (sell_signals):
        sell_date_idx = list(data.index).index(datetime.strptime(curr_date,'%Y-%m-%d')) + 1
        sell_price = data.iloc[sell_date_idx].Open
        idx = sell_date_idx + 1
        fell_below_param3 = False
        while idx < sell_date_idx + k_days:
            if not fell_below_param3:
                if D[idx] < param3:
                    fell_below_param3 = True
                idx+=1
                continue
            if D[idx] > param3:
                break
            idx+=1
        buy_date_idx = idx+1
        buy_price = data.iloc[buy_date_idx].Open
        if buy_price > sell_price:
            print(f'Profit={(sell_price-buy_price)/buy_price}\tholding_period={buy_date_idx-sell_date_idx}')
            profit += ((sell_price - buy_price)/buy_price)
            number_of_profitable_signals += 1
        elif buy_price >= sell_price:
            print(f'Profit={(sell_price-buy_price)/buy_price}\tholding_period={buy_date_idx-sell_date_idx}')
            loss += ((buy_price - sell_price)/buy_price)
            number_of_lossy_signals += 1
        print(data.iloc[buy_date_idx].name,data.iloc[sell_date_idx].name,sell_price-buy_price)
    print(profit,loss,number_of_profitable_signals,number_of_lossy_signals)
    return (profit-loss)/(number_of_profitable_signals + number_of_lossy_signals)

def calculate_score_optimizing_days(param1,param2,param3,param4,data,stock,buy_signals,sell_signals,D):
    number_of_signals = len(buy_signals) + len(sell_signals)
    number_of_profitable_signals = [0]*70
    number_of_lossy_signals = [0]*70
    profit = [0]*70
    loss = [0]*70
    # print(buy_signals,sell_signals)
    for curr_date in buy_signals:
        buy_date_idx = list(data.index).index(datetime.strptime(curr_date,'%Y-%m-%d')) + 1
        buy_price = data.iloc[buy_date_idx].Open
        idx = buy_date_idx + 1
        rose_above_param3 = False
        while idx < buy_date_idx + 70:
            if data.iloc[idx].Open > buy_price:
                profit[idx-buy_date_idx] += (data.iloc[idx].Open - buy_price)/buy_price
                number_of_profitable_signals[idx-buy_date_idx] += 1
            else:
                loss[idx-buy_date_idx] += (buy_price - data.iloc[idx].Open)/buy_price
                number_of_lossy_signals[idx-buy_date_idx] += 1
            if (data.iloc[idx].Open - buy_price)/buy_price < -0.05:
                idx += 1
                break
            if not rose_above_param3:
                if D[idx] > param3:
                    rose_above_param3 = True
                idx+=1
                continue
            if D[idx] < param4:
                idx += 1
                break
            idx+=1
        sell_date_idx = idx
        sell_price = data.iloc[sell_date_idx].Open
        if sell_price > buy_price:
            curr_profit = (sell_price - buy_price) / buy_price
            for i in range(sell_date_idx-buy_date_idx,70):
                profit[i] += curr_profit
                number_of_profitable_signals[i] += 1
        else:
            curr_profit = (buy_price - sell_price) / buy_price
            for i in range(sell_date_idx-buy_date_idx,70):
                loss[i] += curr_profit
                number_of_lossy_signals[i] += 1
    for curr_date in sell_signals:
        sell_date_idx = list(data.index).index(datetime.strptime(curr_date,'%Y-%m-%d')) + 1
        sell_price = data.iloc[sell_date_idx].Open
        idx = sell_date_idx + 1
        fell_below_param3 = False
        while idx < sell_date_idx + 70:
            if data.iloc[idx].Open < sell_price:
                profit[idx-sell_date_idx] += (sell_price - data.iloc[idx].Open)/data.iloc[idx].Open
                number_of_profitable_signals[idx-sell_date_idx] += 1
            else:
                loss[idx-sell_date_idx] += (data.iloc[idx].Open - sell_price)/data.iloc[idx].Open
                number_of_lossy_signals[idx-sell_date_idx] += 1
            if (sell_price - data.iloc[idx].Open)/data.iloc[idx].Open < -0.05:
                idx += 1
                break
            if not fell_below_param3:
                if D[idx] < param3:
                    fell_below_param3 = True
                idx+=1
                continue
            if D[idx] > param4:
                idx += 1
                break
            idx+=1
        buy_date_idx = idx
        buy_price = data.iloc[buy_date_idx].Open
        if sell_price > buy_price:
            # print(f'Profit={(sell_price-buy_price)/buy_price}\tholding_period={sell_date_idx-buy_date_idx}')
            curr_profit = (sell_price- buy_price) / buy_price
            for i in range(idx-sell_date_idx,70):
                profit[i] += curr_profit
                number_of_profitable_signals[i] += 1
        elif buy_price >= sell_price:
            # print(f'Profit={(sell_price-buy_price)/buy_price}\tholding_period={sell_date_idx-buy_date_idx}')
            curr_profit = (buy_price - sell_price) / buy_price
            for i in range(idx-sell_date_idx,70):
                loss[i] += curr_profit
                number_of_lossy_signals[i] += 1
    scores = [None]*70
    max_idx = 1
    for i in range(1,70):
        scores[i] = (profit[i] - loss[i])/(number_of_profitable_signals[i] + number_of_lossy_signals[i])
        if(scores[i] > scores[max_idx]):
            max_idx = i
    return scores[max_idx],max_idx

def optimize_parameters(stock,startDate,endDate):
    collect_data(stock,startDate,endDate)
    data = yf.download(stock,((datetime.strptime(startDate,'%Y-%m-%d')-timedelta(100))).strftime('%Y-%m-%d'),'2030-01-01')
    print(data)
    best_params1 = [(5,10,60,1)]
    max_score = 0
    D = TA.STOCHD(data)
    for param1 in range(3,31):
        print(f'Testing with {param1}')
        for param2 in range(param1+5,param1+30):
            buy_signals, sell_signals = get_signals_from_data(stock,data,param2,param1)
            if len(buy_signals) == 0 and len(sell_signals) == 0:
                continue
            for param3 in range(90,100):
                for param4 in range(70,90):
                    # score = average_profit
                    curr_score,k_days = calculate_score_optimizing_days(param1,param2,param3,param4,data,stock,buy_signals,sell_signals,D)
                    if curr_score > max_score:
                        max_score = curr_score
                        best_params1 = [(param1,param2,param3,param4,k_days)]
                        # print(f"Updated Best Params: {curr_score}")
                    elif curr_score == max_score:
                        best_params1.append((param1,param2,param3,param4,k_days))
                        # print(f'Updated Best Params: {best_params1}')
    max_score = -100000
    best_params2 = []
    for param1 in range(70,98):
        print(f'Testing with {param1}')
        for param2 in range(param1-30,param1-5):
            buy_signals, sell_signals = get_signals_from_data(stock,data,param2,param1)
            # print(buy_signals,sell_signals)
            if len(buy_signals) == 0 and len(sell_signals) == 0:
                continue
            for param3 in range(0,10):
                for param4 in range(10,30):
                    # score = average_profit
                    curr_score,k_days = calculate_score_optimizing_days(param1,param2,param3,param4,data,stock,buy_signals,sell_signals,D)
                    if curr_score > max_score:
                        max_score = curr_score
                        best_params2 = [(param1,param2,param3,param4,k_days)]
                        # print(f"Updated Best Params: {best_params2}")
                    elif curr_score == max_score:
                        best_params2.append((param1,param2,param3,param4,k_days))
                        # print(f'Updated Best Params: {best_params2}')
    return best_params1,best_params2

def stochDivParamOptimization(stock):
    best_params = optimize_parameters(stock,(datetime.now()-timedelta(110+365)).strftime('%Y-%m-%d'),(datetime.now()-timedelta(110)).strftime('%Y-%m-%d'))
    with open(f'best_params/{stock}.pkl','wb') as f:
        pickle.dump(best_params,f)
