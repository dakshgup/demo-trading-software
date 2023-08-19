import yfinance as yf
from datetime import timedelta

def IntermarketDivergence(stock1,stock2,startDate = '2010-01-01',endDate = '2100-01-01'):
    d1 = yf.download(stock1,startDate,endDate,interval='1wk')
    d2 = yf.download(stock2,startDate,endDate,interval='1wk')
    if d1.iloc[-1].name - d1.iloc[-2].name < timedelta(7):
        d1.drop(d1.tail(1).index,inplace=True)
    if d2.iloc[-1].name - d2.iloc[-2].name < timedelta(7):
        d2.drop(d2.tail(1).index,inplace=True)

    assert(d1.iloc[-1].name == d2.iloc[-1].name)

    # print(d1[-40:])
    # print(d2[-40:])

    i = len(d1) - 2
    
    starts = -1
    ends = -1
    while i >= 0:
        # print(d1.iloc[i])
        dir1 = 0
        if d1.iloc[i].High > d1.iloc[-1].High:
            dir1 = -1
        elif d1.iloc[i].High < d1.iloc[-1].High:
            dir1 = 1
        dir2 = 0
        if d2.iloc[i].High > d2.iloc[-1].High:
            dir2 = -1
        elif d2.iloc[i].High < d2.iloc[-1].High:
            dir2 = 1
        # print(d1.iloc[i].name,d2.iloc[i].name,dir1,dir2)

        if not (dir1 == dir2):
            if starts == -1:
                starts = len(d1) - i - 1
            ends = len(d1) - i - 1
        if starts != -1:
            break
        i -= 1
        
    if starts != -1:
        return True,starts,ends,1

    i = len(d1) - 2
    
    starts = -1
    ends = -1
    while i >= 0:
        dir1 = 0
        if d1.iloc[i].Low > d1.iloc[-1].Low:
            dir1 = -1
        elif d1.iloc[i].Low < d1.iloc[-1].Low:
            dir1 = 1
        dir2 = 0
        if d2.iloc[i].Low > d2.iloc[-1].Low:
            dir2 = -1
        elif d2.iloc[i].Low < d2.iloc[-1].Low:
            dir2 = 1

        if not (dir1 == dir2):
            if starts != -1:
                starts = len(d1) - i - 1
            ends = len(d1) - i - 1
        if starts != -1:
            break
        i -= 1
    if starts != -1:
        return True,starts,ends,-1
    return False,-1,-1,0


if __name__ == "__main__":
    s1 = "^NDX"
    s2 = "^DJI"
    found,start,end,dirn = IntermarketDivergence(s1,s2)
    if found:
        print(f'{s1} and {s2} diverge between {"highs" if dirn==1 else "lows"}. Divergence starts at T-{start} week and continues for {end-start+1} week(s)')
    s1 = "^GSPC"
    s2 = "^DJI"
    found,start,end,dirn = IntermarketDivergence(s1,s2)
    if found:
        print(f'{s1} and {s2} diverge between {"highs" if dirn==1 else "lows"}. Divergence starts at T-{start} week and continues for {end-start+1} week(s)')
    s1 = "^NDX"
    s2 = "^GSPC"
    found,start,end,dirn = IntermarketDivergence(s1,s2)
    if found:
        print(f'{s1} and {s2} diverge between {"highs" if dirn==1 else "lows"}. Divergence starts at T-{start} week and continues for {end-start+1} week(s)')
