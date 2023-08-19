import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from datetime import timedelta

def IntermarketDivergenceHighs(stock1,stock2,startDate = '2010-01-01',endDate = '2100-01-01'):
    d1 = yf.download(stock1,startDate,endDate,interval='1wk')
    d2 = yf.download(stock2,startDate,endDate,interval='1wk')
    if d1.iloc[-1].name - d1.iloc[-2].name < timedelta(7):
        d1.drop(d1.tail(1).index,inplace=True)
    if d2.iloc[-1].name - d2.iloc[-2].name < timedelta(7):
        d2.drop(d2.tail(1).index,inplace=True)

    assert(d1.iloc[-1].name == d2.iloc[-1].name)

    i = len(d1) - 2
    
    starts = -1
    ends = -1
    while i >= 0:
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

        if not (dir1 == dir2):
            # print("HIHHI",i)
            # print(dir1,dir2,d1.iloc[i].High,d1.iloc[-1].High,d2.iloc[i].High,d2.iloc[-1].High)
            if starts == -1:
                starts = i
            ends = i
        if starts != -1:
            break
        i -= 1
        
    if starts == -1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.01,subplot_titles=(stock1, stock2))
        fig.add_trace(go.Candlestick(x=d1.index,open=d1['Open'],high=d1['High'],low=d1['Low'],close=d1['Close']),row=1,col=1)
        fig.add_trace(go.Candlestick(x=d2.index,open=d2['Open'],high=d2['High'],low=d2['Low'],close=d2['Close']),row=2,col=1)
        fig.update_yaxes(type='log',row=1,col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(xaxis2_rangeslider_visible=False)
        fig.update_yaxes(type='log',row=2,col=1)
        return False,-1,-1,fig

    start_date = d1.iloc[starts].name
    end_date = d1.iloc[ends].name
    d1 = d1.iloc[ends-3:]
    d2 = d2.iloc[ends-3:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.01,subplot_titles=(stock1, stock2))
    fig.add_trace(go.Candlestick(x=d1.index,open=d1['Open'],high=d1['High'],low=d1['Low'],close=d1['Close']),row=1,col=1)
    fig.add_trace(go.Candlestick(x=d2.index,open=d2['Open'],high=d2['High'],low=d2['Low'],close=d2['Close']),row=2,col=1)
    fig.update_yaxes(type='log',row=1,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(xaxis2_rangeslider_visible=False)
    fig.update_yaxes(type='log',row=2,col=1)
    shapes = [dict( x0=d1.loc[start_date].name, x1=d1.iloc[-1].name,y0=d1.loc[start_date].High, y1=d1.iloc[-1].High,line_width=2,type='line',xref='x',yref='y')]
    shapes.append(dict( x0=d2.loc[start_date].name, x1=d2.iloc[-1].name,y0=d2.loc[start_date].High, y1=d2.iloc[-1].High,line_width=2,type='line',xref='x2',yref='y2'))
    if(end_date != start_date):
        shapes.append(dict( x0=d1.loc[end_date].name, x1=d1.iloc[-1].name,y0=d1.loc[end_date].High, y1=d1.iloc[-1].High,line_width=2,type='line',xref='x',yref='y'))
        shapes.append(dict( x0=d2.loc[end_date].name, x1=d2.iloc[-1].name,y0=d2.loc[end_date].High, y1=d2.iloc[-1].High,line_width=2,type='line',xref='x2',yref='y2'))
    fig.update_layout(shapes = shapes)
    #fig.show()
    return True,start_date,d1.iloc[-1].name,fig

def IntermarketDivergenceLows(stock1,stock2,startDate = '2010-01-01',endDate = '2100-01-01'):
    d1 = yf.download(stock1,startDate,endDate,interval='1wk')
    d2 = yf.download(stock2,startDate,endDate,interval='1wk')
    if d1.iloc[-1].name - d1.iloc[-2].name < timedelta(7):
        d1.drop(d1.tail(1).index,inplace=True)
    if d2.iloc[-1].name - d2.iloc[-2].name < timedelta(7):
        d2.drop(d2.tail(1).index,inplace=True)

    if d1.iloc[-1].name != d2.iloc[-1].name:
        print(d1.iloc[-1].name,d2.iloc[-1].name)
        print(d1)
        print(d2)
        assert(0)

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
    if starts == -1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.01,subplot_titles=(stock1, stock2))
        fig.add_trace(go.Candlestick(x=d1.index,open=d1['Open'],high=d1['High'],low=d1['Low'],close=d1['Close']),row=1,col=1)
        fig.add_trace(go.Candlestick(x=d2.index,open=d2['Open'],high=d2['High'],low=d2['Low'],close=d2['Close']),row=2,col=1)
        fig.update_yaxes(type='log',row=1,col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(xaxis2_rangeslider_visible=False)
        fig.update_yaxes(type='log',row=2,col=1)
        return False,-1,-1,fig
    start_date = d1.iloc[starts].name
    end_date = d1.iloc[ends].name
    d1 = d1.iloc[ends-3:]
    d2 = d2.iloc[ends-3:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.01,subplot_titles=(stock1, stock2))
    fig.add_trace(go.Candlestick(x=d1.index,open=d1['Open'],high=d1['High'],low=d1['Low'],close=d1['Close']),row=1,col=1)
    fig.add_trace(go.Candlestick(x=d2.index,open=d2['Open'],high=d2['High'],low=d2['Low'],close=d2['Close']),row=2,col=1)
    fig.update_yaxes(type='log',row=1,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(xaxis2_rangeslider_visible=False)
    fig.update_yaxes(type='log',row=2,col=1)
    shapes = [dict( x0=d1.loc[start_date].name, x1=d1.iloc[-1].name,y0=d1.loc[start_date].Low, y1=d1.iloc[-1].Low,line_width=2,type='line',xref='x',yref='y')]
    shapes.append(dict( x0=d2.loc[start_date].name, x1=d2.iloc[-1].name,y0=d2.loc[start_date].Low, y1=d2.iloc[-1].Low,line_width=2,type='line',xref='x2',yref='y2'))
    if(end_date != start_date):
        shapes.append(dict( x0=d1.loc[end_date].name, x1=d1.iloc[-1].name,y0=d1.loc[end_date].Low, y1=d1.iloc[-1].Low,line_width=2,type='line',xref='x',yref='y'))
        shapes.append(dict( x0=d2.loc[end_date].name, x1=d2.iloc[-1].name,y0=d2.loc[end_date].Low, y1=d2.iloc[-1].Low,line_width=2,type='line',xref='x2',yref='y2'))
    fig.update_layout(shapes = shapes)
    #fig.show()
    return True,start_date,d1.iloc[-1].name,fig

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
    s1 = "AAPL"
    s2 = "META"
    found,start,end,fig = IntermarketDivergenceHighs(s1,s2)
    if found:
        print(f'Bearish Divergence Observed between {s1} and {s2}. Observed against weeks from {end} to {start}.')
    s1 = "^GSPC"
    s2 = "^DJI"
    found,start,end,fig = IntermarketDivergenceHighs(s1,s2)
    if found:
        print(f'Bearish Divergence Observed between {s1} and {s2}. Observed against weeks from {end} to {start}.')
    s1 = "^NDX"
    s2 = "^GSPC"
    found,start,end,fig = IntermarketDivergenceHighs(s1,s2)
    if found:
        print(f'Bearish Divergence Observed between {s1} and {s2}. Observed against weeks from {end} to {start}.')


    s1 = "^NDX"
    s2 = "^GSPC"
    found,start,end,fig = IntermarketDivergenceLows(s1,s2)
    if found:
        print(f'Bullish Divergence Observed between {s1} and {s2}. Observed against weeks from {end} to {start}.')
    s1 = "^GSPC"
    s2 = "^DJI"
    found,start,end,fig = IntermarketDivergenceLows(s1,s2)
    if found:
        print(f'Bullish Divergence Observed between {s1} and {s2}. Observed against weeks from {end} to {start}.')
    s1 = "^NDX"
    s2 = "^GSPC"
    found,start,end,fig = IntermarketDivergenceLows(s1,s2)
    if found:
        print(f'Bullish Divergence Observed between {s1} and {s2}. Observed against weeks from {end} to {start}.')
