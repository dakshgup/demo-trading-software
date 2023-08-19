import yfinance_dup as yf
import numpy as np
import plotly.graph_objects as go

def check_for_all_time_high(stock,interval='1d'):
    df = yf.download(stock,interval=interval)
    highs = df['High'].to_numpy()
    curr_max_idx = highs.argmax()
    if(curr_max_idx != len(df)-1):
        shape=[dict(x0=df.iloc[curr_max_idx].name,y0=df.iloc[curr_max_idx].High,x1=df.iloc[-1].name,y1=df.iloc[curr_max_idx].High,type='line',line_width=6)]
        df = df.iloc[curr_max_idx-20:]
        fig = go.Figure(data = [go.Candlestick(x=df.index,open=df['Open'],close=df['Close'],high=df['High'],low=df['Low'])])
        fig.update_yaxes(type='log')
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(shapes=shape)
        return False,fig

    prev_max_idx = highs[:-1].argmax()
    # Check this parameter 5 - It is the parameter to ensure that this is not a new high shortly after new high.
    if prev_max_idx <= len(df)-5:
        print(prev_max_idx,len(df))
        shape=[dict(x0=df.iloc[prev_max_idx].name,y0=df.iloc[prev_max_idx].High,x1=df.iloc[-1].name,y1=df.iloc[prev_max_idx].High,type='line',line_width=6)]
        df = df.iloc[prev_max_idx-20:]
        fig = go.Figure(data = [go.Candlestick(x=df.index,open=df['Open'],close=df['Close'],high=df['High'],low=df['Low'])])
        fig.update_yaxes(type='log')
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(shapes=shape)
        return True,fig
    i = -1
    # This 4 is one less than the 5 above
    while prev_max_idx >= len(df)+i-4:
        i-=1
        prev_max_idx = highs[:i].argmax()

    shape=[dict(x0=df.iloc[prev_max_idx].name,y0=df.iloc[prev_max_idx].High,x1=df.iloc[-1].name,y1=df.iloc[prev_max_idx].High,type='line',line_width=6)]
    df = df.iloc[prev_max_idx-20:]
    fig = go.Figure(data = [go.Candlestick(x=df.index,open=df['Open'],close=df['Close'],high=df['High'],low=df['Low'])])
    fig.update_yaxes(type='log')
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(shapes=shape)
    # print("HERE")
    return False,fig


if __name__ == "__main__":
    ans,fig = check_for_all_time_high('AAPL')
    print(ans)
    fig.show()
