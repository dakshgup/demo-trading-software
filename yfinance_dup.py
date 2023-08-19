import yfinance as yf1
import os.path
from datetime import datetime,timedelta
import pickle

def download(stock,start=None,end=None,interval='1d'):
    if not os.path.isfile(os.path.join(f'temp_data_storage',f'{stock}_{datetime.now().strftime("%Y-%m-%d")}_{interval}')):
        df = yf1.download(stock,interval=interval)
        if interval == '1wk':
            df = yf1.download(stock,interval=interval)
            if df.iloc[-1].name - df.iloc[-2].name < timedelta(5):
                df.drop(df.tail(1).index,inplace=True)
        print(os.path.join(f'temp_data_storage',f'{stock}_{datetime.now().strftime("%Y-%m-%d")}_{interval}'))
        with open(os.path.join(f'temp_data_storage',f'{stock}_{datetime.now().strftime("%Y-%m-%d")}_{interval}'),'wb') as f:
            pickle.dump(df,f)

    df = None
    with open(os.path.join('temp_data_storage',f'{stock}_{datetime.now().strftime("%Y-%m-%d")}_{interval}'),'rb') as f:
        df = pickle.load(f)
    # print(type(start),start)
    # print(type(end),end)
    if start is None and end is None:
        return df
    if start is None:
        return df.loc[:end]
    if end is None:
        return df.loc[start:]
    if not isinstance(start,str):
        start = start.strftime('%Y-%m-%d')
    if not isinstance(end,str):
        end = end.strftime('%Y-%m-%d')
    # print(type(start),start)
    # print(type(end),end)
    return df.loc[start:end]
