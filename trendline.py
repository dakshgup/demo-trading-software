import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.subplots 
from plotly.subplots import make_subplots
from scipy import stats
import yfinance as yf
from plotly.subplots import make_subplots
from IPython.display import display
def is_pivot(candle, window, df):
    if candle - window < 0 or candle + window >= len(df):
        return 0
    
    pivot_high = 1
    pivot_low = 2
    
    for i in range(candle - window, candle + window+1 ):
        if df.iloc[candle].Low > df.iloc[i].Low:
            pivot_low = 0
        if df.iloc[candle].High < df.iloc[i].High:
            pivot_high = 0
    
    if pivot_high and pivot_low:
        return 3
    elif pivot_high:
        return pivot_high
    elif pivot_low:
        return pivot_low
    else:
        return 0

def calculate_point_pos(row):
    if row['isPivot'] == 2:
        return row['Low'] - 1e-3
    elif row['isPivot'] == 1:
        return row['High'] + 1e-3
    else:
        return np.nan

def collect_channel(candle, backcandles, window, df):
    best_r_squared_low = 0
    best_r_squared_high = 0
    best_slope_low = 0
    best_intercept_low = 0
    best_slope_high = 0
    best_intercept_high = 0
    best_backcandles_low = 0
    best_backcandles_high = 0
    
    for i in range(backcandles-backcandles//2, backcandles + backcandles//2,window):
        local_df = df[candle - i - window : candle - window ]
        #for j in range(candle - i - window, candle - 2 * window):
          #local_df.loc[j, 'isPivot'] = is_pivot(j, window, local_df)
        lows = local_df[local_df['isPivot'] == 2].Low.values[-4:]
        idx_lows = local_df[local_df['isPivot'] == 2].Low.index[-4:]
        highs = local_df[local_df['isPivot'] == 1].High.values[-4:]
        idx_highs = local_df[local_df['isPivot'] == 1].High.index[-4:]
    
        if len(lows) >= 2:
            slope_low, intercept_low, r_value_l, _, _ = stats.linregress(idx_lows, lows)
            if (r_value_l ** 2)*len(lows)> best_r_squared_low and (r_value_l ** 2) > 0.85:
                best_r_squared_low = (r_value_l ** 2)*len(lows)
                best_slope_low = slope_low
                best_intercept_low = intercept_low
                best_backcandles_low = i
        
        if len(highs) >= 2:
            slope_high, intercept_high, r_value_h, _, _ = stats.linregress(idx_highs, highs)
            if (r_value_h ** 2)*len(highs) > best_r_squared_high and (r_value_h ** 2)> 0.85:
                best_r_squared_high = (r_value_h ** 2)*len(highs)
                best_slope_high = slope_high
                best_intercept_high = intercept_high
                best_backcandles_high = i
    
    return best_backcandles_low, best_slope_low, best_intercept_low, best_r_squared_low, best_backcandles_high, best_slope_high, best_intercept_high, best_r_squared_high

def is_breakout(candle, backcandles, window, df,stop_percentage):
    if 'isBreakOut' not in df.columns:
        return 0
    #here we avoid taking trade if we took just on previous candle
    for i in range(1,2):
        if df['isBreakOut'][candle - i] != 0:
            return 0
    if candle - backcandles - window < 0:
        return 0

    best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(candle, backcandles, window, df)
    
    thirdback = candle-2
    thirdback_low= df.iloc[thirdback].Low
    thirdback_high= df.iloc[thirdback].High
    thirdback_volume= df.iloc[thirdback].Volume

    prev_idx = candle - 1
    prev_high = df.iloc[prev_idx].High
    prev_low = df.iloc[prev_idx].Low
    prev_close = df.iloc[prev_idx].Close
    prev_open = df.iloc[prev_idx].Open
    
    curr_idx = candle
    curr_high = df.iloc[curr_idx].High
    curr_low = df.iloc[curr_idx].Low
    curr_close = df.iloc[curr_idx].Close
    curr_open = df.iloc[curr_idx].Open
    curr_volume= max(df.iloc[candle].Volume,df.iloc[candle-1].Volume)
    breakpclow = (sl_lows * prev_idx + interc_lows  - curr_low)/curr_open
    breakpchigh = (curr_high - sl_highs * prev_idx - interc_highs)/curr_open

    if ( 
        thirdback_high > sl_lows * thirdback + interc_lows and
        #breakpclow >= stop_percentage/3 and
        curr_volume >thirdback_volume and
        prev_close < prev_open and
        curr_close < curr_open and
        sl_lows > 0 and
        prev_close < sl_lows * prev_idx + interc_lows and
        #curr_open < sl_lows * curr_idx + interc_lows and
        curr_close < sl_lows * prev_idx + interc_lows):
        return 1
    
    elif (  
        thirdback_low < sl_highs * thirdback + interc_highs and
        curr_volume >thirdback_volume and
        #breakpchigh >= stop_percentage/3 and
        prev_close > prev_open and 
        curr_close > curr_open and
        sl_highs < 0 and
        prev_close > sl_highs * prev_idx + interc_highs and
        #curr_open > sl_highs * curr_idx + interc_highs and
        curr_close > sl_highs * prev_idx + interc_highs):
        return 2
    
    else:
        return 0

def calculate_breakpoint_pos(row):
    if row['isBreakOut'] == 2:
        return row['Low'] - 3e-3
    elif row['isBreakOut'] == 1:
        return row['High'] + 3e-3
    else:
        return np.nan
def exportcsv(df, ticker, start_date, end_date, interval, profit_percentage, stop_percentage,level):
    window=3*level
    backcandles=10*window
    trades = []
    header_data = pd.DataFrame({'Ticker': [ticker],
                                'Start Date': [start_date],
                                'End Date': [end_date],
                                'Interval': [interval]})
    for i in range(1, len(df)):
        signal_type = df['isBreakOut'].iloc[i]
        signal=""
        if signal_type == 2:
            signal="long"
            entry_date = pd.to_datetime(df['Date'].iloc[i])
            entry_price = df['High'].iloc[i]
            exit_price = None
            for j in range(i + 1, len(df)):
                if df['High'].iloc[j] >= entry_price * (1 + profit_percentage):
                    exit_date = pd.to_datetime(df['Date'].iloc[j])
                    exit_price = entry_price * (1 + profit_percentage)
                    exit_price = df['High'].iloc[j]
                    break
                elif df['Low'].iloc[j] <= entry_price * (1 - stop_percentage):
                    exit_date = pd.to_datetime(df['Date'].iloc[j])
                    exit_price = entry_price * (1 - stop_percentage)
                    exit_price = df['Low'].iloc[j]
                    break
            if exit_price is None:
                exit_date = pd.to_datetime(df['Date'].iloc[-1])
                exit_price = df['Close'].iloc[-1]
            profit_or_stopped = calculate_profit_or_stopped(entry_price, exit_price, signal_type)
            trades.append((entry_date, entry_price, exit_date, exit_price, profit_or_stopped,signal,level))
        elif signal_type == 1:
            signal="short"
            entry_date = pd.to_datetime(df['Date'].iloc[i])
            entry_price = df['Close'].iloc[i]
            exit_price = None
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= entry_price * (1 - profit_percentage):
                    exit_date = pd.to_datetime(df['Date'].iloc[j])
                    exit_price = entry_price * (1 - profit_percentage)
                    exit_price = df['Low'].iloc[j]
                    break
                elif df['High'].iloc[j] >= entry_price * (1 + stop_percentage):
                    exit_date = pd.to_datetime(df['Date'].iloc[j])
                    exit_price = entry_price * (1 + stop_percentage)
                    exit_price = df['High'].iloc[j]
                    break
            if exit_price is None:
                exit_date = pd.to_datetime(df['Date'].iloc[-1])
                exit_price = df['Close'].iloc[-1]
            profit_or_stopped = calculate_profit_or_stopped(entry_price, exit_price, signal_type)
            trades.append((entry_date, entry_price, exit_date, exit_price, profit_or_stopped,signal,level))
    
    if len(trades) > 0:
        success_count = sum([1 for trade in trades if trade[4] == 1])
        success_rate = success_count / len(trades) * 100
    else:
        success_rate = 0.0
    header_data['Success Rate'] = success_rate
    trade_data = pd.DataFrame(trades, columns=['Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'Profit/Loss','Signal type','Level'])
    trade_data['Return'] = trade_data['Profit/Loss'] * profit_percentage - (1 - trade_data['Profit/Loss']) * stop_percentage
    trade_data = pd.concat([header_data, trade_data], ignore_index=True)
   
    #filename = f"trade_{ticker}_{interval}_{profit_percentage}.csv"
    #trade_data.to_csv(filename, index=False)
    print(window, backcandles,len(trades),success_rate)
    return trade_data

def calculate_exit_date(entry_date, interval):
    if interval == "1D":
        return entry_date + pd.Timedelta(days=1)
    elif interval == "1wk":
        return entry_date + pd.Timedelta(weeks=1)
    elif interval == "1mo":
        return entry_date + pd.DateOffset(months=1)

def calculate_profit_or_stopped(entry_price, exit_price, long_or_short):
  if long_or_short==2:
    if exit_price >= entry_price :
        return 1
    else:
        return 0
  elif long_or_short==1:
    if exit_price <= entry_price :
        return 1
    else:
        return 0

def calculate_combined_trades(ticker, start_date, end_date, interval):
    # Initialize an empty DataFrame to store the combined trade data
    combined_trades = pd.DataFrame()
    st=2000-27-12
    # Iterate through levels 1 to 10
    for level in range(2, 11,2):
        # Obtain the DataFrame for the current level
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        data.to_csv("Edata.csv")
        df = pd.read_csv("Edata.csv")
        window = 3 * level
        backcandles = 10 * window
        data1 = yf.download(ticker,start=st,end=end_date, interval=interval, progress=False)
        data1.to_csv("Edata1.csv")
        df1 = pd.read_csv("Edata1.csv")
        filtered_df = df1[(df1['Date'] >= start_date) & (df1['Date'] <= end_date)]
        atr = ta.atr(high=filtered_df['High'], low=filtered_df['Low'], close=filtered_df['Close'], length=14)
        atr_multiplier = 2 
        stop_percentage = atr.iloc[-1] * atr_multiplier / filtered_df['Close'].iloc[-1]
        profit_percentage = (1 + (level - 1) / 4) * stop_percentage
        print(f"stoploss: {stop_percentage:.2f}%_takeprofit: {profit_percentage}%")
        df['isPivot'] = df.apply(lambda row: is_pivot(row.name, window, df), axis=1)
        df['isBreakOut'] = 0
        for i in range(backcandles + window, len(df)):
            df.loc[i, 'isBreakOut'] = is_breakout(i, backcandles, window, df, stop_percentage)
        trades_data = exportcsv(df, ticker, start_date, end_date, interval, profit_percentage, stop_percentage, level)
        combined_trades = pd.concat([combined_trades, trades_data])

    combined_trades = combined_trades.sort_values(by='Entry Date')

    combined_trades = combined_trades.drop_duplicates(subset=['Entry Date', 'Exit Date'], keep='first')

    total_trades = len(combined_trades)
    profitable_trades = len(combined_trades[combined_trades['Profit/Loss'] > 0])
    success_rate = (profitable_trades / total_trades) * 100

    valid_trades = combined_trades.dropna(subset=['Return']).copy()

    valid_trades['Cumulative Return'] = (1 + valid_trades['Return'] / 100).cumprod()

    overall_return = (valid_trades['Cumulative Return'].iloc[-1] - 1) * 100

    print(f"Overall Return: {overall_return:.2f}%")
    print(f"Success Rate: {success_rate:.2f}%")

    filename = f"{ticker}_{interval}_{success_rate:.2f}_{overall_return*100:.1f}%.csv"
    combined_trades.to_csv(filename, index=False)
    return combined_trades,success_rate,overall_return

def plot_stock_data(ticker, start_date, end_date, interval, level=2):
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=True)
    data.to_csv("Edata.csv")
    df = pd.read_csv("Edata.csv")
    df_pl = df
    window = 3 * level
    backcandles = 10 * window

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Calculate ATR using pandas_ta
    atr = ta.atr(high=filtered_df['High'], low=filtered_df['Low'], close=filtered_df['Close'], length=14)

    atr_multiplier = 2
    stop_percentage = atr.iloc[-1] * atr_multiplier / filtered_df['Close'].iloc[-1]
    profit_percentage = (1 + (level - 1) / 4) * stop_percentage
    print(f"stoploss: {stop_percentage:.2f}%_takeprofit: {profit_percentage:.2f}%")

    df_pl['Date'] = pd.to_datetime(df_pl['Date'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(ticker, 'Volume'), row_width=[0.1, 0.7])

    fig.add_trace(
        go.Candlestick(
            hovertext=df_pl['Date'].dt.strftime('%Y-%m-%d'),
            x=df_pl.index,
            open=df_pl['Open'],
            high=df_pl['High'],
            low=df_pl['Low'],
            close=df_pl['Close'],
            name="Candlestick"
        ),
        row=1, col=1
    )

    df_pl['isPivot'] = df_pl.apply(lambda row: is_pivot(row.name, window, df_pl), axis=1)
    df_pl['pointpos'] = df_pl.apply(calculate_point_pos, axis=1)

    fig.add_trace(
        go.Scatter(
            x=df_pl.index,
            y=df_pl['pointpos'],
            mode="markers",
            marker=dict(size=10, color="MediumPurple"),
            name="Pivot"
        ),
        row=1, col=1
    )

    df['isBreakOut'] = 0
    for i in range(backcandles + window, len(df)):
        df.loc[i, 'isBreakOut'] = is_breakout(i, backcandles, window, df, stop_percentage)

    df['breakpointpos'] = df.apply(calculate_breakpoint_pos, axis=1)
    df_breakout = df[df['isBreakOut'] != 0]

    for candle in range(backcandles + window, len(df)):
        if df.iloc[candle].isBreakOut != 0:
            best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(
                candle, backcandles, window, df)
            extended_x = np.array(range(candle + 1, candle + 15))
            x1 = np.array(range(candle - best_back_l - window, candle + 1))
            x2 = np.array(range(candle - best_back_h - window, candle + 1))

            if r_sq_l >= 0.80 and df.iloc[candle].isBreakOut == 1:
                extended_y_lows = sl_lows * extended_x + interc_lows
                fig.add_trace(
                    go.Scatter(
                        x=extended_x,
                        y=extended_y_lows,
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Lower Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x1,
                        y=sl_lows * x1 + interc_lows,
                        mode='lines',
                        name='Lower Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )

            if r_sq_h >= 0.80 and df.iloc[candle].isBreakOut == 2:
                extended_y_highs = sl_highs * extended_x + interc_highs
                fig.add_trace(
                    go.Scatter(
                        x=extended_x,
                        y=extended_y_highs,
                        mode='lines',
                        line=dict(dash='dash'),
                        name='Max Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x2,
                        y=sl_highs * x2 + interc_highs,
                        mode='lines',
                        name='Max Slope',
                        showlegend=False
                    ),
                    row=1, col=1
                )

    colors = ['black', 'blue']
    for color in colors:
        mask = df_breakout['isBreakOut'].map({1: 'black', 2: 'blue'}) == color
        fig.add_trace(
            go.Scatter(
                x=df_breakout.index[mask],
                y=df_breakout['breakpointpos'][mask],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color
                ),
                marker_symbol="hexagram",
                name="Breakout",
                legendgroup=color
            ),
            row=1, col=1
        )

    fig.update_traces(
        name="Break Below",
        selector=dict(legendgroup="black")
    )

    fig.update_traces(
        name="Break Above",
        selector=dict(legendgroup="blue")
    )

    fig.add_trace(
        go.Bar(
            x=df_pl['Date'],
            y=df_pl['Volume'],
            marker_color='red',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        title_text=interval,
        xaxis_range=[df_pl.index.min(), df_pl.index.max()],
        yaxis_range=[df_pl['Low'].min(), df_pl['High'].max()],
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(rangemode='tozero'),
    )

    fig.show()
