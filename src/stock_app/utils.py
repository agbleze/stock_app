
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.core.indexes.multi import MultiIndex
import functools
from prophet.plot import get_seasonality_plotly_props
from prophet import Prophet
import plotly.graph_objects as go
from prophet.serialize import model_to_json, model_from_json
import json
import pandas_market_calendars as mcal
import math
import plotly.express as px


def get_market_type_data(df, market_type):
    if market_type not in ["premarket", "regular", "afterhrs"]:
        raise ValueError("Invalid market type. Use 'premarket', 'afterhrs', or'regular'.")
    market_open = pd.Timestamp("09:30", tz="US/Eastern").time()
    market_close = pd.Timestamp("16:00", tz="US/Eastern").time()
    
    if market_type == "premarket":
        df = df[(df.index.time <= market_open)]
    elif market_type == "regular":
        open_marktime_list = df[(df.index.time >= market_open)].index.to_list()
        reguhr = [item for item in open_marktime_list if item.time() <= market_close]
        df = df[df.index.isin(reguhr)]
    elif market_type == "afterhrs":
        df = df[df.index.time >= market_close]
        
    return df

def cal_proba_regular_lowest_in_after_hours(df, 
                                            percent_to_reduce_regular_lowest_price=0,
                                            target_col="Close"
                                            ):
    case_count = 0
    all_count = 0
    case_date = []
    reg_df = get_market_type_data(df=df, market_type="regular") 
    afterhr_df = get_market_type_data(df=df, market_type="afterhrs")
    unique_date = np.unique(df.index.date)
    
    for item in unique_date:
        day_reguhr = reg_df[reg_df.index.date == item]
        day_afterhr = afterhr_df[afterhr_df.index.date == item]
        day_reguhr_Lowmin = day_reguhr[target_col].min()
        day_reguhr_Lowmin = (((100 - percent_to_reduce_regular_lowest_price) / 100)
                            * day_reguhr_Lowmin
                            )
        day_afterhr_Lowmin = day_afterhr[target_col].min()
        if day_afterhr_Lowmin <= day_reguhr_Lowmin:
            print(f"day_afterhr_Lowmin : {day_afterhr_Lowmin}")
            print(f"day_reguhr_Lowmin : {day_reguhr_Lowmin}")
            case_count += 1
            all_count += 1
            case_date.append(item)
        else:
            all_count += 1
    if case_count > 0:
        proba = (case_count / all_count) * 100
    else:
        proba = 0.0
    return {"probability": proba,
            "case_date": case_date
            }
    

#%% TODO: Add plots with horizontal lines  showing the lowest price
# in regular hours and a vertical line showing when it went long 
# in after hours and another vertical for sell time
def buy_afterhrs_at_regular_lowest(df, profit_percent=1,
                                   percent_to_reduce_regular_lowest_price=0,
                                   target_col="Close"
                                   ):
    """Estimate the scenario of buying in the after hours at the 

    Args:
        df (_type_): _description_
        profit_percent (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    reg_df = get_market_type_data(df=df, market_type="regular") 
    unique_date = np.unique(df.index.date)
    afterhr_df = get_market_type_data(df=df, market_type="afterhrs")
    buy_price_list = []
    buy_day_list = []
    sell_price_list = []
    sell_day_list = []
    profit_lose_list = []
    profit_lose_percent_list = []
    for item in unique_date:
        day_reguhr = reg_df[reg_df.index.date == item]
        day_afterhr = afterhr_df[afterhr_df.index.date == item]
        day_reguhr_Lowmin = day_reguhr[target_col].min()
        if percent_to_reduce_regular_lowest_price > 0:
            day_reguhr_Lowmin = ((100 - float(percent_to_reduce_regular_lowest_price) / 100)
                                    * day_reguhr_Lowmin
                                    )
        enter_post = False
        buy_price = 0
        exit_price = 0
        exit_post = False
        for row_index, row_data in day_afterhr.iterrows():
            if not enter_post:
                if row_data[target_col] <= day_reguhr_Lowmin:
                    if row_index == day_afterhr.index[-1]:
                        print(f"Not bought because it is last time of after hours {row_index}")
                    else:
                        buy_price = row_data[target_col]
                        buy_price_list.append(buy_price)
                        buy_day_list.append(row_index)
                        enter_post = True
                        exit_price = ((100 + profit_percent)/100) * buy_price
            elif enter_post:
                if not exit_post:
                    if row_data[target_col] >= exit_price:
                        sell_price = row_data[target_col]
                        sell_price_list.append(sell_price)
                        sell_day_list.append(row_index)
                        profit_lose = sell_price - buy_price
                        profit_lose_list.append(profit_lose)
                        exit_post = True
                        profit = ((sell_price - buy_price)/buy_price) * 100
                        profit_lose_percent_list.append(profit)
                    elif row_index == day_afterhr.index[-1]:
                        sell_price = row_data[target_col]
                        sell_price_list.append(sell_price)
                        sell_day_list.append(row_index)
                        profit_lose = sell_price - buy_price
                        profit_lose_list.append(profit_lose)
                        enter_post = False
                        profit = ((sell_price - buy_price)/buy_price) * 100
                        profit_lose_percent_list.append(profit)
    return {"buy_price_list": buy_price_list,
            "buy_day_list": buy_day_list,
            "sell_price_list": sell_price_list,
            "sell_day_list": sell_day_list,
            "profit_lose_list": profit_lose_list,
            "profit_lose_percent_list": profit_lose_percent_list
            }          
                   

#%%
def buy_regular_at_premarket_lowest(df, profit_percent=1, 
                                    percent_to_reduce_premarket_lowest=0,
                                    target_col="Close"
                                    ):
    reg_df = get_market_type_data(df=df, market_type="regular")
    unique_date = np.unique(df.index.date)
    
    premarket_df = get_market_type_data(df=df, market_type="premarket")
    buy_price_list = []
    buy_day_list = []
    sell_price_list = []
    sell_day_list = []
    profit_lose_list = []
    profit_lose_percent_list = []
    
    for item in unique_date:
        day_reguhr = reg_df[reg_df.index.date == item]
        day_premarket = premarket_df[premarket_df.index.date == item]
        day_premarket_Lowmin = day_premarket[target_col].min()
        
        if percent_to_reduce_premarket_lowest > 0:
            day_premarket_Lowmin = ((100 - float(percent_to_reduce_premarket_lowest) / 100)
                                    * day_premarket_Lowmin
                                    )
            
        enter_post = False
        buy_price = 0
        exit_price = 0
        exit_post = False
        for row_index, row_data in day_reguhr.iterrows():
            if not enter_post:
                if row_data[target_col] <= day_premarket_Lowmin:
                    buy_price = row_data[target_col]
                    buy_price_list.append(buy_price)
                    buy_day_list.append(row_index)
                    enter_post = True
                    exit_price = ((100 + profit_percent)/100) * buy_price
            elif enter_post:
                if not exit_post:
                    if row_data[target_col] >= exit_price:
                        sell_price = row_data[target_col]
                        sell_price_list.append(sell_price)
                        sell_day_list.append(row_index)
                        profit_lose = sell_price - buy_price
                        profit_lose_list.append(profit_lose)
                        exit_post = True
                        profit = ((sell_price - buy_price)/buy_price) * 100
                        profit_lose_percent_list.append(profit)
                    elif row_index == day_reguhr.index[-1]:
                        sell_price = row_data[target_col]
                        sell_price_list.append(sell_price)
                        sell_day_list.append(row_index)
                        profit_lose = sell_price - buy_price
                        profit_lose_list.append(profit_lose)
                        enter_post = False
                        profit = ((sell_price - buy_price)/buy_price) * 100
                        profit_lose_percent_list.append(profit)
    return {"buy_price_list": buy_price_list,
            "buy_day_list": buy_day_list,
            "sell_price_list": sell_price_list,
            "sell_day_list": sell_day_list,
            "profit_lose_list": profit_lose_list,
            "profit_lose_percent_list": profit_lose_percent_list
            }          
                
    
def cal_proba_premarket_low_in_regular_hr(df, percent_to_reduce_premarket_lowest=0,
                                          target_col="Close"
                                          ):
    case_count = 0
    all_count = 0
    case_date = []
    reg_df = get_market_type_data(df=df, market_type="regular")
    unique_date = np.unique(df.index.date)
    premarket_hr_df = get_market_type_data(df=df, market_type="premarket")
    for item in unique_date:
        curr_regular_df = reg_df[reg_df.index.date == item]
        curr_premarket_df = premarket_hr_df[premarket_hr_df.index.date == item]
        curr_regular_lowmin = curr_regular_df[target_col].min()
        curr_premarket_lowmin = curr_premarket_df[target_col].min()
        if percent_to_reduce_premarket_lowest > 0:
            curr_premarket_lowmin = (((100 - percent_to_reduce_premarket_lowest) / 100)
                                    * curr_premarket_lowmin
                                    )
        if curr_premarket_lowmin >= curr_regular_lowmin:
            print(f"current premarket lowest : {curr_premarket_lowmin}")
            print(f"day_reguhr_Lowmin : {curr_regular_lowmin}")
            case_count += 1
            all_count += 1
            case_date.append(item)
        else:
            all_count += 1
    if case_count > 0:
        proba = (case_count / all_count) * 100
    else:
        proba = 0.0
    return {"probability": proba,
            "case_date": case_date
            }
    
def calculate_proba_close_lower_than_nextday_high_conditional(df):
    """if current high, low and close are higher than previous than buy at close and 
        sell next day at profit

    Args:
        df (DataFrame): _description_

    Returns:
        Dict: _description_
    """
    
    higher_high = 0
    all_comp = 0
    profit_percent = 0
    profit_percent_scores = []
    curr_closeprice_higher_thn_nextday_high_samples = []
    for rowdata_index, row_data in df.iterrows():
        curr_high = row_data["High"]
        curr_low = row_data["Low"]
        curr_close = row_data["Close"]
        
        curr_close_price = row_data["Close"]
        if rowdata_index != df.index[-1]:
            previous_day_high = df[df.index <= rowdata_index].tail(2).iloc[0]["High"]
            previous_day_low = df[df.index <= rowdata_index].tail(2).iloc[0]["Low"]
            previous_day_close = df[df.index <= rowdata_index].tail(2).iloc[0]["Close"]
            
            if ((curr_high > previous_day_high) and (curr_low > previous_day_low) 
                and (curr_close > previous_day_close)
                ):
                
                nextday_high = df[df.index >= rowdata_index].iloc[1]["High"]
                if nextday_high > curr_close:
                    higher_high += 1
                    profit = ((nextday_high / curr_close_price) * 100) -100
                    profit_percent += profit
                    profit_percent_scores.append(profit)
                        
                    all_comp += 1
                else:
                    all_comp += 1
                    curr_and_nextday_df = df[df.index >= rowdata_index].iloc[0:2]
                    curr_closeprice_higher_thn_nextday_high_samples.append(curr_and_nextday_df)
        else:
            print(f"last day: {rowdata_index}")

    if higher_high == 0:
        prob = 0
    else:
        prob = (higher_high / all_comp) * 100
    return {"probability": prob, 
            "profit_percent": profit_percent,
            "profit_percent_scores": profit_percent_scores,
            "total_instances": all_comp,
            "num_observations": higher_high,
            "curr_closeprice_higher_thn_nextday_high_samples": curr_closeprice_higher_thn_nextday_high_samples
            }



def download_stock_price(stock_ticker, start_date=None, end_date=None, **kwargs):
    data = yf.download(stock_ticker, start=start_date, end=end_date, **kwargs)
    if isinstance(data.columns, MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data


def cal_proba_current_close_is_lower_than_nextday_high(df):
    case_count = 0
    event_count = 0
    percent_lower = 0
    profit_percent_scores = []
    for rowdata_index, rowdata in df.iterrows():
        curr_close_price = rowdata["Close"]
        if rowdata_index != df.index[-1]:
            nextday_high = df[df.index >= rowdata_index].iloc[1]["High"]
            if nextday_high > curr_close_price:
                case_count += 1
                percent = ((nextday_high / curr_close_price) * 100) -100
                percent_lower += percent
                profit_percent_scores.append(percent)
                    
                event_count += 1
            else:
                event_count += 1
        else:
            print(f"last day: {rowdata_index}")

    prob = (case_count / event_count) * 100
    return {"probability": prob, 
            "percent_lower": percent_lower, # profit_percent
            "percent_lower_scores": profit_percent_scores,  # profit_percent_scores
            "total_instances": event_count
            }

def calculate_prob_close_lower_than_open(df):
    """Calculate the probability that the stock closes lower than 
        the open price

    Args:
        df (_type_): data in the format of stock price data with columns
                    Close, Open, High, Low

    Returns:
        Dict: keys are total_samples (all samples in the data), probability, loss_percent_list (for instances 
                where close was lower than open price), loss_percent (total 
                loss percentage)
    """
    proba = 0
    all_comp = 0
    loss_percent_list = []
    loss = 0
    for row_index, row_data in df.iterrows():
        if row_data["Close"] < row_data["Open"]:
            proba += 1
            all_comp += 1
            loss_score = (row_data["Close"] / row_data["Open"]) * 100
            loss_res = loss_score - 100
            loss_percent_list.append(loss_res)
            loss_percent = loss - loss_res
        else:
            all_comp += 1
            
    return {"total_samples": all_comp, "loss_percent_list": loss_percent_list,
            "probability": (proba / all_comp) * 100,
            "loss_percent": sum(loss_percent_list)
            }
        

def calculate_price_change(data, col="Close"):
    data["pct_change"] = data[col].pct_change()
    data["pct_change"] = round(data["pct_change"]*100, 2)
    data["diff"] = data[col].diff()
    data["status_rise"] = data["diff"].map(lambda x: x > 0)
    data["color"] = ["red" if x == False else "green" for x in data[["status_rise"]].values]
    return data


def calculate_proba_close_higher_than_nextday_low(df):
    """

    Args:
        df (DataFrame): _description_

    Returns:
        Dict: Keys of dict are probability, percent_higher_scores, total_instances
    """
    count_case = 0
    all_cases = 0 
    percent_higher_scores = []
    for rowdata_index, row_data in df.iterrows():
        curr_close = row_data["Close"]
        if rowdata_index != df.index[-1]:
            next_day = df[df.index >= rowdata_index].head(2).iloc[-1]
            next_day_low = next_day["Low"]
            if curr_close > next_day_low:
                count_case += 1
                all_cases += 1
                percent = ((curr_close / next_day_low) * 100) -100
                percent_higher_scores.append(percent)
            else:
                all_cases += 1
        else:
            print(f"last day: {rowdata_index}")
    if count_case == 0:
        prob = 0
    else:
        prob = (count_case / all_cases) * 100
    return {"probability": prob,
            "percent_higher_scores": percent_higher_scores,
            "total_instances": all_cases
            }
  

def low_open_diff(df):
    """Calculates the percentage difference between Low and Open price

    Args:
        df (DataFrame): Data in the format of stock price with columns Open,
                        Low, High and Close price

    Returns:
        DataFrame: A DataFrame with low_open_pct_change column added to existing columns
    """
    df["low_open_pct_change"] = ((df["Low"] - df["Open"])/df["Open"]) * 100
    return df


# Analysis of worst case scenario of O-H-LC
def cal_proba_close_eq_low(df):
    """Calculates the probability that the Close price was the Lowest 
       price.

    Args:
        df (_type_): Data in the format of stock prices with Open, Low, High and Close

    Returns:
        Dict: Keys are probability cases (positives, list of data row)
    """
    close_eq_low = []
    count = 0
    num_all_samples = 0
    for row_index, row_data in df.iterrows():
        if row_data["Low"]==row_data["Close"]:
            close_eq_low.append(row_data)
            count += 1
            num_all_samples += 1
        else:
            num_all_samples += 1
    proba = (count / num_all_samples) * 100
    
    return {"probability": proba, 
     "cases": close_eq_low
    }
        


def high_low_diff(df):
    """Calculate the percentage difference between the High and Low

    Args:
        df (DataFrame): Data in the format of stock price with columns Open, Low, High and
                        Close

    Returns:
        DataFrame: Adds the column high_low_pct_change to the existing columns
    """
    df["high_low_pct_change"] = ((df["High"] - df["Low"])/df["Low"]) * 100
    return df


def close_open_diff(df):
    """Calculate the percentage change between Open and Close (Close - Open)

    Args:
        df (DataFrame): Data in the format with columns Open, Close, High and Low

    Returns:
        DataFrame: Adds a column close_open_pct_change to existing columns
    """
    df["close_open_pct_change"] = ((df["Close"] - df["Open"])/df["Open"]) * 100
    return df

def plot_column_chart(data, y="Close", hover_data=["pct_change"],
                      marker_color="color"
                      ):
    fig = px.bar(data_frame=data, x=data.index, y=y,
                template="plotly_dark",
                #width=1800,
                height=800,
                hover_data=hover_data
                # color="status_rise"
                )
    fig.update_traces(marker_color=data[marker_color].to_list())
    return fig


def days_to_double(principal, daily_rate):
    """
    Calculate the number of days required to double the principal with a given daily compounded interest rate.
    
    Parameters:
    - principal (float): The initial principal amount.
    - daily_rate (float): The daily interest rate (e.g., 0.01 for 1%).
    
    Returns:
    - int: The number of days required to double the principal.
    """
    # Calculate the target amount (double the principal)
    target_amount = principal * 2
    
    # Calculate the number of days required to double the principal
    days = math.log(2) / math.log(1 + daily_rate)
    
    return int(days)

def cal_lowest_percent_target_is_below_base_market(df, base_market = "premarket",
                                                    target_market="regular"
                                                    ):
    res= {}
    
    if base_market not in ["premarket", "regular"]:
        raise ValueError("Invalid base market type. Use 'premarket' or 'regular'.")
    
    if target_market not in ["premarket", "regular"]:
        raise ValueError("Invalid target market type. Use 'premarket', 'afterhrs', or'regular'.")
    
    base_market_df = get_market_type_data(df=df, market_type=base_market) 
    target_market_df = get_market_type_data(df=df, market_type=target_market)
        
    unique_date = np.unique(df.index.date)

    for day in unique_date:
        day_target_df = target_market_df[target_market_df.index.date==day]
        day_base_df = base_market_df[base_market_df.index.date==day]
        day_target_df_min = day_target_df["Low"].min()
        day_base_df_min = day_base_df["Low"].min()
        if day_target_df_min < day_base_df_min:
            low_percent =  (((day_base_df_min - day_target_df_min)
                             /day_base_df_min) * 100
                            )
            res[day] = {f'percent_lower': {low_percent}
                        }
    res["Note"] = f"""{target_market} is less than the {base_market} by the percentages indicated whenever
                    {target_market} goes below {base_market}"""
    return res
        
        

def get_trading_dates(year, exchange="NYSE"):
    """Get list of trading dates

    Args:
        year (int): Year to get trading dates for
        exchange (str, optional): Name of Stock Exchange . Defaults to "NYSE".

    Returns:
        List: List of trading dates
    """
    stock_exchange = mcal.get_calendar(exchange)
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    schedule = stock_exchange.schedule(start_date=start_date, end_date=end_date)
    trading_dates = schedule.index
    trading_dates_list = trading_dates.tolist()
    return trading_dates_list

def compounded_amount(principal, daily_rate, num_trades):
    """
    Calculate the amount of money accumulated after a given number of trades with daily compounded interest.
    
    Parameters:
    - principal (float): The initial principal amount.
    - daily_rate (float): The daily interest rate (e.g., 0.01 for 1%).
    - num_trades (int): The number of trades (days) over which the interest is compounded.
    
    Returns:
    - float: The amount of money accumulated after the given number of trades.
    """
    accumulated_amount = principal * (1 + daily_rate) ** num_trades
    
    return accumulated_amount

def create_new_lows(df, target_col="Close"):
    new_minimums = []
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame")
    
    current_min = float('inf')
    for index, row in df.iterrows():
        if row[target_col] < current_min:
            current_min = row[target_col]
        new_minimums.append(current_min)
    df['new_minimum'] = new_minimums
    return df

def create_new_highs(df, target_col="Close"):
    new_maximums = []
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame")
    
    current_max = float('-inf')
    for index, row in df.iterrows():
        if row[target_col] > current_max:
            current_max = row[target_col]
        new_maximums.append(current_max)
    df['new_maximum'] = new_maximums
    return df
 
def plot_new_lows_highs(df, title: str):
    unique_date = np.unique(df.index.date)
    for day in unique_date:
        df = df[df.index.date==day]
        df = create_new_lows(df=df) 
        df = create_new_highs(df=df)
        fig = px.line(df,
                        x=df.index,
                        y=['Close', 'new_minimum', "new_maximum"],
                        title=f'{title}: {day}',
                        labels={'value': 'Price', 'variable': 'Metric'},
                    )
        fig.show()
        

def download_minute_interval_data(ticker, start_date=None, end_date=None, 
                                  include_premarket_afterhours=True
                                  ):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date,
                            prepost=include_premarket_afterhours,
                            interval='1m', 
                            period='8d',
                            )
    return data

def get_time_of_event_in_market_type(df, event="Highest", market_type="regular"):
    time_of_event_price = []
    df = get_market_type_data(df=df, market_type=market_type) 
    unique_date = np.unique(df.index.date)
    
    for day in unique_date:
        day_df = df[df.index.date==day]
        if event == "Highest":
            event_df = day_df[day_df["High"] == day_df["High"].max()]
        elif event == "Lowest":
            event_df = day_df[day_df["Low"] == day_df["Low"].min()]
        else:
            raise ValueError("Invalid event type. Use 'Highest' or 'Lowest'.")
        event_time = event_df.index
        time_of_event_price.append(event_time)
    return time_of_event_price

def get_premarket_stats(df, market_type="premarket"):
    res= {}
    df = get_market_type_data(df=df, market_type=market_type)
    unique_date = np.unique(df.index.date)
   
    for day in unique_date:
        day_df = df[df.index.date==day]
        min_value = day_df["Low"].min()
        max_value = day_df["High"].max()
        mean_value = day_df["Close"].mean()
        median_value = day_df["Close"].median()
        std_value = day_df["Close"].std()
        min_time = day_df[day_df["Low"] == min_value].index
        max_time = day_df[day_df["High"] == max_value].index
        
        res[day] = {f'{market_type}_min': {min_value}, 
                    f'{market_type}_max': {max_value}, 
                    f'{market_type}_mean': {mean_value}, 
                    f'{market_type}_median': {median_value},
                    f'{market_type}_std': {std_value},
                    f'{market_type}_min_time': {str(min_time.time)},
                    f'{market_type}_max_time': {str(max_time.time)}
                    }
                    
    return res

def get_current_stats_for_market(ticker=None, df=None, market_type = "premarket"):
    res = {}
    if not ticker and not df:
        raise ValueError("Both ticker and df are not given. Please provide at least one of them"
                         )
    if ticker:
        df = download_minute_interval_data(ticker)
    df = get_market_type_data(df=df, market_type=market_type)  
    curr_date = df.index.date[-1]
    curr_data = df[df.index.date==curr_date]
    curr_min = curr_data["Low"].min()
    curr_max = curr_data["High"].max()
    curr_std = curr_data["Close"].std()
    curr_median = curr_data["Close"].median()
    curr_mean = curr_data["Close"].mean()
    curr_close_price = curr_data["Close"].values[-1]
    print(f"current date: {curr_date}")
    
    res[f"{curr_date}"] = {"min": curr_min,
                           "max": curr_max,
                           "std": curr_std,
                           "median": curr_median,
                           "mean": curr_mean,
                           "current_close_price": curr_close_price,
                           "market_type": market_type
                           }
    return res

def cal_proba_low_preceds_high(df, market_type="regular"):
    case_counts = 0
    events_counts = 0
    df = get_market_type_data(df=df, market_type=market_type)
    unique_date = np.unique(df.index.date)
    for day in unique_date:
        day_df = df[df.index.date==day]
        low_day = day_df[day_df["Low"] == day_df["Low"].min()]
        low_time = low_day.index[0]
        high_day = day_df[day_df["High"] == day_df["High"].max()]
        high_time = high_day.index[0]
        
        if low_time < high_time:
            case_counts += 1
            events_counts += 1
            print(f"low time {low_time}  === higg time {high_time}")
        else:
            events_counts += 1
            
    if case_counts == 0:
        proba = 0
    else:
        proba = (case_counts / events_counts) * 100
        
    return {f"{market_type}_probability": proba}
            
# get n highest prices and their time

def get_price_and_time(df, market_type: str, num: int, direction: str):
    if direction not in ["highest", "lowest"]:
        raise ValueError("Invalid direction. Use 'highest' or 'lowest'.")
    
    df = get_market_type_data(df=df, market_type=market_type)
    
    if direction == "highest":
        direction_df = df.nlargest(num, columns="Close")
    else:
        direction_df = df.nsmallest(num, columns="Close")

    price = direction_df["Close"]
    event_date_time = direction_df.index
    price_time = zip(price, event_date_time)
    price_time_list = [item for item in price_time] 
    return price_time_list

def get_daily_price_and_time(df, market_type, num, direction):
    unique_date = np.unique(df.index.date)
    daily_price_and_time = {}
    for day in unique_date:
        day_df = df[df.index.date==day]
        price_time = get_price_and_time(df=day_df, num=num, direction=direction,
                                        market_type=market_type
                                        )
        daily_price_and_time[day] = price_time
    return daily_price_and_time

def cal_proba_of_status(df, status="open_eq_high"):
    if status not in ["open_eq_high", "open_eq_low", "open_eq_close", "close_eq_low", "close_eq_high"]:
        raise ValueError("Invalid status. Use 'open_eq_high', 'open_eq_low', 'open_eq_close' or 'close_eq_low', 'close_eq_high'.")
    df = df.dropna()
    all_events = len(df)
    if status == "open_eq_high":
        cases = df[df["Open"]==df["High"]]
    elif status == "open_eq_low":
        cases = df[df["Open"]==df["Low"]]
    elif status == "open_eq_close":
        cases = df[df["Open"]==df["Close"]]
    elif status == "close_eq_low":
        cases = df[df["Close"]==df["Low"]]
    elif status == "close_eq_high":
        cases = df[df["Close"]==df["High"]]
    case_ccounts = len(cases)
    proba = (case_ccounts / all_events) * 100
    return {"probability":proba,
            "cases":cases
            }
             
def O_H_LC(df, trading_dates):
    pass
    # check if timestamp for high is before L and L and Close are same time




