
from utils import (cal_proba_current_close_is_lower_than_nextday_high,
                   download_stock_price,
                   calculate_prob_close_lower_thn_open,
                   calculate_proba_close_lower_than_nextday_high_conditional,
                   cal_close_eq_low, high_low_diff,
                   calculate_proba_close_higher_than_nextday_low,
                   download_minute_interval_data,
                   buy_regular_at_premarket_lowest,
                   get_current_stats_for_market,
                   get_time_of_event_in_regular_market,
                   get_premarket_stats,
                   cal_proba_low_preceds_high, low_open_diff,
                   buy_afterhrs_at_regular_lowest,
                   close_open_diff, plot_column_chart,
                   cal_proba_regular_lowest_in_after_hours,
                   calculate_price_change,
                   cal_lowest_percent_target_is_below_base_market
                   
                   )
import plotly.express as px

## trading algorithm
#1. Buy at close price and exit at 1% profit margin


#%%
dwave = download_stock_price(stock_ticker="QBTS")


#%%
dwave_close_lower_than_nextday_high = cal_proba_current_close_is_lower_than_nextday_high(dwave)

dwave_close_lower_than_nextday_high["probability"]

#%%
pft_score = dwave_close_lower_than_nextday_high["percent_lower_scores"]

more_thn_1 = [pft for pft in pft_score if pft > 1]

(len(more_thn_1) / dwave_close_lower_than_nextday_high["total_instances"]) * 100
#%%

bigbear = download_stock_price(stock_ticker="BBAI")

#%%
bigbear_res = cal_proba_current_close_is_lower_than_nextday_high(bigbear)


#%%

pft_score = bigbear_res["percent_lower_scores"]

more_thn_1 = [pft for pft in pft_score if pft > 1]

#%%

(len(more_thn_1) / bigbear_res["total_instances"]) * 100
#%%
liveperson = download_stock_price(stock_ticker="LPSN")

liveperson_res = cal_proba_current_close_is_lower_than_nextday_high(liveperson)

(len(liveperson_res["percent_lower_scores"]) / liveperson_res["total_instances"]) * 100
more_thn_1 = [pft for pft in liveperson_res["percent_lower_scores"] if pft > 1]
(len(more_thn_1) / liveperson_res["total_instances"]) * 100
#%%
nvda = download_stock_price(stock_ticker="NVDA")
nvda_res = cal_proba_current_close_is_lower_than_nextday_high(nvda)
(len(nvda_res["percent_lower_scores"]) / nvda_res["total_instances"]) * 100


more_thn_1 = [pft for pft in nvda_res["percent_lower_scores"] if pft > 1]
(len(more_thn_1) / nvda_res["total_instances"]) * 100
#%%
laes = download_stock_price(stock_ticker="LAES")
laes_res = cal_proba_current_close_is_lower_than_nextday_high(laes)
more_thn_1 = [pft for pft in laes_res["percent_lower_scores"] if pft > 1]
(len(more_thn_1) / laes_res["total_instances"]) * 100



#%%
cerence = download_stock_price(stock_ticker="CRNC")
cerence_res = cal_proba_current_close_is_lower_than_nextday_high(cerence)
more_thn_1 = [pft for pft in cerence_res["percent_lower_scores"] if pft > 1]
(len(more_thn_1) / cerence_res["total_instances"]) * 100

#%%
qsi = download_stock_price(stock_ticker="QSI")
qsi_res = cal_proba_current_close_is_lower_than_nextday_high(qsi)
more_thn_1 = [pft for pft in qsi_res["percent_lower_scores"] if pft > 1]
(len(more_thn_1) / qsi_res["total_instances"]) * 100




laes_close_lwr_thn_opn = calculate_prob_close_lower_thn_open(df=laes)

#%%

laes_close_open_loss = laes_close_lwr_thn_opn["loss_percent_list"]

#%%
max(laes_close_open_loss)


len(laes_close_open_loss)

#%%
# by determing the how low stock usually closes below the opening when ever 
# the close is lower than the opening, a horizontal threshold can be determined
# for entry and exit. From the analysis below, there is about 84% probability 
# that stock price will close 10% or less lower than it opens in scenarios when the close price 
# is lower than open price. This suggests 
# that when a stock opens and the price is falls more than 10%, we could buy and take 
# at least 1% proft in 84% of such cases. In other words, when the price falls to 11% + 
# below open price there is about 84% probability that we will successful scalp 
# at least 1% profit by close price (when stock closes)



# In the case of price falling to 16% + lower,probability increases to 95%
# For 5% lower,probability decreases to 53%


len([x for x in laes_close_open_loss if x > -5]) / len(laes_close_open_loss)


#%%

len([x for x in laes_close_open_loss if (x <= -1) and (x >= -15)]) / len(laes_close_open_loss)

#%%
"""
What is the usual difference between the open price and low price daily



if the price falls to a certain lower percentage, what is the probability of recovering 
or closing at least at 1% profit


write an algo for the early stage of market open


will probable need multiple algorithms for same stock

visualize the stock price to know and understand what happens mostly


"""


#%%

laes_nextday_profit_res = calculate_proba_close_lower_than_nextday_high_conditional(laes)


#%%
laes_nextday_profit_res["probability"]

#%%
laes_nextday_profit_res["curr_closeprice_higher_thn_nextday_high_samples"][3]
#%% TODO: For the calculate_proba_close_lower_than_nextday_high_conditional results
# explore the samples that failed. 
# Find by how much Close was lower than the next day High so that becomes 
# a suggested % reduction of Close to enter the market in the After hours, Overnight 
# and Premarket


"""
it has been observed that when lower High are being observed, it is better 
to set a lower draw-down for entry.

Moreover, buying below the daily low is a better way.
"""



#%%

laes = low_open_diff(df=laes)


#%%
int(laes["low_open_pct_change"].min())


#%%
from collections import Counter

#%%
low_open_pct_int = [int(val) for val in laes["low_open_pct_change"].values]
low_open_pct_int = Counter(low_open_pct_int)


#%%

laes[laes["low_open_pct_change"] < -1]
#%%
# TODO: Analyze how high price rises after hitting -3% from open price

less__3 = laes[(laes["low_open_pct_change"] < int(-3))]

less__3[less__3["low_open_pct_change"] > int(-10)]

laes_int_low_open_pct_change = [int(x) for x in laes["low_open_pct_change"].values.tolist()]
laes["int_low_open_pct_change"] = laes_int_low_open_pct_change

# this analysis was not concluded because it was realized that High can occur before low
# hence designated for analysis


#%% TODO: Analyze edge case of O - H- LC  where by High occur before low
# what is the probability of it happening



'''
When current low is higher than next day high, a regime change 
may have occurred
'''




#%%  
close_eq_low = cal_close_eq_low(laes)

close_eq_low["probability"]

#%%
close_eq_low["cases"]
#%%


#%%
stock = yf.Ticker(ticker)
start = str(close_eq_low[-5].date())
end = str((close_eq_low[-5] + pd.Timedelta(days=1)).date())
laes_close_eq_low = stock.history(start="2024-12-18",
                                  end="2024-12-19",
                                  period='1d',
                                interval='1m', 
                                prepost=False
                                )
       
#%%
"""
It was observed that when a lower High occurs buying at Close and 
selling next day is not profitable. This needs further investigation
"""


high_low_diff(laes)


#%%
high_eql_open_row_indices = []
for row_index, row_data in laes.iterrows():
    if row_data["High"] == row_data["Open"]:
        high_eql_open_row_indices.append(row_index)


#%%

high_eql_low_df = laes[laes.index.isin(high_eql_open_row_indices)]


(len(high_eql_low_df)/len(laes)) * 100

#%%

laes[laes["Volume"] == laes["Volume"].min()]

#%%
px.histogram(data_frame=laes["low_open_pct_change"])

#%% 
#%%  Anlyze scenario for open == low and find probability of occurrence

laes_open_eq_low = laes[laes["Open"]==laes["Low"]]

(len(laes_open_eq_low) / len(laes)) * 100

#%%
"""
For laes, it is observed that the probability of open == low is about 6.57%.

This scenario will not be triggered for trading if we use the technique of 
going long only after a % decline in open price.
"""

#%%

calculate_prob_close_lower_thn_open(df=laes_open_eq_low)

#%%
close_open_diff(df=laes_open_eq_low)


#%%
"""
LAES
For the scenario of Open==Low, it is observed that there is 77.77% chance thate the Close price
will more than 3% higher than the Open price.

This is for trendy scenarios where price keep rising and never falls below the Open price.
When price falls less than or at most 3% higher than the Open price, buying will provide 
a 77.77% chance of making a profit at Close price.
"""
(len(laes_open_eq_low[laes_open_eq_low["close_open_pct_change"]>3]) / len(laes_open_eq_low))*100

#%%
"""
what is the proba that when close is lower than open, next day 
high will be higher than prevous day high
"""

#%% algo
"""
use open price as baseline and when price falls below it for about 2% buy 
and sell at 1% profit

this needs to be based on analysis of the typical difference between open and 
close particularly when it closes lower than open


# TODO: investigate setting the entry based on whether a new low as formed
"""



"""
for LAES, there is a band of .90 to .02 that can is observed
to be oscillating. 
Eg: Buy at 7.90 and sell at 8.02
2. Buy at 6.90 and sell at 7.02
3. Buy at 5.90 and sell at 6.02

"""


"""
for LAES, it has been be observed that at about 14:00 there is a price spike up 
and some few minutes prior to that is downward spiral. 
Strategy: time and buy during the downward spiral and take profit from 14:00 to 15:00

REPEATED AGAIN ON 2024-01-10

Trading zone for entry and exit is from 13:00 to 15:00

"""


#%%
"""

For LAES, it is observed that most of the volume occurs prior to 17:00

After 17:00, price action slows down significantly and can only be used to scalping of maximum 3%.
A safe and recommended is 1%.

Find the band of oscillation and scrap them for 1% profit
"""


#%%

import yfinance as yf
import pandas as pd
import datetime


#%%
ticker_symbol = 'AAPL'  # Example: Apple Inc.
start_date = '2025-01-11'
#end_date = '2023-01-02'

data = yf.download("LAES", start=start_date, interval='1m')
#data.head()


#%%

data.tail()

#%%

data = data.dropna()


#%%
data_8 = data[data.index.day == 8]


#%%

ionq_df = download_stock_price(stock_ticker="IONQ")

ionq_res = cal_proba_current_close_is_lower_than_nextday_high(ionq_df)

#%%

ionq_res["probability"]

#%%

pft_score = ionq_res["percent_lower_scores"]
more_thn_1 = [pft for pft in pft_score if pft > 1]
(len(more_thn_1) / ionq_res["total_instances"]) * 100


#%%

ionq_close_lwr_thn_opn = calculate_prob_close_lower_thn_open(df=ionq_df)

ionq_close_lwr_thn_opn["probability"]

#%%
low_open_diff(df=ionq_df)
ionq_low_open_pct_int = [int(val) for val in ionq_df["low_open_pct_change"].values]
ionq_low_open_pct_int = Counter(ionq_low_open_pct_int)

#%%
px.histogram(data_frame=ionq_df["low_open_pct_change"])

#%%

applovin_df = download_stock_price(stock_ticker="APP")

applovin_res = cal_proba_current_close_is_lower_than_nextday_high(applovin_df)

#%%

applovin_res["probability"]

#%%

pft_score = applovin_res["percent_lower_scores"]
more_thn_1 = [pft for pft in pft_score if pft > 1]
(len(more_thn_1) / applovin_res["total_instances"]) * 100


#%%

applovin_close_lwr_thn_opn = calculate_prob_close_lower_thn_open(df=applovin_df)

applovin_close_lwr_thn_opn["probability"]

#%%
low_open_diff(df=applovin_df)
applovin_low_open_pct_int = [int(val) for val in applovin_df["low_open_pct_change"].values]
applovin_low_open_pct_int = Counter(applovin_low_open_pct_int)

#%%
px.histogram(data_frame=applovin_df["low_open_pct_change"])


#%%

close_open_diff(df=applovin_df)

#%%
calculate_proba_close_lower_than_nextday_high_conditional(applovin_df)

#%%
calculate_price_change(applovin_df)
#%% 
px.histogram(data_frame=applovin_df["close_open_pct_change"])

#%%
plot_column_chart(applovin_df)
#%% TODO.: use linear regression for prediction of close_open_pct_change


#%%
data_8 = calculate_price_change(data=data_8)
plot_column_chart(data=data_8)

#%%
data_6 =  data[data.index.day == 6]

data_6[data_6["Close"]==data_6["Close"].min()]

#%%
data_7 =  data[data.index.day == 7]

data_7[data_7["Close"]==data_7["Close"].min()]

#%%
data_8 =  data[data.index.day == 8]

data_8[data_8["Close"]==data_8["Close"].min()]

#%%

data_8["Volume"].cumsum().is_monotonic_increasing

#%%

data.describe()
#%%
(97/100)*5.16
#%%
dwave[["Close"]].shift(-1) 





intc_stock = yf.Ticker("CRNC")

intc_prepost =intc_stock.history(start="2025-01-27", prepost=True,
                                  interval='1m', 
                                  period='8d',
                                  )
intc_lowreg_in_afterhr = cal_proba_regular_lowest_in_after_hours(intc_prepost)

intc_lowreg_in_afterhr["probability"]


#%%
#jpm_prepost.to_csv("/home/lin/codebase/stock_app/src/stock_app/minute_data/PEP_2025_01_11_to_2025_01_17.csv")

# %% 
after_hrs_res = buy_afterhrs_at_regular_lowest(intc_prepost)

after_hrs_res["profit_lose_percent_list"]


#%%
print(f"buy price: {after_hrs_res['buy_price_list']}")

print(f"sell price: {after_hrs_res['sell_price_list']}")
#%%
print(f"buy day: {after_hrs_res['buy_day_list']}")

print(f"sell day: {after_hrs_res['sell_day_list']}")

#%%


intc_df_day = intc_prepost[intc_prepost.index.date == intc_lowreg_in_afterhr["case_date"][-1]]

px.line(intc_df_day, x=intc_df_day.index, y="Close")

#%%
px.line(intc_prepost, x=intc_prepost.index, y="Close")


#%%

premarket_str_res = buy_regular_at_premarket_lowest(intc_prepost)

premarket_str_res["profit_lose_percent_list"]

#%%
#premarket_str_res.keys()

#%%
premarket_str_res["buy_price_list"]

#%%
premarket_str_res["sell_price_list"]

#%%
premarket_str_res["buy_day_list"]

#%%
premarket_str_res["sell_day_list"]


#%%
monitor_premarket_stocks = ["QBTS", "WKEY", "APP", "HSAI", "CRNC",
                            "QUBT", "DDD", "RKLB", "NUE", "RGTI",
                            "BBAI", "GWW", "TROW", "COIN", "MARA",
                            "RR", "NNE"
                    
                            ]

short_sell = ["SSTK", "MPW"]
selected_premarket_stocks = ["APP", "SMCI", "NOW", "QBTS", 
                             "RGTI", "LAES",
                             "AVGO", "SAP", "JPM", "NFLX",
                             "PEP", "WMT", "WDAY", "PLTR", "CRNC",
                             "QUBT", "AI", "HSAI", "LLY", "TSM", 
                             "BLK", "MSTR", "MCD", "LOW",
                             "PG", "WKEY", "TMO", "MPW", "SCHW",
                             "SSTK", "DDD", "AIR", "NSANY", "EQT",
                             "RR", "VRME", "CLSK", "TSLA", "META",
                             "AMZN", "GOOGL", "COIN", "ADP",
                             "CSCO", "JNJ", "NUE", "TROW",
                             "SYY", "GWW", "AZN", "NVAX", "MRNA",
                             "NVS", "BNTX", "GME", "AMC", "ZM",
                             "ALUR", 
                             "MSFT", "LUNR", "RKLB",
                             "SERV", "BEN", "SBUX", "DUK",
                             "C", "SIDU", "UPST", "HOOD", "RDW", "BABA",
                             "MARA", "NNE"
                             
                             ]

#%% train a model for detecting stocks to trade for premarket and regular
#%%
selected_aftermarket_stocks = ["CRWD", "ANET", "AVGO", 
                   "NFLX", "SAP", "IBM", "WMT",
                   "JPM", "GOOGL",
                   "MA", "LOW", "MAD",
                   "QCOM", "PEP", "TMO", "NOW",
                   "QUBT",
                   "WDAY", "LLY", "TSM",
                   "BLK", "MSTR", "MAIN", "PG",
                   "ABR", "LIN", "EAT", "MMM",
                   "ASML", "WKEY", "INTC", "SCHW",
                   "SSTK", "AIR", "EQT", "VRME", "CLSK",
                   "AAPL", "ADP", "CSCO", "SOFI", "NUE",
                   "ITW", "TROW", "SYY", "GWW", "AZN",
                   "MRK", "NVS", "BNTX", "AMC","ZM",
                   "MSFT", "SIDU", "NNN"
                
                   ]



import yfinance as yf

# Example: Apple Inc.
ticker = 'IONQ'
stock = yf.Ticker(ticker)
stock_price = download_stock_price(stock_ticker=ticker)

calculate_proba_close_higher_than_nextday_low(stock_price)




ticker = yf.Ticker("IONQ")

historical_data = ticker.history(#start="2025-01-27", 
                                        # prepost=True,
                                        interval='1m', 
                                        period='1d',
                                        )

#%%

#%%

ionq_min_data = download_minute_interval_data(ticker)

#%%
get_current_stats_for_market(ticker="IONQ", market_type="regular")

#%%
print(f"{ticker}: Highest time")
highest_time = get_time_of_event_in_regular_market(df=ionq_min_data)
highest_time

#%%

lowest_time = get_time_of_event_in_regular_market(df=ionq_min_data, 
                                                  event="Lowest")
print(f"{ticker}: Lowest time")
lowest_time

#%%
premart_stat = get_premarket_stats(ionq_min_data, market_type="premarket")
regular_stat = get_premarket_stats(ionq_min_data, market_type="regular")

#%%
print(ticker)
for item in regular_stat.keys():
    reg_max = regular_stat[item]["regular_max"]
    premart_max = premart_stat[item]["premarket_max"]
    print(f"{item}: premarket  max {premart_max} ==== regular max {reg_max}")


#%%
premart_stat

#%%
regular_stat


cal_proba_low_preceds_high(df=ionq_min_data, market_type="premarket")

cal_lowest_percent_target_is_below_base_market(ionq_min_data)