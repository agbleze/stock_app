


#%%
import yfinance as yf
import os
# Example: Apple Inc.
ticker = 'LAES'
stock = yf.Ticker(ticker)


#%%
save_dir = "/home/lin/codebase/stock_app/src/stock_app/minute_data/09_04_2025_to_11_04_2025"
os.makedirs(save_dir, exist_ok=True)
#%% Download data including extended hours
# hist = stock.history(start="2025-01-11", #period='1d',
#                      interval='1m', prepost=True)


#%%
tickers = ["NVDA", "SMCI","AI", "RGTI", "QSI", "QUBT",
           "PLTR", "IONQ", "QBTS", "CRNC", "AVGO","ANET",
           "LLY", "AAPL", "LOM", "BLK", "WMT", "IBM", "O",
           "TMO","SOUN", "APP", "WKEY", "EQT", "AISP",
           "SAP",
           "RHM.DE", "LMT", "TRE", "HEI", "UNCRY", "ENL", "INTC",
           "ACA", "GFT", "FDX", "LIN", "V", "META", "QCOM",
           "NVO", "CRWD", "NFLX", "MCD", "AMAT", "BNP",
           "HO", "ADN1", "RBI"
           ]


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
                             "MARA", "NNE", "NNN",
                             "QTUM", "QS", "ARQQ", "PKST"
                             
                             ]
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
#%%
# tickers = ["RHM.DE", "LMT", "TRE", "HEI", "UNCRY", "ENL", "INTC",
#            "ACA", "GFT", "FDX", "LIN", "V", "META", "QCOM",
#            "NVO", "CRWD", "NFLX", "MCD", "AMAT", "BNP",
#            "HO", "ADN1", "RBI"]
#tickers = ["RHM.DE"]

#%% ["SES", "DECK", "PONY", "JILL", "PKST", "OKLO", "COIN", "SSTK", "CRNC"]
short_sell_tickers = ["SARO", "BBAI", "QUAD", "NVRI", "DJT", "COIN",
                    "UBI", "DNA", "ACMR", "NXT", "CHRD", "UAL",
                    "FPH", "PACK", "GTLB", "NSKOG", "RDW",
                    "OLP", "DEC", "CGEO", "KIE", "CRAYN",
                    "AMSSY", "ELMRA", "OLN", "WBA", "PFSI", "APPF",
                    "ENVX", "NKLA", "CRNC", "STEM", "SSTK",
                    "SES", "DECK", "PONY", "JILL", "PKST", "OKLO",
                    "RDWR"
                    ]
# AMSSY needs debugging for premarket
# no data -- CGEO, NSKOG, ELMRA
#preselected_shortsell = ["ASTS"]
for ticker in tickers:
    stock = yf.Ticker(ticker)
    # hist = stock.history(start="2025-01-27", period='8d',
    #                      interval='1m', prepost=True
    #                      )
    # hist.to_csv(f"{save_dir}/{ticker}_2025_01_27_to_2025_01_31.csv")
    #start_date = ""
    start="2025-04-09"
    end="2025-04-12"
    #start="2025-03-17"
    #end="2025-03-22"
    hist = stock.history(start=start, 
                         end=end,
                        prepost=True,
                        interval='1m', 
                        period='8d',
                        )
    hist.to_csv(f"{save_dir}/{ticker}_2025_04_09_to_2025_04_11.csv")#2025_02_03_to_2025_02_07.csv")
    #print(hist.index[0])
    #print(hist.index[-1])


# %%
start="2025-04-09"
end="2025-04-12"
import yfinance as yf
stock = yf.Ticker("IONQ")
df = stock.history(start=start, 
                    end=end,
                    prepost=True,
                    interval='1m', 
                    period='8d',
                    )

# %%
df.index[-1]
# %%
df.index[0]
# %%
import pandas as pd

#%%
data_path = "/home/lin/codebase/stock_app/src/stock_app/IONQ_2021_10_1_to_2025_2_23.csv"

df = pd.read_csv(data_path)


#%%

df.head()


#%%
def convert_to_us_eastern(df):
    """
    Convert the index of the DataFrame from UTC to US/Eastern time zone.
    
    Parameters:
    df (pd.DataFrame): DataFrame with DatetimeIndex in UTC.

    Returns:
    pd.DataFrame: DataFrame with DatetimeIndex in US/Eastern.
    """
    # Convert the index to the US/Eastern time zone
    df.index = df.index.tz_convert('US/Eastern')
    return df


from pandas.core.indexes.multi import MultiIndex


def preprocess_data(df):
    if isinstance(df, MultiIndex):
        df = df.droplevel(0)
    df.columns = [col.capitalize() for col in df.columns]
    df = convert_to_us_eastern(df=df)
    return df
       
#%%
from datetime import datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])#.columns#["timestamp"]

#%%

df.index = df['Timestamp']


#%%

ionq_df = preprocess_data(df)

#%%
import numpy as np
import plotly.express as px
import os
unique_date = np.unique(ionq_df.index.date)


#%%
savedir = "/home/lin/codebase/stock_app/src/stock_app/ionq_daily_plots"
for day in unique_date:
    df = ionq_df[ionq_df.index.date==day]
    fig = px.line(data_frame=df, x=df.index, 
                y="Close", #title=header, 
                template="plotly_dark"
                )
    save_path = os.path.join(savedir, f"ionq_{day}.png")
    fig.write_image(save_path)
# %%
