


#%%
import yfinance as yf
import os
# Example: Apple Inc.
ticker = 'LAES'
stock = yf.Ticker(ticker)


#%%
save_dir = "/home/lin/codebase/stock_app/src/stock_app/minute_data/17_02_2025_to_21_02_2025"
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
    start="2025-02-17"
    end="2025-02-21"
    hist = stock.history(start=start, 
                         end=end,
                        prepost=True,
                        interval='1m', 
                        period='8d',
                        )
    hist.to_csv(f"{save_dir}/{ticker}_2025_02_17_to_2025_02_21.csv")#2025_02_03_to_2025_02_07.csv")
    #print(hist.index[0])
    #print(hist.index[-1])


# %%
import yfinance as yf
stock = yf.Ticker("IONQ")
stock.history(#start=start, 
                         #end=end,
                        prepost=True,
                        interval='1m', 
                        period='8d',
                        )

# %%
