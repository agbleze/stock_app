# %%

import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
import time

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'YOUR_API_KEY'
fx = ForeignExchange(key=api_key)
def get_forex_data(base_currency, target_currency, outputsize='compact'):
    data, _ = fx.get_currency_exchange_daily(
        from_symbol=base_currency,
        to_symbol=target_currency,
        outputsize=outputsize
    )
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['date', 'open', 'high', 'low', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df

# Define parameters
base_currency = 'USD'
target_currency = 'EUR'

# Get forex data
forex_data = get_forex_data(base_currency, target_currency, outputsize='full')

# Save to CSV
forex_data.to_csv('forex_data.csv', index=False)
print('Forex data saved to forex_data.csv')

#%%

import pandas as pd
import requests

# Replace 'YOUR_API_KEY' with your actual Twelve Data API key
api_key = 'YOUR_API_KEY'

def get_forex_data(base_currency, target_currency, interval='1day', start_date='2021-01-01', end_date='2022-01-01'):
    url = f'https://api.twelvedata.com/time_series?symbol={base_currency}/{target_currency}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    if "values" in data:
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values('datetime', inplace=True)
        return df[['datetime', 'open', 'high', 'low', 'close']]
    else:
        print('Error fetching data:', data.get('message', 'Unknown error'))
        return pd.DataFrame()

# Define parameters
base_currency = 'USD'
target_currency = 'EUR'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Get forex data
forex_data = get_forex_data(base_currency, target_currency, start_date=start_date, end_date=end_date)

# Save to CSV
forex_data.to_csv('forex_data.csv', index=False)
print('Forex data saved to forex_data.csv')
#%%


#%%

import requests
import pandas as pd

# Replace with your IEX Cloud API key
api_key = 'YOUR_API_KEY'
symbol = 'AAPL'  # Replace with the desired stock symbol
start_date = '2022-04-18'  # Specify the start date in 'YYYY-MM-DD' format

url = f'https://cloud.iexapis.com/stable/stock/{symbol}/chart/date/{start_date.replace("-", "")}?chartByDay=true&token={api_key}'

response = requests.get(url)
data = response.json()

# Convert the data to a DataFrame
df = pd.DataFrame(data)
df['dateTime'] = pd.to_datetime(df['date'] + ' ' + df['minute'])
df = df.set_index('dateTime')
df = df[['open', 'high', 'low', 'close', 'volume']]

print(df)


import nasdaqdatalink
nasdaqdatalink.ApiConfig.api_key = nasdaq_api
mydata = nasdaqdatalink.get("FRED/GDP")

# probability of high == open

# Deuteronomy 10:17
# Ephesians 6:12

