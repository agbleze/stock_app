from dash import html, Input, Output, State, dcc, callback_context, callback, Dash
from datetime import date
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
#from dashapp.helper_components import (make_boxplot, plot_barplot, get_path)
import dash
import joblib
import functools
import plotly.express as px
import dash_trich_components as dtc
from style import page_style
import yfinance as yf


app = dash.Dash(__name__, external_stylesheets=[
                                                dbc.themes.SOLAR,
                                                dbc.icons.BOOTSTRAP,
                                                dbc.icons.FONT_AWESOME
                                            ],
                suppress_callback_exceptions=True,
                )


main_layout = html.Div(
    [
        dbc.NavbarSimple(
            brand="Company",
            brand_href="/",
            light=True,
            brand_style={"color": "#FFFFFF", "backgroundColor": "#00624e"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Location(id="location"),
                        html.Div(id="main_content"),
                    ]
                )
            ]
        ),
    ],
    #style=page_style,
)

stockprice_layout = html.Div(
    children=[
        #dbc.Row([
            #dbc.Col(lg=3,
                    #children=[
                        html.H5('Stock ticker', 
                                  #style=input_style
                                  ),
                        dbc.Row(children=[dbc.Col(lg=3,
                                                  children=[dbc.Input(id="id_stock_ticker",
                                                                      placeholder="Input stock ticker as shown in yahoo finance",
                                                                      type="text"
                                                                      )
                                                            ]
                                                  ), 
                                          dbc.Col(children=[dcc.DatePickerRange(id="id_stock_date",
                                                                                )
                                                            ]
                                                  ),
                                          dbc.Col(
                                              dbc.Button(id="id_submit_stock_request",
                                                         children="Get Stock price"
                                                         )
                                              )
                                          ]
                                ),
                        
                        
                            #]
                    #),
            dbc.Row(dcc.Loading(type='circle',
                                children=[dbc.Col(#lg=9,
                                                  children=[dcc.Graph(id='stock_price_graph')]
                                                    )
                                            ]
                                )
                    )
        #])
    ]
)

appside_layout = html.Div(
                            [dbc.NavbarSimple(
                                                brand="Stock Analysis",
                                                brand_href="/",
                                                light=True,
                                                brand_style={"color": "#FFFFFF", "backgroundColor": "#00624e"},
                                            ),
                                dbc.Row(
                                dbc.Col([
                                                dtc.SideBar(
                                                    [
                                                        dtc.SideBarItem(id="id_proj_desc", label="Projection Description", 
                                                                        icon="bi bi-body-text" 
                                                                        ),
                                                        dtc.SideBarItem(id="id_price_chart", #id="id_model_eval", 
                                                                        label="Price Chart", 
                                                                        icon="bi bi-bezier"
                                                                        ),
                                                        # dbc.Collapse(id="id_collapse_model_eval", is_open=False,
                                                        #             children=[dtc.SideBarItem(label="Cross validation",
                                                        #                                     id="id_crossval_btn", 
                                                        #                                     icon="fa-solid fa-caret-up",
                                                        #                                     n_clicks=0,
                                                        #                                     ),
                                                        #                     dtc.SideBarItem(label="Classification report",
                                                        #                                     id="id_classification_btn", 
                                                        #                                     icon="fa-brands fa-intercom",
                                                        #                                     n_clicks=0,
                                                        #                                 ),
                                                        #                     dtc.SideBarItem(label="ROC Curve",
                                                        #                                     id="id_roc_btn", 
                                                        #                                     icon="bi bi-graph-up-arrow",
                                                        #                                     n_clicks=0,
                                                        #                                 )
                                                        #                 ]
                                                        #             ),
                                                        
                                                        
                                                        dtc.SideBarItem(id="id_data_viz", label="Data Visualization", 
                                                                        icon="bi bi-bar-chart"
                                                                        ),
                                                        
                                                        dtc.SideBarItem(id="id_prediction", label="Prediction", 
                                                                        icon="bi bi-plus-slash-minus"
                                                                        ),
                                                    ],
                                                    bg_color="#0ca678",
                                                ),
                                                dbc.Col([], id="page_content", style=page_style)
                                                ]
                                              )
                                )
                            ]
                        )

app.layout = appside_layout

app.validation_layout = html.Div([appside_layout, stockprice_layout])

@app.callback(Output(component_id="page_content", component_property="children"),
              Input(component_id="id_price_chart", component_property="n_clicks_timestamp")
              )
def sidebar_display(price_chart: str, ):#boxplot: str, scatter: str, corr: str):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if not ctx.triggered:
        return appside_layout
        #return appside_layout
    elif button_id == "id_price_chart": ##"id_hist":
        return stockprice_layout
    # elif button_id == "id_boxplot":
    #     return boxplot_layout
    # elif button_id == "id_scatter":
    #     return scatter_layout
    # elif button_id == "id_corr":
    #     return multicoll_layout
    else:
        pass
        #return intro_layout 

@app.callback(Output(component_id="stock_price_graph", component_property="figure"),
              Input(component_id="id_stock_date", component_property="start_date"),
            Input(component_id="id_stock_date", component_property="end_date"),
            Input(component_id="id_submit_stock_request", component_property="n_clicks"),
            Input(component_id="id_stock_ticker", component_property="value")
            )
def get_date(start_date, end_date, button_click, stock_ticker):
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id != 'id_submit_stock_request':
        PreventUpdate
        
    if button_id == 'id_submit_stock_request':
        print(f"start_date: {start_date}")
        print(f"end_date: {end_date}")
        data = yf.download(stock_ticker, start=start_date, end=end_date)
        # fig = px.line(data_frame=data, y="Close", 
        #                 template="plotly_dark",
        #                 title=f"{stock_ticker} Close price - crowdstrike",
        #                 #height=800, width=1800
        #                 )
        # return fig
    #else:
    #    PreventUpdate
        



if __name__ == "__main__":
    app.run_server(debug=True)
# TODO
# Add ticker selection option



#%%

from pystocktopus import Stock

def get_ticker(company_name):
    stock = Stock()  # Create an instance of the Stock class
    companies = stock.find_tickers()  # Get a list of all tickers and their companies
    for company in companies:
        if company_name.lower() in company['name'].lower():
            return company['symbol']
    return None

# Example usage
company_name = "Apple"
ticker = get_ticker(company_name)
print(f"The ticker for {company_name} is: {ticker}")

# %%
from pytickersymbols import PyTickerSymbols

stock_data = PyTickerSymbols()
countries = stock_data.get_all_countries()
indices = stock_data.get_all_indices()
industries = stock_data.get_all_industries()
# %%
