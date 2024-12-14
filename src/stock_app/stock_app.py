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
from pandas.core.indexes.multi import MultiIndex
import functools


card_icon = {
    "color": "#0088BC",
    "textAlign": "center",
    "fontSize": "4em",
    "margin": "auto"
}

company_ticker = {"Tecnicas Reunidas SA": "0MKT.IL",
                  "SAP": "SAP", "Dell Technologies": "DELL",
                  "Brinker International": "BKJ.F", "Unipol Gruppo SpA": "UIPN.MU",
                  "CaixaBank SA": "48CA.MU", "Bilfinger": "0NRG.IL",
                  "ASML Holding": "ASME.DE", "Süss MicroTec": "SMHN.DE",
                  "Heidelberg Materials": "HEIU.MU", "Brenntag": "BNR.DE",
                  "C3.ai": "724.DE", "ING Groep": "1INGA.MI", "Wienerberger": "WIB.DE",
                  "Rio Tinto": "RIO1.DE", "UniCredit SpA": "CRIN.DU",
                  "Enel SpA": "ENL.DE", "ProCredit Holding": "PCZ.DE",
                  "Super Micro Computer": "MS51.BE", "Intel": "INTC",
                  "Iberdrola SA": "IBE1.DU", "Credit Agricole SA": "ACAP.XC",
                  "Palantir Technologies": "PTX.DU",
                  "Smith & Wesson Brands": "SWS.DU", "Walmart": "WMT.DE",
                  "Microsoft": "MSF.DE", "Rheinmetall": "RHM.DE",
                  "Coinbase Global": "1QZ.DE", "Qualcomm": "QCI.F", 
                  "Koninklijke Philips": "PHIA.F", "L'Oreal SA": "LOR.MU",
                  "FedEx": "FDX.DE", "Linde": "LIN.DE", "Visa": "3V6.F",
                  "The Walt Disney": "WDPD.XC", "Deutsche Lufthansa": "LHA.DE",
                  "Volkswagen": "VOW3.DE", "Novo Nordisk": "NOV.DE",
                  "Schaeffler": "SHA0.DE", "TechnipFMC": "1T1.DU", "Crowdstrike Holdings": "45C.F",
                  "Deutz": "DEZ.DE", "Colgate-Palmolive": "0P59.L", "Immersion Corp": "IMV.MU",
                  "Vesta Wind Systems": "VWSB.DE", "Hensoldt": "HAG.DE",
                  "Verizon Communications": "BACB.F", "flatexDEGIRO": "FTK.DE",
                  "McDonald's": "MDO0.F", "The Coca-Cola": "CCC3.DE", "Nvidia": "NVDA.DE",
                  "Applied Materials": "AP2.MU", "PUMA": "PUM.DE",
                  "Leonardo SpA": "FMNB.MU", "NN Group": "2NN.DE", "BAE Systems": "BSP.DE",
                  "Bitcoin Group": "ADE.MU", "BNP Paribas SA": "BNPH.F", "Allianz": "ALV.DE",
                  "Broadcom": "1YD.DU", "Uber Technologies": "UBER", "Thales SA": "THAL.VI",
                  "International Business Machines": "IBM.DE", "Western Digital": "WDC.MU",
                  "Veolia Environment SA": "VVD.MU", "Heineken": "HEIA.VI", "AXA SA": "AXA.DU",
                  "adesso": "ADN1.DE", "Advanced Micro Devices": "AMD.F", 
                  "Vienna Insurance Group": "WSV2.DU", "Grupo Catalana Occidente SA": "OCZA.MU",
                  "Siemens": "SIE.DE", "FLSmidth": "F6O1.DU", "Bank of America": "NCB.F",
                  "Banco Bilbao Vizcaya Argentaria SA": "BBVA.MU", "Alphabet": "GOOG",
                  "Unilever": "UNA.AS", "Airbus": "AIR.PA", "PDD Holdings": "PDD",
                  "Mondelez International": "KTF.DE", "Freeport-McMoRan": "FPMB.DE",
                  "voestalpine": "VASS.MU", "LVMH Moet HennessyLouis Vuitton": "MOH.BE",
                  "Bechtle": "BC8.DE", "CENIT": "CSH.DE", "Rolls-Royce Holdings": "RRU.DE",
                  "OMV": "OMV.VI", "Merck": "MRK", "Intesa Sanpaolo SpA": "ISPM.XD",
                  "Commerzbank": "CBK.DE", "Microstrategy": "MSTR", "RWE": "RWE.DE",
                  "Micro Technology": "MU", "Jackson Financial": "8WF.MU", "Porsche": "PAH3.DE",
                  "Infineon Technologies": "IFX.DE", "Dropbox": "1Q5.F", 
                  "Riot Platforms": "RIOT", "Tenet Healthcare": "THC1.MU", "Applovin Corp": "APP",
                  "Palo Alto Networks": "5AP.MU", "C3.ai": "724.DE", "Amazon": "AMZN",
                  "Tesla": "TL0.DE", "Apple": "APC.DE", "ON Semiconductor": "XS4.F",
                  "Grand City Properties SA": "GYC.DU", "GFT Technologies": "GFT.MU",
                  "Mapfre SA": "MAPE.XC"
                  }




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
            children=[html.Span("Click here", className="bi bi-menu-down")]
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
                        html.H5('Stock ticker', className="bi bi-menu-down",
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
                                                         children="Get Stock price",
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

brand_holder = html.Span("  Stock Analysis", className="bi bi-menu-down", id="id_brand_holder")
appside_layout = html.Div(
                            [dbc.NavbarSimple(
                                                brand=brand_holder, #"Stock Analysis",
                                                brand_href="/",
                                                #light=True,
                                                #brand_style={"color": "#FFFFFF", "backgroundColor": "#00624e"},
                                                
                                            ),
                            #  dbc.Offcanvas(id="id_sidebar_offcanvas", is_open=False,
                            #                children=[dtc.SideBar([dtc.SideBar([dtc.SideBarItem(id="id_proj_desc", label="Projection Description", 
                            #                                                                         icon="bi bi-menu-down" 
                            #                                                                         ),
                            #                                                     dtc.SideBarItem(id="id_price_chart",
                            #                                                                     label="Price Chart", 
                            #                                                                     icon="bi bi-bezier"
                            #                                                                     ),
                            #                                                     dtc.SideBarItem(id="id_portfolio", label="Portfolio Monitory", 
                            #                                                                     icon="bi bi-bar-chart"
                            #                                                                     ),
                                                                                
                            #                                                     dtc.SideBarItem(id="id_stock_perf", label="Performance", 
                            #                                                                     icon="bi bi-plus-slash-minus"
                            #                                                                     ),
                            #                                                     ],
                            #                                                 ),
                            #                                     ],
                            #                                     #bg_color="#0ca678",
                            #                                 )
                            #                         ]
                            #                ),
                                dbc.Row([
                                dbc.Col([#html.Div(id="id_show_offcanvas"),
                                         dbc.Offcanvas(id="id_sidebar_offcanvas", is_open=False,
                                                        children=#[dtc.SideBar([
                                                            #dtc.SideBar(
                                                                [
                                                                # dtc.SideBarItem(id="id_proj_desc", label="Projection Description", 
                                                                #             icon="bi bi-menu-down" 
                                                                #             ),
                                                                # dtc.SideBarItem(id="id_price_chart",
                                                                #                 label="Price Chart", 
                                                                #                 icon="bi bi-bezier"
                                                                #                 ),
                                                                # dtc.SideBarItem(id="id_portfolio", label="Portfolio Monitory", 
                                                                #                 icon="bi bi-bar-chart"
                                                                #                 ),
                                                                
                                                                # dtc.SideBarItem(id="id_stock_perf", label="Performance", 
                                                                #                 icon="bi bi-plus-slash-minus"
                                                                #                 ),
                                                                
                                                                dbc.DropdownMenuItem(children=[html.H5(" Projection Description", className="bi bi-menu-down")], 
                                                                                     id="id_proj_desc"
                                                                                     ), html.Br(),
                                                                dbc.DropdownMenuItem(children=[html.H5(" Price Chart", className="bi bi-bezier")], 
                                                                                     id="id_price_chart",
                                                                                     ), html.Br(),
                                                                dbc.DropdownMenuItem(children=[html.H5(" Portfolio Monitory", className="bi bi-bar-chart")],
                                                                                     id="id_portfolio",
                                                                                     ), html.Br(),
                                                                dbc.DropdownMenuItem(children=[html.H5(" Performance", className="bi bi-plus-slash-minus")],
                                                                                     id="id_stock_perf",
                                                                                     ),
                                                                ],
                                                                                           # ),
                                                                                #],
                                                                                #bg_color="#0ca678",
                                                                            #)
                                                                    #]
                                                        ),
                                                dbc.Col([], id="page_content", 
                                                        #style=page_style
                                                        )
                                                ]
                                              )
                            ])
                            ]
                        )



def output_card(id: str = None, card_label: str =None,
                style={"backgroundColor": 'yellow'},
                icon: str ='bi bi-cash-coin', card_size: int = 4):
    return dbc.Col(lg=card_size,
                    children=dbc.CardGroup(
                        children=[
                            dbc.Card(
                                    children=[
                                        dcc.Loading(type='circle', children=html.H3(id=id)),
                                        html.P(card_label)
                                    ]
                                ),
                            dbc.Card(
                                    children=[
                                        html.Div(
                                            className=icon,
                                            style=card_icon
                                        )
                                    ],
                                    style=style
                            )
                        ]
                    )
                )


new_div = html.Div(dbc.Row(id="id_portfolio_monitor"))

portfolio_list = dbc.ListGroup([dbc.ListGroupItem(stock) for stock, ticker in company_ticker.items()])

portfolio_button = dbc.ButtonGroup([dbc.Button(stock) for stock in company_ticker.keys()], vertical=True)

portfolio_canvas = dbc.Offcanvas(children=portfolio_list, 
                                 is_open=True, placement="end"
                                )

portfolio_page = html.Div([dbc.Row([dbc.Col(dcc.Dropdown(id="id_portfolio_db", 
                                                 options=[{"label": label, "value": value}
                                                          for label, value in company_ticker.items()
                                                          ],#.keys(),
                                                 #value=company_ticker.values(),
                                                 placeholder="Portfolio"
                                                 ),
                                            width=3
                                            ),
                                    dbc.Col(dbc.Button("Watch List"), width=3)
                                    ]
                                   ),
                           dbc.Row(id="id_portfolio_items")
                           ]
                        )

app.layout = appside_layout

app.validation_layout = html.Div([appside_layout, stockprice_layout, main_layout])

def create_portfolio_graphs(company_ticker):
    head_component = [dbc.Row(dcc.DatePickerRange(id="id_portfolio_date")), html.Br(),]
    portfolio_graphs = []
    for company, stock_ticker in company_ticker.items():
        data = yf.download(stock_ticker)
        if isinstance(data.columns, MultiIndex):
            data.columns = data.columns.droplevel(1)
        fig = px.line(data_frame=data, y="Close", 
                    template="plotly_dark",
                    title=f"{company} Close price",
                    )
        stock_plot = dcc.Graph(figure=fig)
        # TODO: calculate stock price change from first and last dates
            
        portfolio_graph = dbc.Col(id=f"id_{stock_ticker}", width=6,children=[stock_plot])
        portfolio_graphs.append(portfolio_graph)
        portfolio_graphs.append(html.Br())
    if portfolio_graphs:
        head_component.append(dbc.Row(children=portfolio_graphs))
    else:
        print(f"No portfolio graphs created")
    return html.Div(children=head_component)
            
    
    
@functools.lru_cache(maxsize=None)
@app.callback(Output(component_id="page_content", component_property="children"),
              Input(component_id="id_price_chart", component_property="n_clicks_timestamp"),
              Input(component_id="id_portfolio", component_property="n_clicks_timestamp"),
              Input(component_id="id_stock_perf", component_property="n_clicks_timestamp")
              )
def sidebar_display(price_chart: str, portfolio_id, stock_portfolio):#boxplot: str, scatter: str, corr: str):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if not ctx.triggered:
        pass
        #return main_layout
        #return appside_layout
        #return appside_layout
    elif button_id == "id_price_chart":
        return stockprice_layout
    elif button_id == 'id_portfolio':
        return portfolio_page    
    elif button_id == "id_stock_perf":
        return portfolio_canvas
    else:
        print("nothing new to show")
        
        
@app.callback(Output(component_id="id_sidebar_offcanvas",component_property="is_open"),
              Input(component_id="id_brand_holder", component_property="n_clicks"),
              State(component_id="id_sidebar_offcanvas", component_property="is_open")
              )
def show_sidebar_offcanvas(brand_holder_click, is_open):
    if brand_holder_click:
        return not is_open
    return is_open
    



@app.callback(Output(component_id="id_portfolio_items", component_property="children"),
              Input(component_id="id_portfolio_db", component_property="value")
              )
def update_portfolio_items(stock_ticker: str):
    return html.Div(stock_ticker)
    
    
    
@functools.lru_cache(maxsize=None)
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
        return  dash.no_update
        
    if button_id == 'id_submit_stock_request':
        print(f"start_date: {start_date}")
        print(f"end_date: {end_date}")
        print(f"stock_ticker: {stock_ticker}")
    
        data = yf.download(stock_ticker, start=start_date, end=end_date)
        if isinstance(data.columns, MultiIndex):
            data.columns = data.columns.droplevel(1)
        #print(f"data: {data.head()}")
        fig = px.line(data_frame=data, y="Close", 
                        template="plotly_dark",
                        title=f"{stock_ticker} Close price",
                        height=500, width=600
                        )
        return fig

# @functools.lru_cache(maxsize=None)
# @app.callback(Output(component_id="page_content", component_property="children"),
#               Input(component_id="id_portfolio", component_property="n_clicks_timestamp")
#             )
# def create_portfolio_view(id_portfolio_click):
#     ctx = dash.callback_context
#     button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
#     if button_id != 'id_submit_stock_request':
#         return  dash.no_update
        
#     if button_id == 'id_portfolio':
#         head_component = [dbc.Row(dcc.DatePickerRange(id="id_portfolio_date"))]
#         portfolio_graphs = []
#         for stock_ticker in company_ticker:
#             data = yf.download(stock_ticker)
#             if isinstance(data.columns, MultiIndex):
#                 data.columns = data.columns.droplevel(1)
#             fig = px.line(data_frame=data, y="Close", 
#                         template="plotly_dark",
#                         title=f"{stock_ticker} Close price",
#                         )
#             stock_plot = dcc.Graph(figure=fig)
#             # TODO: calculate stock price change from first and last dates
                
#             portfolio_graph = dbc.Col(id=f"id_{stock_ticker}", width=2,children=[stock_plot])
#             portfolio_graphs.append(portfolio_graph)
#         if portfolio_graphs:
#             head_component.append(dbc.Row(children=portfolio_graphs))
#         else:
#             print(f"No portfolio graphs created")
#         return html.Div(children=head_component)
            
    

# TODO: Portfolio update
# get all portfolio ticker, download data, create graph and show % change for each
# when percentage change is greater than 10% then send an email notification

# what happens when a company beats its guidance 


# TODO: performance calculator similar to that on SAP wensite
# https://www.sap.com/investors/en/stock.html


# TODO. Add off canvas to sidebar menu to it can disapper when a buthon is clicked

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
