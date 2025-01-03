#%%
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
from prophet.plot import get_seasonality_plotly_props
from prophet import Prophet
import plotly.graph_objects as go
from prophet.serialize import model_to_json, model_from_json
import json
import numpy as np
from model_trainer import Model_Trainer, Transformer, plot_loss, expand_dates_excluding_weekends
import tensorflow as tf

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

predcol = ['High', 'Low', 'Close', 'Adj Close', 'Volume']

model_type = ("lstm", "bilstm", "cnn")

def download_stock_price(stock_ticker, start_date=None, end_date=None, **kwargs):
    data = yf.download(stock_ticker, start=start_date, end=end_date, **kwargs)
    if isinstance(data.columns, MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

        
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
                        dbc.Row(children=[dbc.Col(#lg=3,
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
                                              ),
                                          dbc.Col(dbc.Collapse(dbc.Col(dbc.Button(id="id_train_model",
                                                                          children="Train model"
                                                                          )
                                                               ),
                                                       id="id_collapse_train_model",
                                                       is_open=False
                                                       )
                                                  ),
                                          dbc.Col(dbc.Collapse(dbc.Col(dbc.Button(id="id_model_prediction",
                                                                          children="Predict"
                                                                          )
                                                               ),
                                                       is_open=False,
                                                       id="id_collapse_model_prediction"
                                                       )
                                                  )
                                          ]
                                ),
                        
                        
                            #]
                    #),
            dbc.Row([dcc.Loading(type='circle',
                                children=[dbc.Col(#lg=9,
                                                  children=[dcc.Graph(id='stock_price_graph')]
                                                    )
                                            ]
                                )
                     ]
                    ),
            html.Br(),
            dbc.Row([dcc.Loading(type="circle",
                                 children=[dbc.Col(children=[
                                                            dcc.Graph(id="seasonality_graph")
                                                            ]
                                                   )
                                           ]
                                 )
                     ]
                    ),
            html.Br(),
            dbc.Row([dcc.Loading(type="circle",
                                 children=[dbc.Col(children=[
                                                            dcc.Graph(id="pred_graph")
                                                            ]
                                                   )
                                           ]
                                 )
                     ]
                    ),
            dbc.Row(id="id_config_dialog"),
            dbc.Row(id="id_prediction_result_dialog")
        #])
    ]
)


model_performance_children = []
model_performance = html.Div(dbc.Row(id="id_model_performance", 
                                     children=[html.Span("No model trained yet") 
                                               if not model_performance_children
                                               else model_performance_children
                                               ][0]
                                     )
                             )

def create_model_performance_ui(model_performance_children):
    model_performance = html.Div(dbc.Row(id="id_model_performance", 
                                     children=[html.Span("No model trained yet") 
                                               if not model_performance_children
                                               else model_performance_children
                                               ][0]
                                     )
                             )
    return model_performance




brand_holder = html.Span("  Stock Analysis", className="bi bi-menu-down", id="id_brand_holder")
appside_layout = html.Div(
                            [dbc.NavbarSimple(
                                                #brand=brand_holder, #"Stock Analysis",
                                                brand_href="/",
                                                #class_name="bi bi-menu-down",
                                                children=brand_holder,
                                                #light=True,
                                                #brand_style={"color": "#FFFFFF", "backgroundColor": "#00624e"},
                                                
                                            ),
                             dcc.Loading(dcc.Store(id="id_trained_model_path", storage_type="local")),
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
                                                                html.Br(),
                                                                dbc.DropdownMenuItem(children=[html.H5(" Model Performance", className="bi bi-plus-slash-minus")],
                                                                                     id="id_model_perf",
                                                                                     )
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

train_config_layout = html.Div([dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Config model training")),
                                          dbc.ModalBody([dbc.Row([dbc.Col([dbc.Label("Training data size"),
                                                                          dbc.Input(type="number", min=0, max=1, step=0.1, id="id_train_size")
                                                                          ]
                                                                         ),
                                                                 dbc.Col([dbc.Label("Save model as"),
                                                                          dbc.Input(type="string",  id="id_save_model_as", value=None)
                                                                          ]
                                                                         ),
                                                                #  dbc.Col([dbc.Label("Testing data size"),
                                                                #           dbc.Input(type="number", min=0, max=1, step=0.1, id="id_test_size")
                                                                #           ]
                                                                #          )
                                                                ]
                                                                 ),
                                                         dbc.Row([dbc.Col([dbc.Label("Window - size of training timesteps"),
                                                                           dbc.Input(type="number", min=5, id="id_window_size",
                                                                                     value=180
                                                                                     )
                                                                           ]
                                                                          ),
                                                                  dbc.Col([dbc.Label("Horizon - size of prediction timesteps"),
                                                                           dbc.Input(type="number", min=5, id="id_horizon_size",
                                                                                     value=90
                                                                                     )
                                                                           ]
                                                                          ),
                                                                  dbc.Col([dbc.Label("Model type"),
                                                                           dcc.Dropdown(id="id_model_type",
                                                                                        options=[{"label": model, "value": model}
                                                                                                 for model in model_type
                                                                                                 ],
                                                                                        value="cnn"
                                                                                        )
                                                                           ], 
                                                                          width=2
                                                                          )
                                                                  ]
                                                                ),
                                                         dbc.Row([dbc.Col([dbc.Label("Buffer size"),
                                                                          dbc.Input(type="number", id="id_buffer_size",
                                                                                    value=100
                                                                                    )
                                                                          ]
                                                                          ),
                                                                  dbc.Col([dbc.Label("Number of Epochs"),
                                                                           dbc.Input(type="number", id="id_num_epochs",
                                                                                     #value=3
                                                                                     )
                                                                           ]
                                                                          ),
                                                                  dbc.Col([dbc.Label("Steps per Epoch"),
                                                                           dbc.Input(type="number", id="id_steps_per_epoch")
                                                                           ]
                                                                          )
                                                                ]
                                                                  ),
                                                         dbc.Row([dbc.Col([dbc.Label("Validation steps"),
                                                                           dbc.Input(type="number", id="id_val_steps"),
                                                                           ]
                                                                          ),
                                                                  
                                                                  dbc.Col([dbc.Label("Batch size"),
                                                                           dbc.Input(type="number", id="id_batch_size",
                                                                                     value=64),
                                                                           ]
                                                                          ),
                                                                  dbc.Col([#dbc.Label("Train Model"),
                                                                           html.Br(),
                                                                           dbc.Button("Start Model Training", id="id_start_model_train")
                                                                           ]
                                                                          )
                                                                  ]
                                                                 )
                                                         ]
                                                        ),
    
                                         ], is_open=True, size="lg"
                                        )
                               ]
                               )

prediction_config_layout = html.Div(dbc.Modal([dbc.ModalHeader([dbc.ModalTitle("Model prediction")]),
                                               dbc.ModalBody([dbc.Row([dbc.Col([dbc.Label("Model name"),
                                                                                                 dbc.Input(id="id_model_name",
                                                                                                           type="string",
                                                                                                           value=None
                                                                                                           )
                                                                                                 ]
                                                                                                ),
                                                                                        dbc.Col([html.Br(),
                                                                                                 dbc.Button("Start Prediction", id="id_start_model_prediction")
                                                                                                 ]
                                                                                                )
                                                                                        ]
                                                                                       )
                                                                               ]
                                                                              )
                                               ], is_open=True, size="lg"
                                              )
                                    )
app.layout = appside_layout

app.validation_layout = html.Div([appside_layout, stockprice_layout, main_layout, 
                                  train_config_layout, model_performance, 
                                  prediction_config_layout
                                  ]
                                )

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
            

def plot_forecast_component(data):
    data["Date"] = data.index.values
    data.index = pd.to_datetime(data.index)
    data.set_index(data["Date"], inplace=True)
    data = (data.rename(columns={"Date": "ds", "Close": "y"})
                [["ds", "y", "Volume"]]
            )
    model = Prophet()
    model.fit(df=data)
    yr_seasonality = get_seasonality_plotly_props(model, name="yearly")
    scatter = yr_seasonality["traces"][0]
    yval = scatter.y
    xval = scatter.x
    fig = px.line(x=xval, y=yval, template="plotly_dark",
                  title="Forecast components")
    return fig
   

def plot_model_fit(data, forecast_period=120):
    data["Date"] = data.index.values
    data.index = pd.to_datetime(data.index)
    data.set_index(data["Date"], inplace=True)
    data = (data.rename(columns={"Date": "ds", "Close": "y"})
                [["ds", "y", "Volume"]]
            )
    model = Prophet()
    model.fit(df=data)
    future = model.make_future_dataframe(periods=forecast_period)

    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], 
                            y=forecast["yhat"],
                            mode="lines", name="stock prediction"
                            )
                )
    fig.add_trace(go.Scatter(x=data["ds"], 
                            y=data["y"],
                            mode="lines", name="stock price"
                            )
                )
    fig.add_trace(go.Scatter(x=forecast["ds"],
                            y=forecast["yhat_upper"],
                            mode="lines", name="upper prediction",
                            fill="tonexty", fillcolor="rgba(68, 68, 68, 0.3)"
                            )
                )
    fig.add_trace(go.Scatter(x=forecast["ds"],
                            y=forecast["yhat_lower"],
                            mode="lines", name="lower prediction",
                            fill="tonexty", fillcolor="rgba(150, 150, 50, 0.3)"
                            ))
    fig.update_layout(legend=dict(yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    orientation="h"
                                ),
                      template="plotly_dark"
                      )
    return fig
    
       
@functools.lru_cache(maxsize=None)
@app.callback(Output(component_id="page_content", component_property="children"),
              Input(component_id="id_price_chart", component_property="n_clicks_timestamp"),
              Input(component_id="id_portfolio", component_property="n_clicks_timestamp"),
              Input(component_id="id_stock_perf", component_property="n_clicks_timestamp"),
              Input(component_id="id_model_perf", component_property="n_clicks_timestamp"),
              Input(component_id="id_trained_model_path", component_property="data")
              )
def sidebar_display(price_chart: str, portfolio_id, stock_portfolio,
                    model_perf, stored_data
                    ):#boxplot: str, scatter: str, corr: str):
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
    elif button_id == "id_model_perf":
        #print(f"stored_data: {stored_data}")
        if stored_data:
            for trained_stock in stored_data:
                for val in trained_stock.values():
                    if "model_performance_children" in val:
                        model_performance_children = val["model_performance_children"]
                    else:
                        model_performance_children = []
        else:
            model_performance_children = []
        #model_performance_children = stored_data["model_performance_children"]
        model_performance_ui = create_model_performance_ui(model_performance_children=model_performance_children)
        return model_performance_ui
    else:
        return dash.no_update
        
        
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
              Output(component_id="seasonality_graph", component_property="figure"),
              Output(component_id="pred_graph", component_property="figure"),
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
        #print(f"start_date: {start_date}")
        #print(f"end_date: {end_date}")
        #print(f"stock_ticker: {stock_ticker}")
    
        data = yf.download(stock_ticker, start=start_date, end=end_date)
        if isinstance(data.columns, MultiIndex):
            data.columns = data.columns.droplevel(1)
        #print(f"data: {data.head()}")
        fig = px.line(data_frame=data, y="Close", 
                        template="plotly_dark",
                        title=f"{stock_ticker} Close price",
                        height=500, #width=600
                        )
        seasonality_fig = plot_forecast_component(data=data)
        pred_fig = plot_model_fit(data=data)
        return fig, seasonality_fig, pred_fig

@app.callback(Output(component_id="id_collapse_train_model", component_property="is_open"),
              Output(component_id="id_collapse_model_prediction", component_property="is_open"),
              Input(component_id="id_stock_ticker", component_property="value"),
              Input(component_id="id_trained_model_path", component_property="data"),
              )
def show_model_button(stock_ticker, stored_data):
    if not stored_data:
        trained_stocks_ticker = []
    else:
        trained_stocks_ticker = []
        for st in stored_data:
            for ticker in st.keys():
                trained_stocks_ticker.append(ticker)
                
    print(f"trained_stocks_ticker: {trained_stocks_ticker}")
    if not stock_ticker:
        return dash.no_update, dash.no_update
    if stock_ticker and stock_ticker in trained_stocks_ticker:
        return True, True
    if stock_ticker and not stock_ticker in trained_stocks_ticker:
        return True, dash.no_update
    
@app.callback(Output(component_id="id_config_dialog", component_property="children"),
              Input(component_id="id_train_model", component_property="n_clicks"),
              Input(component_id="id_model_prediction", component_property="n_clicks")
              )
def show_model_config_dialog(model_config_button_click, prediction_config_button_click):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"button_id: {button_id}")
    if button_id == "id_train_model":
    #if model_config_button_click:
        return train_config_layout
    elif button_id == "id_model_prediction":
        return prediction_config_layout
    else:
        dash.no_update
        
@app.callback(Output(component_id="id_trained_model_path", component_property="data"),
              #Output(component_id="id_model_performance", component_property="children"),
              Input(component_id="id_train_size", component_property="value"),
              Input(component_id="id_save_model_as", component_property="value"),
              #Input(component_id="id_test_size", component_property="value"),
              Input(component_id="id_window_size", component_property="value"),
              Input(component_id="id_horizon_size", component_property="value"),
              Input(component_id="id_buffer_size", component_property="value"),
              Input(component_id="id_batch_size", component_property="value"),
              Input(component_id="id_num_epochs", component_property="value"),
              Input(component_id="id_start_model_train", component_property="n_clicks"),
              Input(component_id="id_stock_date", component_property="start_date"),
              Input(component_id="id_stock_date", component_property="end_date"),
              Input(component_id="id_stock_ticker", component_property="value"),
              Input(component_id="id_steps_per_epoch", component_property="value"),
              Input(component_id="id_val_steps", component_property="value"),
              Input(component_id="id_trained_model_path", component_property="data"),
              Input(component_id="id_model_type", component_property="value")
              )  
def train_model(train_size, save_model_as, #val_size, test_size, 
                window_size, horizon_size, buffer_size,
                batch_size, num_epochs, start_model_train_button, start_date, end_date,
                stock_ticker, steps_per_epoch, validation_steps, stored_data, model_type
                ):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"model_type: {model_type}")
    if button_id == "id_start_model_train":
        #total = train_size + val_size + test_size
    #if round(total, 2) != 1:
        #raise ValueError(f"Sum of train_size, val_size, and test_size should be equal 1 but got {train_size + val_size + test_size}")
    #else:
        data = download_stock_price(stock_ticker=stock_ticker, start_date=start_date, end_date=end_date)
                    
        #test_size = int(len(data)*test_size)
        #test_df = data.tail(test_size)
        test_df = data.tail(horizon_size)
        train_df = data.drop(test_df.index)
        train_endpoint = int(len(train_df) * train_size)
        fit_end_index = len(train_df)
        #print(f"fit_end_index: {fit_end_index}")
        trn = Transformer(data=train_df[["Volume"]])
        train_df[["Volume"]] = trn.transform(train_df[["Volume"]])
        test_df[["Volume"]] = trn.minmax_scaler.transform(test_df[["Volume"]])
        predictors = train_df[predcol]
        model_name = [stock_ticker if not save_model_as else save_model_as][0]
        save_model_path = f"model_store/{model_name}.h5"
        target = train_df[['Close']]
    
        mod_cls = Model_Trainer(steps_per_epoch=steps_per_epoch, 
                                epochs=num_epochs, 
                                predictors=predictors,
                                target=target, start=0,
                                train_endpoint=train_endpoint,
                                window=window_size, horizon=horizon_size, 
                                validation_steps=validation_steps,
                                batch_size=batch_size, buffer_size=buffer_size,
                                save_model_path=save_model_path,
                                model_type=model_type
                                )
        train_hist, model = mod_cls.run_model_training()
        loss_graph = plot_loss(history=train_hist, title=f"{stock_ticker} model loss")
        model_loss_grp = dcc.Graph(id=f"id_{stock_ticker}_model_loss", figure=loss_graph)
        model_perf_col = dbc.Col(model_loss_grp, width=6)
        eval_predictors = predictors.tail(window_size)
        eval_data_rescaled = np.array(eval_predictors).reshape(1, eval_predictors.shape[0], eval_predictors.shape[1])
        eval_results = model.predict(eval_data_rescaled)
        test_date = {"date": test_df.index}
        eval_data_placeholder = pd.DataFrame(test_date)
        #eval_data_placeholder = expand_dates_excluding_weekends(df=test_df, horizon=horizon_size)
        eval_data_placeholder["predicted_stock_price"] = eval_results[0]
        eval_data_placeholder["actual_stock_price"] = test_df["Close"].values
        res = mod_cls.timeseries_evaluation_metrics(y_true=test_df["Close"].values,
                                                    y_pred=eval_results[0]
                                                    )
        rmse = round(res["root_mean_squared_error"], 3)
        r2 = round(res["R2"], 3)
        #print(f"test_df['Close']: {test_df['Close']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eval_data_placeholder["date"], 
                                y=eval_data_placeholder["actual_stock_price"],
                                mode="lines", name="Actual Stock close price"
                                )
                    )
        fig.add_trace(go.Scatter(x=eval_data_placeholder["date"], 
                                y=eval_data_placeholder["predicted_stock_price"],
                                mode="lines", name="predicted stock close price"
                                )
                    )
        fig.update_layout(legend=dict(yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1,
                                        orientation="h"
                                    ),
                        template="plotly_dark",
                        title=f"{stock_ticker} {model_type} Model Test Evaluation   RMSE: {rmse}  R2: {r2}"
                        )
        fig
        model_eval_graph = dcc.Graph(figure=fig)
        model_eval_col = dbc.Col(model_eval_graph, width=6)                  
        model_performance_children.append(model_perf_col)
        model_performance_children.append(model_eval_col)
        
        
        if not stored_data:
            res_stored_data = []
        else:
            res_stored_data = stored_data
        res_stored_data.append({f"{stock_ticker}": {#"train_history": train_hist, 
                                    "model_path": save_model_path,
                                    "model_performance_children": model_performance_children, #model_performance_children,
                                    "scaler_info": {"fit_end_index": fit_end_index},
                                    "window_size": window_size, 
                                    "horizon_size": horizon_size,
                                    "model_name": model_name
                                    }
                })
        #print(f"res_stored_data: {res_stored_data}")
        return res_stored_data
        
            
@app.callback(Output(component_id="id_prediction_result_dialog", component_property="children"),
              Input(component_id="id_stock_date", component_property="start_date"),
              Input(component_id="id_stock_date", component_property="end_date"),
              Input(component_id="id_stock_ticker", component_property="value"),
              Input(component_id="id_model_name",component_property="value"),
              Input(component_id="id_start_model_prediction", component_property="n_clicks"),
              Input(component_id="id_trained_model_path", component_property="data")
              )
def make_prediction(start_date, end_date, stock_ticker, model_name, 
                    start_predic_click, stored_data
                    ):
    #print(f"pred_horizon: {pred_horizon}")
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "id_start_model_prediction":
        if not stock_ticker:
            raise ValueError(f"stock_ticker {stock_ticker} not defined")
        
        if not stored_data:
            trained_model_names = []
        else:
            trained_model_names = []
            for data in stored_data:
                stored_model_name = [data_item.get("model_name") for data_item in data.values()][0]
                trained_model_names.append(stored_model_name)
        print(f"trained_model_names: {trained_model_names}")            
        #print(f"trained_stocks_ticker: {trained_stocks_ticker}")
        if model_name not in trained_model_names:
            raise ValueError(f"No trained model found for {stock_ticker}. Please train model before creating prediction")
        
        for stored_instance in stored_data:
            if stock_ticker in stored_instance.keys():
                if model_name == stored_instance[stock_ticker]["model_name"]:
                    stock_model_path = stored_instance[stock_ticker]["model_path"]
                    fit_end_index = stored_instance[stock_ticker]["scaler_info"]["fit_end_index"]
                    window_size = stored_instance[stock_ticker]["window_size"]
                    horizon_size = stored_instance[stock_ticker]["horizon_size"]
                    loaded_model = tf.keras.models.load_model(stock_model_path)
                else:
                    raise FileNotFoundError(f"Selected model: {model_name} not found: Need to train a model saved as {model_name}")
        
        data = download_stock_price(stock_ticker=stock_ticker, start_date=start_date, end_date=end_date)
        train_df = data.head(fit_end_index)
        trn = Transformer(data=train_df[["Volume"]])
        scaler = trn.get_minmax_scaler()
        
        pred_data = data.tail(window_size)
        pred_data[["Volume"]] = scaler.transform(pred_data[["Volume"]])
        pred_data = pred_data[predcol]
        pred_data_rescaled = np.array(pred_data).reshape(1, pred_data.shape[0], pred_data.shape[1])
        predicted_results = loaded_model.predict(pred_data_rescaled)
        pred_data_placeholder = expand_dates_excluding_weekends(df=pred_data, horizon=horizon_size)
        pred_data_placeholder["predicted_stock_price"] = predicted_results[0]
        #print(f"predicted_results: {predicted_results}")
        #print(f"pred_data_placeholder: {pred_data_placeholder}")
        title = f"{stock_ticker} Price prediction"
        pred_graph = px.line(data_frame=pred_data_placeholder, 
                        x="date", y="predicted_stock_price",
                        template="plotly_dark", title=title
                        )
        pred_figure = dcc.Graph(figure=pred_graph)
        # data_val = train_df[train_df.columns[1:]].tail(180)
        # val_rescaled = np.array(data_val).reshape(1, data_val.shape[0], data_val.shape[1])
        # predicted_results = model.predict(val_rescaled)
        
        
        #predicted_results = loaded_model.predict(data)
        return dbc.Col(children=pred_figure)
    
            # mod_cls.plot_loss_history()
            # data_val = train_df[train_df.columns[1:]].tail(180)
            # val_rescaled = np.array(data_val).reshape(1, data_val.shape[0], data_val.shape[1])
            # predicted_results = model.predict(val_rescaled)
            # test_df["Close"].to_list()
            # mod_cls.timeseries_evaluation_metrics(y_true=test_df["Close"].to_list(),
            #                                     y_pred=predicted_results.tolist()[0]
            #                                     )

            
            
            
              
#TODO: format predictions and plot
# store train config required during prediction like window size and use this for prediction

# TODO: add validation of model config input

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


#%% TODO. Add off canvas to sidebar menu to it can disapper when a buthon is clicked

if __name__ == "__main__":
    app.run_server()
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
(101/100)*9.17

#%%
(8.20/7.94)*100

# %%

# %%

## trading algorithm
#1. Buy at close price and exit at 1% profit margin


#%%
dwave = download_stock_price(stock_ticker="QBTS")

#%%
dwave_rows = [i for i in dwave.iterrows()]

for i in dwave_rows:
    print(type(i[1]))

for rowdata_index, rowdata in monitor_data.iterrows():
#%%
def calculate_profit(df, profit_percent):
    pass