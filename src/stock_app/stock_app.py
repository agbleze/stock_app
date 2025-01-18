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
#from .style import page_style
import yfinance as yf
from pandas.core.indexes.multi import MultiIndex
import functools
from prophet.plot import get_seasonality_plotly_props
from prophet import Prophet
import plotly.graph_objects as go
from prophet.serialize import model_to_json, model_from_json
import json
import numpy as np
#from model_trainer import Model_Trainer, Transformer, plot_loss, expand_dates_excluding_weekends
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
  

# TODO: Portfolio update
# get all portfolio ticker, download data, create graph and show % change for each
# when percentage change is greater than 10% then send an email notification

# what happens when a company beats its guidance 


# TODO: performance calculator similar to that on SAP wensite
# https://www.sap.com/investors/en/stock.html

if __name__ == "__main__":
    app.run_server()

# %%
(107/100)*7.6

#%%
(8.20/7.94)*100

# %%

# %%

## trading algorithm
#1. Buy at close price and exit at 1% profit margin


#%%
dwave = download_stock_price(stock_ticker="QBTS")


#%%

#%%
dwave_rows = [i for i in dwave.iterrows()]



#%%
higher_high = 0
all_comp = 0
for rowdata_index, rowdata in dwave.iterrows():
    curr_close_price = rowdata["Close"]
    if rowdata_index != dwave.tail(1).index:
        nextday_high = dwave[dwave.index >= rowdata_index].iloc[1]["High"]
        if nextday_high > curr_close_price:
            higher_high += 1
            all_comp += 1
        else:
            all_comp += 1
    else:
        print(f"last day: {rowdata_index}")
    #break 

(higher_high / all_comp) * 100

#%%
def calculate_prob(df, type="close lower than next day High", cal_profit_percent=True):
    higher_high = 0
    all_comp = 0
    profit_percent = 0
    profit_percent_scores = []
    for rowdata_index, rowdata in df.iterrows():
        #if type == "close lower than next day High":
        curr_close_price = rowdata["Close"]
        if rowdata_index != df.index[-1]:
            nextday_high = df[df.index >= rowdata_index].iloc[1]["High"]
            if nextday_high > curr_close_price:
                higher_high += 1
                if cal_profit_percent:
                    profit = ((nextday_high / curr_close_price) * 100) -100
                    #print(f"profit: {profit}")
                    profit_percent += profit
                    profit_percent_scores.append(profit)
                    
                all_comp += 1
            else:
                all_comp += 1
        else:
            print(f"last day: {rowdata_index}")
        #break 

    prob = (higher_high / all_comp) * 100
    return {"probability": prob, 
            "profit_percent": profit_percent,
            "profit_percent_scores": profit_percent_scores,
            "total_instances": all_comp
            }
    

#%%
dwave_close_lower_than_nextday_high = calculate_prob(dwave)

dwave_close_lower_than_nextday_high["probability"]

#%%
pft_score = dwave_close_lower_than_nextday_high["profit_percent_scores"]

more_thn_1 = [pft for pft in pft_score if pft > 1]

(len(more_thn_1) / dwave_close_lower_than_nextday_high["total_instances"]) * 100
#%%

bigbear = download_stock_price(stock_ticker="BBAI")

#%%
bigbear_res = calculate_prob(bigbear)


#%%

pft_score = bigbear_res["profit_percent_scores"]

more_thn_1 = [pft for pft in pft_score if pft > 1]

#%%

(len(more_thn_1) / bigbear_res["total_instances"]) * 100
#%%
liveperson = download_stock_price(stock_ticker="LPSN")

liveperson_res = calculate_prob(liveperson)

(len(liveperson_res["profit_percent_scores"]) / liveperson_res["total_instances"]) * 100
more_thn_1 = [pft for pft in liveperson_res["profit_percent_scores"] if pft > 1]
(len(more_thn_1) / liveperson_res["total_instances"]) * 100
#%%
nvda = download_stock_price(stock_ticker="NVDA")
nvda_res = calculate_prob(nvda)
(len(nvda_res["profit_percent_scores"]) / nvda_res["total_instances"]) * 100


more_thn_1 = [pft for pft in nvda_res["profit_percent_scores"] if pft > 1]
(len(more_thn_1) / nvda_res["total_instances"]) * 100
#%%
laes = download_stock_price(stock_ticker="LAES")
laes_res = calculate_prob(laes)
more_thn_1 = [pft for pft in laes_res["profit_percent_scores"] if pft > 1]
(len(more_thn_1) / laes_res["total_instances"]) * 100



#%%
cerence = download_stock_price(stock_ticker="CRNC")
cerence_res = calculate_prob(cerence)
more_thn_1 = [pft for pft in cerence_res["profit_percent_scores"] if pft > 1]
(len(more_thn_1) / cerence_res["total_instances"]) * 100

#%%
qsi = download_stock_price(stock_ticker="QSI")
qsi_res = calculate_prob(qsi)
more_thn_1 = [pft for pft in qsi_res["profit_percent_scores"] if pft > 1]
(len(more_thn_1) / qsi_res["total_instances"]) * 100


#%%

def calculate_prob_close_lower_thn_open(df):
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
        

#%%

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
def calculate_next_day_profit_from_curr_close(df):
    """if current high, low and close are higher than previous than buy at close and 
        sell next day at profit

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
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
                    #print(f"profit: {profit}")
                    profit_percent += profit
                    profit_percent_scores.append(profit)
                        
                    all_comp += 1
                else:
                    all_comp += 1
                    curr_and_nextday_df = df[df.index >= rowdata_index].iloc[0:2]
                    curr_closeprice_higher_thn_nextday_high_samples.append(curr_and_nextday_df)
        else:
            print(f"last day: {rowdata_index}")
        #break 

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
    
#%%

laes_nextday_profit_res = calculate_next_day_profit_from_curr_close(laes)


#%%
laes_nextday_profit_res["probability"]

#%%
laes_nextday_profit_res["curr_closeprice_higher_thn_nextday_high_samples"][3]
#%% TODO: For the calculate_next_day_profit_from_curr_close results
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
#%%

def low_open_diff(df):
    df["low_open_pct_change"] = ((df["Low"] - df["Open"])/df["Open"]) * 100
    return df


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


import pandas_market_calendars as mcal
import pandas as pd
def get_trading_dates(year, exchange="NYSE"):
    stock_exchange = mcal.get_calendar(exchange)
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    schedule = stock_exchange.schedule(start_date=start_date, end_date=end_date)
    trading_dates = schedule.index
    trading_dates_list = trading_dates.tolist()
    return trading_dates_list


#%%
trading_dates = get_trading_dates(year=2024)

#%%
def O_H_LC(df, trading_dates):
    pass
    # check if timestamp for high is before L and L and Close are same time


import yfinance as yf

# Example: Apple Inc.
ticker = 'LAES'
stock = yf.Ticker(ticker)

#%% Download data including extended hours
hist = stock.history(start="2024-01-16", end="2024-01-17", period='1d',
                     interval='1m', prepost=False)

#%%
hist[hist["High"]==hist["High"].max()].index

#%%
hist[hist["Low"]==hist["Low"].min()].index



#%%


#%%
import nasdaqdatalink
nasdaqdatalink.ApiConfig.api_key = nasdaq_api
mydata = nasdaqdatalink.get("FRED/GDP")
#%%  Analysis of worst case scenario of O-H-LC

def cal_close_eq_low(df):
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

#%%
def high_low_diff(df):
    df["high_low_pct_change"] = ((df["High"] - df["Low"])/df["Low"]) * 100
    return df


high_low_diff(laes)
#%%
laes[laes["int_low_open_pct_change"] == -3]["High"].mean() - laes[laes["int_low_open_pct_change"] == -3]["Low"].mean() 


#%%

(4.98/7.15)*100
#%%
high_eql_open_row_indices = []
for row_index, row_data in laes.iterrows():
    if row_data["High"] == row_data["Open"]:
        high_eql_open_row_indices.append(row_index)


#%%

high_eql_low_df = laes[laes.index.isin(high_eql_open_row_indices)]

#%%

# probability of high == open

(len(high_eql_low_df)/len(laes)) * 100

#%%

laes[laes["Volume"] == laes["Volume"].min()]

#%%

curr_price = 100

((100-10)/100) *  curr_price

#%%
((100-5)/100) * curr_price

#%%
(101 / 100) * 5.89
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
def close_open_diff(df):
    df["close_open_pct_change"] = ((df["Close"] - df["Open"])/df["Open"]) * 100
    return df

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



#%%

nvda
low_open_diff(nvda)

#%%

nvda.tail(50)
#%%

from dataclasses import dataclass

#%%
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

#%%

import yfinance as yf
import pandas as pd
import datetime


#%%
ticker_symbol = 'AAPL'  # Example: Apple Inc.
start_date = '2025-01-05'
end_date = '2023-01-02'


#%%
data = yf.download("LAES", start=start_date, interval='1m')
#data.head()


#%%

data.tail()

#%%

data = data.dropna()


#%%
data_8 = data[data.index.day == 8]


#%%
def calculate_price_change(data, col="Close"):
    data["pct_change"] = data[col].pct_change()
    data["pct_change"] = round(data["pct_change"]*100, 2)
    data["diff"] = data[col].diff()
    data["status_rise"] = data["diff"].map(lambda x: x > 0)
    data["color"] = ["red" if x == False else "green" for x in data[["status_rise"]].values]
    return data

#%%

    """Execution strategy
    
    Place Buy limit order at a certain price of entry and place sell limit 
    order below a certain price below to exit at loss. 
    Then place a buy limit order at the sell limit order used and calculate 
    the percentage  required to recover loss + profit to determine 
    new Sell limit order price.
    
    
    Example usage
    I set buy limit order at $ 10 and went long with a sell limit order of 
    $11 for profit and set Loss limit order at $ 5. 
    Now price fall to $5 and sell limit order was triggered so I sold at a lost.
    Price fell further to $1 and when got to $2, I went long gain and set new sell 
    limit order at $5 which was successful.
    
    If I started with 1000$ for trading, this will be the case.
    1. 1000/10 -> 100 shares initially
    2. Amt after selling at 5 -> 500 hence loss f 500
    3  500/2 -> 250 shares for new buy price
    4. 250 * $5 -> $1250 -> After selling at new price
    Profit = 1250 - 1000 -> 250
    
    In a case where the initial goal of $11 was achieved then, 
    profit = 1100 - 1000 -> 100
    
    In the case of do nothing and wait for price, then I would have 
    ended the day with loss of $500. Hence stop losses when implemented well
    can actually be beneficial to revert a lossing trade to profit.
    Lot of calculations need to be done to determine this entry and exit points.
    
     
    """
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

#%%

ionq_df = download_stock_price(stock_ticker="IONQ")

ionq_res = calculate_prob(ionq_df)

#%%

ionq_res["probability"]

#%%

pft_score = ionq_res["profit_percent_scores"]
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

applovin_res = calculate_prob(applovin_df)

#%%

applovin_res["probability"]

#%%

pft_score = applovin_res["profit_percent_scores"]
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
calculate_next_day_profit_from_curr_close(applovin_df)

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
#%%
def calculate_profit(df, profit_percent):
    pass


#%%
import math

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

# Example usage
principal = 1000
daily_rate = 0.01  # 1% daily interest rate
required_days = days_to_double(principal, daily_rate)
print(f"Number of days required to double ${principal} with a daily interest rate of {daily_rate*100}%: {required_days} days")

#%%
import math
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
    # Calculate the amount of money accumulated
    accumulated_amount = principal * (1 + daily_rate) ** num_trades
    
    return accumulated_amount

#%% Example usage
principal = 1000
daily_rate = 0.01  # 1% daily interest rate
num_trades = 252  # Number of trades
accumulated_amount = compounded_amount(principal, daily_rate, num_trades)
print(f"Accumulated amount after {num_trades} trades with a daily interest rate of {daily_rate*100}%: ${accumulated_amount:.2f}")

# %%
import yfinance as yf

# Example: Apple Inc.
ticker = 'LAES'
stock = yf.Ticker(ticker)

# Download data including extended hours
hist = stock.history(start="2025-01-16", end="2025-01-17", period='1d',
                     interval='1m', prepost=False)

#%%
hist[hist["High"]==hist["High"].max()].index

#%%
hist[hist["Low"]==hist["Low"].min()].index




# %%
import yfinance as yf
import pandas as pd

# Example: Apple Inc.
ticker = 'LAES'
stock = yf.Ticker(ticker)
_date = "2025-01-10"
# Download data including extended hours
hist = stock.history(period='5d', interval='1m', prepost=True)

# Convert the index to a column for easier filtering
hist = hist.reset_index()

# Define market open and close times
market_open = pd.Timestamp("09:30", tz="US/Eastern").time()
market_close = pd.Timestamp("16:00", tz="US/Eastern").time()

# Filter pre-market data
pre_market_data = hist[hist['Datetime'].dt.time < market_open]

# Filter after-hours data
after_hours_data = hist[hist['Datetime'].dt.time > market_close]

print("Pre-market data:")
print(pre_market_data.head())

print("\nAfter-hours data:")
print(after_hours_data.head())





# %%

# %%
import yfinance as yf
import pandas as pd

# Create a ticker object for EUR/USD
ticker = yf.Ticker("EURUSD=X")

# Download historical data
#data = ticker.history(period="5d", interval="1m")

data = ticker.history()
# Display the data
#print(data.head())

# %%
data
# %%
calculate_next_day_profit_from_curr_close(data)
# %%
close_open_diff(data)
# %%
data.tail(20)
# %%
low_open_diff(data)
# %%
(4.45/4.99)*100
# %%
(89/100)*4.99
# %%

import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
import time

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'YOUR_API_KEY'
fx = ForeignExchange(key=api_key)
fx.
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



#%%
"""
well, I asses the 1% compounding interest (1 ci) to better 
than 3% profit set aside (3 ps) because earning 1% profit daily 
on the stock market is more achieveable than 3% because my strategy 
makes it easier to get 1% scalp than 3% due to the longer exposure 
than which can to to a reversal before 3% is reached and loss
"""




# %%
