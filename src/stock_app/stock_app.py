from dash import html, Input, Output, State, dcc, callback_context
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


appside_layout = html.Div(
                            [dbc.NavbarSimple(
                                                brand="Stock Analysis",
                                                brand_href="/",
                                                light=True,
                                                brand_style={"color": "#FFFFFF", "backgroundColor": "#00624e"},
                                            ),
                                dtc.SideBar(
                                    [
                                        dtc.SideBarItem(id="id_proj_desc", label="Projection Description", 
                                                        icon="bi bi-body-text" 
                                                        ),
                                        dtc.SideBarItem(id="id_model_eval", label="Model Evaluation", 
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
                                html.Div([], id="page_content", style=page_style)
                            ]
                        )

app.layout = appside_layout


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
