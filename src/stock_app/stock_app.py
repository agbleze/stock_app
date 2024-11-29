from dash import html, Input, Output, State, dcc, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
from dashapp.helper_components import (make_boxplot, plot_barplot, get_path,
                               )
import dash
import joblib
import functools
import plotly.express as px




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
    style=page_style,
)

app.layout = main_layout


