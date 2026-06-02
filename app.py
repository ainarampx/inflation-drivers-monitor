
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import os 

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
CREAM = "#F6F1E8"
GREEN = "#95AD88"
DARK = "#2B2B2B"
BEIGE = "#E8DDCF"
LIGHT = "#FBF8F1"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv("base_definitive_es_fr.csv")
df.columns = df.columns.str.strip()

# Create date variable
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
elif {"year", "quarter"}.issubset(df.columns):
    df["quarter"] = (
        df["quarter"]
        .astype(str)
        .str.replace("Q", "", regex=False)
        .str.strip()
    )

    df["period_str"] = df["year"].astype(int).astype(str) + "Q" + df["quarter"]

    df["date"] = pd.PeriodIndex(
        df["period_str"],
        freq="Q"
    ).to_timestamp()
elif "period" in df.columns:
    df["date"] = pd.PeriodIndex(df["period"].astype(str), freq="Q").to_timestamp()
elif "TIME_PERIOD" in df.columns:
    df["date"] = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q").to_timestamp()
else:
    raise ValueError("No date variable found. Please check your column names.")

# Identify country column
if "country" in df.columns:
    country_col = "country"
elif "geo" in df.columns:
    country_col = "geo"
elif "country_code" in df.columns:
    country_col = "country_code"
else:
    raise ValueError("No country column found.")

spain = df[df[country_col].isin(["ES", "Spain"])].copy()
spain = spain.sort_values("date")

# ------------------------------------------------------------
# Model variables
# ------------------------------------------------------------
spain["inflation_lag1"] = spain["inflation_total"].shift(1)
spain["d_profit_share"] = spain["profit_share"].diff()

model_data = spain.dropna(subset=[
    "inflation_total",
    "inflation_lag1",
    "inflation_energy",
    "ulc_yoy",
    "d_profit_share",
    "gdp_yoy"
]).copy()

X = model_data[[
    "inflation_lag1",
    "inflation_energy",
    "ulc_yoy",
    "d_profit_share",
    "gdp_yoy"
]]

X = sm.add_constant(X)
y = model_data["inflation_total"]

model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
model_data["fitted"] = model.fittedvalues

# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------
labels = {
    "inflation_lag1": "Past inflation",
    "inflation_energy": "Energy inflation",
    "ulc_yoy": "Unit labour cost growth",
    "d_profit_share": "Change in profit share",
    "gdp_yoy": "Real GDP growth"
}

economic_reading = {
    "inflation_lag1": "Inflation persistence",
    "inflation_energy": "External cost shock",
    "ulc_yoy": "Domestic production-cost pressure",
    "d_profit_share": "Distributional channel tested",
    "gdp_yoy": "Business-cycle conditions"
}

def stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return "n.s."

results = []

for var in labels.keys():
    results.append({
        "Variable": labels[var],
        "Coefficient": round(model.params[var], 3),
        "p-value": round(model.pvalues[var], 3),
        "Significance": stars(model.pvalues[var]),
        "Economic reading": economic_reading[var]
    })

results_table = pd.DataFrame(results)

# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------
fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=model_data["date"],
    y=model_data["inflation_total"],
    mode="lines",
    name="Actual inflation",
    line=dict(color=DARK, width=3)
))

fig_line.add_trace(go.Scatter(
    x=model_data["date"],
    y=model_data["fitted"],
    mode="lines",
    name="Fitted inflation",
    line=dict(color=GREEN, width=3, dash="dash")
))

fig_line.update_layout(
    title="Actual and fitted inflation — Spain",
    paper_bgcolor=CREAM,
    plot_bgcolor=CREAM,
    font=dict(color=DARK, family="Poppins, Arial"),
    margin=dict(l=40, r=30, t=60, b=40),
    legend=dict(orientation="h", y=-0.2),
    xaxis_title="",
    yaxis_title="Inflation rate (%)"
)

coef = model.params.drop("const").rename(index=labels)
pvalues = model.pvalues.drop("const")

coef_colors = [
    GREEN if pvalues[var] < 0.10 else BEIGE
    for var in model.params.drop("const").index
]

fig_coef = go.Figure(go.Bar(
    x=coef.values,
    y=coef.index,
    orientation="h",
    marker_color=coef_colors
))

fig_coef.update_layout(
    title="Which variables are most clearly associated with inflation?",
    paper_bgcolor=CREAM,
    plot_bgcolor=CREAM,
    font=dict(color=DARK, family="Poppins, Arial"),
    margin=dict(l=160, r=30, t=60, b=40),
    xaxis_title="Estimated coefficient",
    yaxis_title=""
)

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    style={
        "backgroundColor": CREAM,
        "minHeight": "100vh",
        "padding": "34px",
        "fontFamily": "Poppins, Arial",
        "color": DARK
    },
    children=[

        html.Div([
            html.H1(
                "Inflation Drivers Monitor — Spain",
                style={"fontSize": "36px", "fontWeight": "700", "marginBottom": "4px"}
            ),
            html.P(
                "A pedagogical dashboard based on the preferred augmented Phillips Curve",
                style={"fontSize": "17px", "color": GREEN, "marginTop": "0"}
            )
        ]),

        html.Div([
            html.Div([
                html.H4("Main message", style={"fontWeight": "700"}),
                html.P(
                    "Spanish inflation appears to be mainly linked to past inflation, energy prices, labour costs and "
                    "economic activity. By contrast, the profit-share effect is positive but not strong enough "
                    "to provide clear evidence of an independent role."
                )
            ], style={
                "backgroundColor": BEIGE,
                "padding": "22px",
                "borderRadius": "22px",
                "width": "100%",
                "marginBottom": "25px"
            })
        ]),

        dbc.Tabs([

            dbc.Tab(label="1. Model fit", children=[
                html.Div([
                    dcc.Graph(figure=fig_line)
                ], style={"paddingTop": "25px"})
            ]),

            dbc.Tab(label="2. Economic drivers", children=[
                html.Div([
                    dcc.Graph(figure=fig_coef)
                ], style={"paddingTop": "25px"})
            ]),

            dbc.Tab(label="3. Interactive results table", children=[
                html.Div([
                    html.H3("From econometric results to economic interpretation",
                            style={"marginTop": "25px", "fontWeight": "700"}),

                    dash_table.DataTable(
                        data=results_table.to_dict("records"),
                        columns=[{"name": i, "id": i} for i in results_table.columns],
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": GREEN,
                            "color": "white",
                            "fontWeight": "bold",
                            "border": "none"
                        },
                        style_cell={
                            "backgroundColor": LIGHT,
                            "color": DARK,
                            "padding": "12px",
                            "fontFamily": "Poppins, Arial",
                            "fontSize": "14px",
                            "border": f"1px solid {BEIGE}",
                            "textAlign": "left"
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{Significance} = 'n.s.'"},
                                "backgroundColor": "#F1E8DC",
                                "color": DARK
                            },
                            {
                                "if": {"filter_query": "{Significance} contains '*'"},
                                "backgroundColor": "#EEF3EA",
                                "color": DARK
                            }
                        ]
                    )
                ], style={"paddingTop": "10px"})
            ]),

            dbc.Tab(label="4. Coefficient-based interpretation tool", children=[
                html.Div([
                    html.H3("Coefficient-based interpretation tool",
                            style={"marginTop": "25px", "fontWeight": "700"}),

                    html.P(
                        "This tool does not produce a forecast. It only shows how predicted inflation "
                        "changes mechanically when one driver changes, using the estimated coefficients."
                    ),

                    html.Div([
                        html.Label("Energy inflation shock"),
                        dcc.Slider(
                            id="energy-slider",
                            min=-10,
                            max=20,
                            step=1,
                            value=5,
                            marks={-10: "-10", 0: "0", 10: "10", 20: "20"}
                        ),

                        html.Br(),

                        html.Label("Unit labour cost growth shock"),
                        dcc.Slider(
                            id="ulc-slider",
                            min=-5,
                            max=10,
                            step=1,
                            value=2,
                            marks={-5: "-5", 0: "0", 5: "5", 10: "10"}
                        ),

                        html.Br(),

                        html.Label("Profit-share change shock"),
                        dcc.Slider(
                            id="profit-slider",
                            min=-5,
                            max=5,
                            step=0.5,
                            value=1,
                            marks={-5: "-5", 0: "0", 5: "5"}
                        ),

                        html.Br(),

                        html.Label("Real GDP growth shock"),
                        dcc.Slider(
                            id="gdp-slider",
                            min=-10,
                            max=10,
                            step=1,
                            value=2,
                            marks={-10: "-10", 0: "0", 10: "10"}
                        ),
                    ], style={
                        "backgroundColor": LIGHT,
                        "padding": "25px",
                        "borderRadius": "22px",
                        "marginBottom": "25px"
                    }),

                    html.Div(id="scenario-output")
                ])
            ])
        ])
    ]
)

# ------------------------------------------------------------
# Scenario callback
# ------------------------------------------------------------
@app.callback(
    Output("scenario-output", "children"),
    Input("energy-slider", "value"),
    Input("ulc-slider", "value"),
    Input("profit-slider", "value"),
    Input("gdp-slider", "value")
)
def update_scenario(energy, ulc, profit, gdp):

    simulated_change = (
        model.params["inflation_energy"] * energy
        + model.params["ulc_yoy"] * ulc
        + model.params["d_profit_share"] * profit
        + model.params["gdp_yoy"] * gdp
    )

    return html.Div([
        html.H4("Estimated mechanical effect",
                style={"fontWeight": "700", "color": GREEN}),

        html.P(
            f"According to the estimated model, this scenario is associated with "
            f"an approximate change of {simulated_change:.2f} percentage points "
            f"in predicted inflation."
        ),

        html.P(
            "Interpretation: this is a model-based association, not a causal forecast.",
            style={"fontStyle": "italic"}
        )
    ], style={
        "backgroundColor": BEIGE,
        "padding": "25px",
        "borderRadius": "22px"
    })


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)