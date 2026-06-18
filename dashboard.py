"""
Customer Personality Analysis — Interactive Dashboard
======================================================
Run with:  python dashboard/dashboard.py
Then open: http://127.0.0.1:8050
"""

import os
import sys
import webbrowser
import threading
import platform
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Make project root importable ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dash import Dash, dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "dash", "dash-bootstrap-components", "-q"])
    from dash import Dash, dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc

# ─── Load data ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")
RAW_PATH     = os.path.join(DATA_DIR, "raw_dataset.csv")

DATA_DIR_ALT = os.path.join(BASE_DIR, "Customer Personality prediction", "data")
RAW_PATH_ALT = os.path.join(DATA_DIR_ALT, "raw_dataset.csv")
CLEANED_PATH_ALT = os.path.join(DATA_DIR_ALT, "cleaned_dataset.csv")

if os.path.exists(CLEANED_PATH):
    df = pd.read_csv(CLEANED_PATH)
elif os.path.exists(CLEANED_PATH_ALT):
    df = pd.read_csv(CLEANED_PATH_ALT)
else:
    # Fallback: run cleaning inline
    fallback_raw = RAW_PATH if os.path.exists(RAW_PATH) else RAW_PATH_ALT
    if not os.path.exists(fallback_raw):
        fallback_raw = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Customer personality dataset", "marketing_campaign.csv")
    df = pd.read_csv(fallback_raw, sep="\t")
    df.columns = df.columns.str.lower().str.strip()

# ─── Theme ───────────────────────────────────────────────────────────────────
BG         = "#0F0F1A"
CARD_BG    = "#1A1A2E"
ACCENT     = "#4361EE"
ACCENT2    = "#F72585"
TEXT       = "#E0E0FF"
BORDER     = "#2A2A4A"
PALETTE    = ["#4361EE","#F72585","#4CC9F0","#7209B7","#3A0CA3",
              "#4895EF","#560BAD","#F3722C","#90BE6D","#F9C74F"]

PLOTLY_THEME = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, Arial"),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor=CARD_BG, bordercolor=BORDER),
    margin=dict(l=40, r=20, t=50, b=40),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig


# ─── KPI helpers ─────────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle="", color=ACCENT):
    return html.Div([
        html.P(title, style={"color": TEXT, "fontSize": "0.78rem",
                              "marginBottom": "4px", "opacity": "0.7",
                              "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.H3(value, style={"color": color, "fontSize": "1.8rem",
                               "fontWeight": "800", "margin": "0"}),
        html.P(subtitle, style={"color": TEXT, "fontSize": "0.75rem",
                                  "opacity": "0.55", "marginTop": "2px"}),
    ], style={
        "background": CARD_BG, "borderRadius": "12px", "padding": "20px 24px",
        "border": f"1px solid {BORDER}", "flex": "1", "minWidth": "160px",
    })


# ─── App ─────────────────────────────────────────────────────────────────────
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP,
                                  "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap"],
           title="Customer Personality Dashboard")

# Education & Marital options
edu_options  = [{"label": "All", "value": "All"}] + \
               [{"label": v, "value": v} for v in sorted(df["education"].dropna().unique())] \
               if "education" in df.columns else [{"label": "All", "value": "All"}]
mar_options  = [{"label": "All", "value": "All"}] + \
               [{"label": v, "value": v} for v in sorted(df["marital_status"].dropna().unique())] \
               if "marital_status" in df.columns else [{"label": "All", "value": "All"}]

income_min = int(df["income"].min()) if "income" in df.columns else 0
income_max = int(df["income"].max()) if "income" in df.columns else 200000

# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div(style={"background": BG, "minHeight": "100vh",
                               "fontFamily": "Inter, Arial", "padding": "0"},
children=[
    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("Customer Personality Analysis",
                    style={"color": TEXT, "margin": "0", "fontWeight": "800",
                           "fontSize": "1.6rem", "letterSpacing": "-0.5px"}),
            html.P("Interactive EDA Dashboard  ·  CodeAlpha Data Science Internship",
                   style={"color": TEXT, "opacity": "0.5", "margin": "4px 0 0",
                          "fontSize": "0.82rem"}),
        ]),
        html.Div("⬡  LIVE", style={"color": "#90BE6D", "fontWeight": "800",
                                    "fontSize": "0.8rem", "letterSpacing": "2px",
                                    "padding": "6px 14px",
                                    "border": "1px solid #90BE6D",
                                    "borderRadius": "20px"})
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "center", "padding": "20px 32px",
              "borderBottom": f"1px solid {BORDER}",
              "background": CARD_BG}),

    # ── Filters ───────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Education Level", style={"color": TEXT, "fontSize": "0.78rem",
                                                  "marginBottom": "6px", "display": "block",
                                                  "opacity": "0.7"}),
            dcc.Dropdown(id="edu-filter", options=edu_options, value="All",
                         clearable=False,
                         style={"background": BG, "color": TEXT, "border": f"1px solid {BORDER}",
                                "borderRadius": "8px", "minWidth": "160px"})
        ], style={"flex": "1"}),
        html.Div([
            html.Label("Marital Status", style={"color": TEXT, "fontSize": "0.78rem",
                                                  "marginBottom": "6px", "display": "block",
                                                  "opacity": "0.7"}),
            dcc.Dropdown(id="mar-filter", options=mar_options, value="All",
                         clearable=False,
                         style={"background": BG, "color": TEXT, "border": f"1px solid {BORDER}",
                                "borderRadius": "8px", "minWidth": "160px"})
        ], style={"flex": "1"}),
        html.Div([
            html.Label(f"Income Range  (${income_min:,} – ${income_max:,})",
                       id="income-label",
                       style={"color": TEXT, "fontSize": "0.78rem",
                              "marginBottom": "8px", "display": "block", "opacity": "0.7"}),
            dcc.RangeSlider(id="income-filter",
                            min=income_min, max=income_max,
                            step=5000,
                            value=[income_min, income_max],
                            marks={income_min: f"${income_min//1000}K",
                                   income_max: f"${income_max//1000}K"},
                            tooltip={"placement": "bottom",
                                     "always_visible": False})
        ], style={"flex": "3"}),
    ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
              "padding": "20px 32px", "borderBottom": f"1px solid {BORDER}"}),

    # ── KPI Row ───────────────────────────────────────────────────────────────
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "16px", "padding": "24px 32px",
                    "flexWrap": "wrap"}),

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    html.Div([
        html.Div(dcc.Graph(id="chart-spend-edu", config={"displayModeBar": False}),
                 style={"flex": "1", "background": CARD_BG, "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"}),
        html.Div(dcc.Graph(id="chart-age-hist", config={"displayModeBar": False}),
                 style={"flex": "1", "background": CARD_BG, "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"}),
    ], style={"display": "flex", "gap": "20px", "padding": "0 32px 20px"}),

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    html.Div([
        html.Div(dcc.Graph(id="chart-scatter", config={"displayModeBar": False}),
                 style={"flex": "2", "background": CARD_BG, "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"}),
        html.Div(dcc.Graph(id="chart-pie", config={"displayModeBar": False}),
                 style={"flex": "1", "background": CARD_BG, "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"}),
    ], style={"display": "flex", "gap": "20px", "padding": "0 32px 20px"}),

    # ── Charts Row 3 ──────────────────────────────────────────────────────────
    html.Div([
        html.Div(dcc.Graph(id="chart-campaign", config={"displayModeBar": False}),
                 style={"flex": "1", "background": CARD_BG, "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"}),
        html.Div(dcc.Graph(id="chart-heatmap", config={"displayModeBar": False}),
                 style={"flex": "1", "background": CARD_BG, "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"}),
    ], style={"display": "flex", "gap": "20px", "padding": "0 32px 20px"}),

    # ── Insights Section ──────────────────────────────────────────────────────
    html.Div([
        html.H3("Key Business Insights", style={"color": TEXT, "fontWeight": "700",
                                                  "marginBottom": "16px", "fontSize": "1.1rem"}),
        html.Div(id="insights-section",
                 style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(300px, 1fr))",
                        "gap": "14px"})
    ], style={"padding": "0 32px 40px"}),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Div("Customer Personality Analysis Dashboard  ·  CodeAlpha Internship  ·  Built with Plotly Dash",
             style={"textAlign": "center", "color": TEXT, "opacity": "0.3",
                    "fontSize": "0.75rem", "padding": "16px",
                    "borderTop": f"1px solid {BORDER}"}),
])


# ─── Callbacks ───────────────────────────────────────────────────────────────

def filter_df(edu, mar, income_range):
    dff = df.copy()
    if edu != "All" and "education" in dff.columns:
        dff = dff[dff["education"] == edu]
    if mar != "All" and "marital_status" in dff.columns:
        dff = dff[dff["marital_status"] == mar]
    if "income" in dff.columns:
        dff = dff[(dff["income"] >= income_range[0]) & (dff["income"] <= income_range[1])]
    return dff


@app.callback(
    Output("kpi-row", "children"),
    Output("chart-spend-edu", "figure"),
    Output("chart-age-hist", "figure"),
    Output("chart-scatter", "figure"),
    Output("chart-pie", "figure"),
    Output("chart-campaign", "figure"),
    Output("chart-heatmap", "figure"),
    Output("insights-section", "children"),
    Input("edu-filter", "value"),
    Input("mar-filter", "value"),
    Input("income-filter", "value"),
)
def update_all(edu, mar, income_range):
    dff = filter_df(edu, mar, income_range)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    n_customers  = f"{len(dff):,}"
    avg_income   = f"${dff['income'].dropna().mean():,.0f}"   if "income"      in dff.columns and len(dff['income'].dropna()) > 0 else "—"
    avg_spend    = f"${dff['total_spend'].mean():,.0f}" if "total_spend" in dff.columns and dff['total_spend'].notna().any() else "—"
    resp_rate    = f"{dff['response'].mean()*100:.1f}%" if "response"   in dff.columns and dff['response'].notna().any() else "—"
    avg_recency  = f"{dff['recency'].mean():.0f} days"  if "recency"    in dff.columns and dff['recency'].notna().any() else "—"

    kpis = html.Div([
        kpi_card("Total Customers",   n_customers,  "in filtered view",     ACCENT),
        kpi_card("Avg Annual Income", avg_income,   "per customer",         "#4CC9F0"),
        kpi_card("Avg Total Spend",   avg_spend,    "across all categories","#90BE6D"),
        kpi_card("Campaign Response", resp_rate,    "final offer acceptance","#F72585"),
        kpi_card("Avg Recency",       avg_recency,  "days since last purchase","#F9C74F"),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "width": "100%"})

    # ── Chart 1: Spend by Education ───────────────────────────────────────────
    if all(c in dff.columns for c in ["education", "total_spend"]):
        agg  = dff.groupby("education")["total_spend"].mean().sort_values(ascending=False).reset_index()
        fig1 = px.bar(agg, x="education", y="total_spend",
                      color="education", color_discrete_sequence=PALETTE,
                      title="Avg Spend by Education Level",
                      labels={"total_spend": "Avg Spend ($)", "education": "Education"})
        fig1.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=0.5)
        apply_theme(fig1)
        fig1.update_layout(showlegend=False)
    else:
        fig1 = go.Figure()

    # ── Chart 2: Age distribution ─────────────────────────────────────────────
    if "age" in dff.columns:
        fig2 = px.histogram(dff, x="age", nbins=30, title="Customer Age Distribution",
                            color_discrete_sequence=[ACCENT],
                            labels={"age": "Age (years)"})
        fig2.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=0.5)
        apply_theme(fig2)
    else:
        fig2 = go.Figure()

    # ── Chart 3: Income vs Spend scatter ─────────────────────────────────────
    if all(c in dff.columns for c in ["income", "total_spend"]):
        color_col = "education" if "education" in dff.columns else None
        fig3 = px.scatter(dff.sample(min(800, len(dff)), random_state=42),
                          x="income", y="total_spend",
                          color=color_col, color_discrete_sequence=PALETTE,
                          opacity=0.6, size_max=8,
                          trendline="ols",
                          title="Income vs Total Spend",
                          labels={"income": "Annual Income ($)", "total_spend": "Total Spend ($)"})
        apply_theme(fig3)
    else:
        fig3 = go.Figure()

    # ── Chart 4: Marital Status pie ───────────────────────────────────────────
    if "marital_status" in dff.columns:
        counts = dff["marital_status"].value_counts().reset_index()
        fig4 = px.pie(counts, names="marital_status", values="count",
                      title="Marital Status Distribution",
                      color_discrete_sequence=PALETTE, hole=0.5)
        fig4.update_traces(textfont_color=TEXT)
        apply_theme(fig4)
    else:
        fig4 = go.Figure()

    # ── Chart 5: Campaign acceptance rates ───────────────────────────────────
    cmp_cols = [c for c in dff.columns if c.startswith("acceptedcmp")]
    if cmp_cols:
        rates = {c.replace("acceptedcmp", "Campaign "): dff[c].mean() * 100 for c in cmp_cols}
        if "response" in dff.columns:
            rates["Final Offer"] = dff["response"].mean() * 100
        fig5 = go.Figure(go.Bar(
            x=list(rates.keys()), y=list(rates.values()),
            marker_color=PALETTE[:len(rates)],
            text=[f"{v:.1f}%" for v in rates.values()],
            textposition="outside",
            textfont=dict(color=TEXT, size=11)
        ))
        fig5.update_layout(title="Campaign Acceptance Rates (%)",
                            yaxis_title="Rate (%)", xaxis_title="Campaign")
        apply_theme(fig5)
    else:
        fig5 = go.Figure()

    # ── Chart 6: Correlation heatmap ─────────────────────────────────────────
    key_cols = [c for c in ["income","age","total_spend","total_purchases",
                              "total_campaigns_accepted","recency",
                              "total_children","response"] if c in dff.columns]
    if len(key_cols) >= 4:
        corr_m = dff[key_cols].corr().round(2)
        fig6 = go.Figure(go.Heatmap(
            z=corr_m.values, x=corr_m.columns, y=corr_m.columns,
            colorscale="RdBu", zmid=0,
            text=corr_m.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 9, "color": "white"},
            hoverongaps=False,
        ))
        fig6.update_layout(title="Feature Correlation Heatmap")
        apply_theme(fig6)
    else:
        fig6 = go.Figure()

    # ── Insights cards ────────────────────────────────────────────────────────
    insight_data = [
        ("💰", "Income Drives Spend",        "Higher income customers are primary revenue drivers. Target with premium loyalty tiers.", ACCENT),
        ("🎓", "Education Signals Value",     "Postgraduate customers spend significantly more. Focus on LinkedIn-style acquisition.", "#4CC9F0"),
        ("👨‍👩‍👧", "Children Reduce Spend",    "Customers with children are more price-sensitive. Offer value bundles for families.", "#F9C74F"),
        ("📣", "Campaign 4 Outperforms",      "Campaign 4 shows highest acceptance rates. Replicate its creative strategy.", "#90BE6D"),
        ("⏱️", "Recency Predicts Response",  "Recent buyers respond ~2× better. Trigger campaigns within 30 days of purchase.", "#F72585"),
        ("🏆", "Middle-Aged Spend Most",      "40–60 year olds are peak spenders. Prioritise this segment in media buys.", "#7209B7"),
        ("🌐", "Web Buyers Are Deal-Driven",  "Online shoppers are coupon-sensitive. Deploy web-exclusive flash offers.", ACCENT),
        ("💍", "Dual-Income HH Earn More",    "Married/Together customers have highest incomes. Target with premium bundles.", "#F3722C"),
        ("🔁", "Multi-Campaign = Higher LTV", "Customers accepting 2+ campaigns spend significantly more. Fast-track to VIP.", "#4895EF"),
        ("📅", "Tenure Builds Revenue",       "Long-tenured customers spend more. Invest in onboarding and retention journeys.", "#560BAD"),
    ]
    cards = [
        html.Div([
            html.Div(icon, style={"fontSize": "1.6rem", "marginBottom": "8px"}),
            html.H5(title, style={"color": color, "fontWeight": "700",
                                   "fontSize": "0.9rem", "marginBottom": "6px"}),
            html.P(desc, style={"color": TEXT, "opacity": "0.7",
                                  "fontSize": "0.8rem", "lineHeight": "1.5", "margin": "0"}),
        ], style={"background": CARD_BG, "borderRadius": "10px", "padding": "18px",
                  "border": f"1px solid {BORDER}", "borderTop": f"3px solid {color}"})
        for icon, title, desc, color in insight_data
    ]

    return kpis, fig1, fig2, fig3, fig4, fig5, fig6, cards


# ─── Run ─────────────────────────────────────────────────────────────────────
def open_browser(url: str, delay: float = 1.0):
    def _open():
        import time
        time.sleep(delay)
        opened = False
        try:
            opened = webbrowser.open_new_tab(url)
        except Exception:
            opened = False
        if not opened:
            try:
                if platform.system() == "Windows":
                    os.startfile(url)
                elif platform.system() == "Darwin":
                    os.system(f"open {url}")
                else:
                    os.system(f"xdg-open {url}")
            except Exception:
                pass
    threading.Thread(target=_open, daemon=True).start()

if __name__ == "__main__":
    url = "http://127.0.0.1:8050"
    print("\n" + "="*55)
    print("  Customer Personality Analysis Dashboard")
    print(f"  Open: {url}")
    print("="*55 + "\n")
    open_browser(url)
    app.run(debug=False, host="127.0.0.1", port=8050, use_reloader=False)
