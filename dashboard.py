#!/usr/bin/env python3
"""
Sector Rotation Dashboard
=========================
Interactive web dashboard for the RRG-based Sector Rotation TAA model.

Usage:
    python dashboard.py                  # http://127.0.0.1:8050
    python dashboard.py --port 8080      # custom port
    python dashboard.py --offline        # skip Yahoo Finance, use synthetic data

Author : Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

# Import all engine functions from the existing module
from sector_rrg_taa import (
    BENCHMARK,
    CASH_PROXY,
    COLORS,
    SECTOR_NAMES,
    SECTOR_TICKERS,
    _compute_metrics,
    _yahoo_reachable,
    backtest,
    composite_signal,
    compute_rrg,
    construct_portfolio,
    factor_scores,
    fetch_data,
)

# ── Colour constants ────────────────────────────────────────────────
QUADRANT_COLORS = {
    "Leading": "#22c55e",
    "Weakening": "#f97316",
    "Lagging": "#ef4444",
    "Improving": "#3b82f6",
}
SIGNAL_COLORS = {
    "Overweight": "#22c55e",
    "Hold": "#eab308",
    "Underweight": "#ef4444",
}
BG_COLOR = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"


# ── Pre-compute data ────────────────────────────────────────────────
def run_engine(offline: bool = False, **kwargs):
    """Run the full engine and return all data needed for the dashboard."""
    horizon_weeks = kwargs.get("horizon_weeks", 5)
    mom_weeks = kwargs.get("mom_weeks", 2)
    sensitivity = kwargs.get("sensitivity", 6)
    top_n = kwargs.get("top_n", 3)
    max_weight = kwargs.get("max_weight", 0.25)
    rrg_weight = kwargs.get("rrg_weight", 0.70)
    factor_weight = kwargs.get("factor_weight", 0.30)
    vol_target = kwargs.get("vol_target", 0.12)
    years = kwargs.get("years", 7)

    weekly, daily = fetch_data(SECTOR_TICKERS, years=years, offline=offline)

    import sector_rrg_taa
    is_offline = offline or not sector_rrg_taa._yahoo_reachable

    rrg_df = compute_rrg(
        weekly, SECTOR_TICKERS, BENCHMARK,
        rs_lookback=horizon_weeks,
        mom_lookback=mom_weeks,
        sensitivity=sensitivity,
    )

    latest_date = rrg_df["date"].max()
    rrg_latest = rrg_df[rrg_df["date"] == latest_date].copy()

    factors = factor_scores(weekly, daily, SECTOR_TICKERS, offline=is_offline)
    signals = composite_signal(rrg_latest, factors, rrg_weight, factor_weight)

    portfolio = construct_portfolio(
        signals, daily, max_sectors=top_n,
        max_weight=max_weight, vol_target=vol_target,
    )

    bt, metrics = backtest(
        weekly, daily, SECTOR_TICKERS, BENCHMARK, CASH_PROXY,
        rs_lookback=horizon_weeks, mom_lookback=mom_weeks,
        sensitivity=sensitivity, top_n=top_n, max_weight=max_weight,
        vol_target=vol_target,
    )

    return {
        "weekly": weekly, "daily": daily,
        "rrg_df": rrg_df, "signals": signals,
        "portfolio": portfolio, "bt": bt, "metrics": metrics,
        "is_offline": is_offline, "latest_date": latest_date,
    }


# ── Plotly figure builders ──────────────────────────────────────────
def build_rrg_figure(rrg_df: pd.DataFrame, signals: pd.DataFrame, tail_length: int = 8) -> go.Figure:
    fig = go.Figure()

    # Quadrant shading
    fig.add_shape(type="rect", x0=100, x1=115, y0=100, y1=115,
                  fillcolor="rgba(34,197,94,0.07)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, x1=115, y0=85, y1=100,
                  fillcolor="rgba(249,115,22,0.07)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=85, x1=100, y0=85, y1=100,
                  fillcolor="rgba(239,68,68,0.07)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=85, x1=100, y0=100, y1=115,
                  fillcolor="rgba(59,130,246,0.07)", line_width=0, layer="below")

    # Crosshairs at 100
    fig.add_hline(y=100, line_dash="dot", line_color="#475569", line_width=1)
    fig.add_vline(x=100, line_dash="dot", line_color="#475569", line_width=1)

    # Quadrant labels
    for label, x, y in [
        ("LEADING", 107, 107), ("WEAKENING", 107, 93),
        ("LAGGING", 93, 93), ("IMPROVING", 93, 107),
    ]:
        fig.add_annotation(
            x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=13, color=QUADRANT_COLORS.get(label.title(), MUTED)),
            opacity=0.35,
        )

    dates = sorted(rrg_df["date"].unique())
    tail_dates = dates[-tail_length:] if len(dates) >= tail_length else dates

    comp_map = {}
    if "composite_norm" in signals.columns:
        comp_map = dict(zip(signals["ticker"], signals["composite_norm"]))

    for ticker in rrg_df["ticker"].unique():
        color = COLORS.get(ticker, "#94a3b8")
        name = SECTOR_NAMES.get(ticker, ticker)
        sub = rrg_df[(rrg_df["ticker"] == ticker) & (rrg_df["date"].isin(tail_dates))]
        sub = sub.sort_values("date")
        if sub.empty:
            continue

        xs = sub["rs_ratio"].values
        ys = sub["rs_momentum"].values

        # Tail
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=2), opacity=0.5,
            showlegend=False, hoverinfo="skip",
        ))

        # Current position bubble
        latest = sub.iloc[-1]
        comp = comp_map.get(ticker, 0.5)
        size = 14 + comp * 26
        fig.add_trace(go.Scatter(
            x=[latest["rs_ratio"]], y=[latest["rs_momentum"]],
            mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=1.5, color="white")),
            text=[ticker], textposition="top right",
            textfont=dict(size=11, color=color, family="Arial Black"),
            name=f"{name} ({ticker})",
            hovertemplate=(
                f"<b>{name} ({ticker})</b><br>"
                "RS-Ratio: %{x:.2f}<br>"
                "RS-Momentum: %{y:.2f}<br>"
                f"Quadrant: {latest['quadrant']}<br>"
                f"Composite: {comp:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        title=dict(text="Relative Rotation Graph", font=dict(size=20, color=TEXT_COLOR)),
        xaxis=dict(title="RS-Ratio", gridcolor="#1e293b"),
        yaxis=dict(title="RS-Momentum", gridcolor="#1e293b"),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(font=dict(size=10, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        height=520,
    )
    return fig


def build_equity_figure(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if bt.empty:
        fig.add_annotation(text="No backtest data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=MUTED))
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG_COLOR,
                          plot_bgcolor=BG_COLOR, height=400)
        return fig

    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["port_cumret"], name="Strategy",
        line=dict(color=ACCENT, width=2.5),
        hovertemplate="Strategy: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["bm_cumret"], name="SPY Benchmark",
        line=dict(color="#64748b", width=1.5, dash="dash"),
        hovertemplate="SPY: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        title=dict(text="Backtest: Growth of $1", font=dict(size=20, color=TEXT_COLOR)),
        yaxis=dict(title="Cumulative Return ($)", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    return fig


def build_drawdown_figure(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if bt.empty:
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG_COLOR,
                          plot_bgcolor=BG_COLOR, height=200)
        return fig

    cum = bt["port_cumret"]
    dd = (cum - cum.cummax()) / cum.cummax()

    fig.add_trace(go.Scatter(
        x=bt.index, y=dd * 100, fill="tozeroy",
        line=dict(color="#ef4444", width=1),
        fillcolor="rgba(239,68,68,0.3)",
        hovertemplate="Drawdown: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        title=dict(text="Drawdown", font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(title="Drawdown %", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        margin=dict(l=60, r=30, t=50, b=40),
        height=220,
    )
    return fig


def build_portfolio_figure(portfolio: pd.DataFrame) -> go.Figure:
    labels = []
    for _, r in portfolio.iterrows():
        t = r["ticker"]
        labels.append(SECTOR_NAMES.get(t, t))

    colors = [COLORS.get(r["ticker"], "#64748b") for _, r in portfolio.iterrows()]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=portfolio["weight"],
        marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        hole=0.45,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        title=dict(text="Portfolio Allocation", font=dict(size=18, color=TEXT_COLOR)),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(font=dict(size=10, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        height=350,
    )
    return fig


# ── HTML helper builders ────────────────────────────────────────────
def metric_card(label: str, value: str, color: str = ACCENT) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase",
                                "letterSpacing": "0.5px", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": color}),
    ], style={
        "backgroundColor": CARD_BG, "borderRadius": "10px", "padding": "16px 20px",
        "flex": "1", "minWidth": "130px",
    })


def signal_badge(signal: str) -> html.Span:
    c = SIGNAL_COLORS.get(signal, MUTED)
    return html.Span(signal, style={
        "backgroundColor": c, "color": "white" if signal != "Hold" else "#1e293b",
        "padding": "3px 10px", "borderRadius": "12px", "fontSize": "12px", "fontWeight": "600",
    })


def build_signal_table(signals: pd.DataFrame) -> html.Table:
    header = html.Tr([
        html.Th(h, style={"padding": "10px 14px", "textAlign": a, "color": MUTED,
                           "fontSize": "11px", "textTransform": "uppercase",
                           "borderBottom": f"1px solid #334155", "letterSpacing": "0.5px"})
        for h, a in [("Sector", "left"), ("Ticker", "left"), ("RS-Ratio", "right"),
                      ("RS-Mom", "right"), ("Composite", "right"), ("Quadrant", "left"),
                      ("Signal", "center")]
    ])

    rows = []
    for _, row in signals.sort_values("composite_norm", ascending=False).iterrows():
        ticker = row["ticker"]
        name = SECTOR_NAMES.get(ticker, ticker)
        quad = row.get("quadrant", "?")
        quad_color = QUADRANT_COLORS.get(quad, MUTED)
        rec = row.get("recommendation", "?")

        rows.append(html.Tr([
            html.Td(name, style={"fontWeight": "600"}),
            html.Td(ticker, style={"color": COLORS.get(ticker, MUTED), "fontWeight": "700"}),
            html.Td(f"{row.get('rs_ratio', 0):.2f}", style={"textAlign": "right"}),
            html.Td(f"{row.get('rs_momentum', 0):.2f}", style={"textAlign": "right"}),
            html.Td(f"{row.get('composite_norm', 0):.2f}", style={"textAlign": "right"}),
            html.Td(quad, style={"color": quad_color, "fontWeight": "600"}),
            html.Td(signal_badge(rec), style={"textAlign": "center"}),
        ], style={"borderBottom": "1px solid #1e293b"}))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "width": "100%", "borderCollapse": "collapse", "fontSize": "13px",
            "color": TEXT_COLOR,
        },
    )


# ── Build the Dash app ──────────────────────────────────────────────
def create_app(data: dict) -> Dash:
    signals = data["signals"]
    portfolio = data["portfolio"]
    bt = data["bt"]
    metrics = data["metrics"]
    rrg_df = data["rrg_df"]
    is_offline = data["is_offline"]
    latest_date = data["latest_date"]

    app = Dash(__name__)
    app.title = "Sector Rotation Dashboard"

    # Figures
    rrg_fig = build_rrg_figure(rrg_df, signals)
    equity_fig = build_equity_figure(bt)
    dd_fig = build_drawdown_figure(bt)
    port_fig = build_portfolio_figure(portfolio)

    # Metric cards row
    metric_cards = []
    metric_defs = [
        ("CAGR", metrics.get("CAGR (Strategy)", "N/A"), ACCENT),
        ("Sharpe", metrics.get("Sharpe Ratio", "N/A"), ACCENT),
        ("Max DD", metrics.get("Max Drawdown", "N/A"), "#ef4444"),
        ("Sortino", metrics.get("Sortino Ratio", "N/A"), ACCENT),
        ("Win Rate", metrics.get("Win Rate", "N/A"), "#22c55e"),
        ("Vol", metrics.get("Ann. Volatility", "N/A"), "#f97316"),
        ("SPY CAGR", metrics.get("CAGR (SPY)", "N/A"), "#64748b"),
    ]
    for label, val, color in metric_defs:
        metric_cards.append(metric_card(label, val, color))

    data_badge = html.Span(
        "SYNTHETIC DATA" if is_offline else "LIVE DATA",
        style={
            "backgroundColor": "#f97316" if is_offline else "#22c55e",
            "color": "white", "padding": "4px 12px", "borderRadius": "12px",
            "fontSize": "11px", "fontWeight": "700", "marginLeft": "12px",
            "verticalAlign": "middle",
        },
    )

    app.layout = html.Div([
        # ── Header
        html.Div([
            html.H1([
                "Sector Rotation Dashboard",
                data_badge,
            ], style={"margin": 0, "fontSize": "28px", "fontWeight": "800", "color": TEXT_COLOR}),
            html.P(
                f"RRG Tactical Allocation Model  |  As of {latest_date.strftime('%Y-%m-%d')}",
                style={"color": MUTED, "margin": "4px 0 0 0", "fontSize": "13px"},
            ),
        ], style={"padding": "24px 32px 12px 32px"}),

        # ── Metrics bar
        html.Div(metric_cards, style={
            "display": "flex", "gap": "12px", "padding": "8px 32px 16px 32px",
            "flexWrap": "wrap",
        }),

        # ── Main grid: RRG + Signal table
        html.Div([
            # Left: RRG
            html.Div([
                dcc.Graph(figure=rrg_fig, config={"displayModeBar": False}),
            ], style={"flex": "1.2", "minWidth": "480px"}),

            # Right: Signal table
            html.Div([
                html.H3("Tactical Signals", style={
                    "color": TEXT_COLOR, "fontSize": "18px", "fontWeight": "700",
                    "margin": "0 0 12px 0",
                }),
                html.Div(
                    build_signal_table(signals),
                    style={
                        "backgroundColor": CARD_BG, "borderRadius": "10px",
                        "padding": "12px 16px", "overflowX": "auto",
                    },
                ),
            ], style={"flex": "1", "minWidth": "420px"}),
        ], style={
            "display": "flex", "gap": "20px", "padding": "0 32px",
            "flexWrap": "wrap",
        }),

        # ── Second row: Portfolio + Equity + Drawdown
        html.Div([
            # Left: Donut
            html.Div([
                dcc.Graph(figure=port_fig, config={"displayModeBar": False}),
            ], style={"flex": "0.7", "minWidth": "300px"}),

            # Right: Equity + Drawdown stacked
            html.Div([
                dcc.Graph(figure=equity_fig, config={"displayModeBar": False}),
                dcc.Graph(figure=dd_fig, config={"displayModeBar": False}),
            ], style={"flex": "1.3", "minWidth": "500px"}),
        ], style={
            "display": "flex", "gap": "20px", "padding": "16px 32px",
            "flexWrap": "wrap",
        }),

        # ── Footer
        html.Div([
            html.P(
                "Sector Rotation RRG TAA Model | Built with Dash & Plotly | Data: Yahoo Finance / Synthetic",
                style={"color": "#475569", "fontSize": "11px", "textAlign": "center", "margin": "8px"},
            ),
        ], style={"padding": "0 32px 20px 32px"}),

    ], style={
        "backgroundColor": BG_COLOR, "minHeight": "100vh",
        "fontFamily": "'Inter', 'Segoe UI', -apple-system, sans-serif",
    })

    return app


# ── Entry point ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sector Rotation Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--offline", action="store_true", help="Use synthetic data")
    parser.add_argument("--years", type=int, default=7, help="Years of history")
    parser.add_argument("--horizon_weeks", type=int, default=5)
    parser.add_argument("--mom_weeks", type=int, default=2)
    parser.add_argument("--sensitivity", type=int, default=6)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--max_weight", type=float, default=0.25)
    parser.add_argument("--vol_target", type=float, default=0.12)
    args = parser.parse_args()

    print("[INFO] Running engine ...")
    data = run_engine(
        offline=args.offline, years=args.years,
        horizon_weeks=args.horizon_weeks, mom_weeks=args.mom_weeks,
        sensitivity=args.sensitivity, top_n=args.top_n,
        max_weight=args.max_weight, vol_target=args.vol_target,
    )

    print("[INFO] Building dashboard ...")
    app = create_app(data)

    print(f"\n  Dashboard ready at: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
