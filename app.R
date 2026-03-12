#!/usr/bin/env Rscript
# =============================================================================
# Sector & Industry Rotation — Interactive Shiny Dashboard
# =============================================================================
# Tabbed dark-theme dashboard powered by the sector_rotation.R engine.
#
# Tabs:
#   1. Sector Overview   — RRG, signals table, portfolio donut, equity curve
#   2. Industry Rotation — Industry-level RRG grouped by sector, signal table
#   3. Economic Cycle    — Regime phase indicator, score bars, indicator table
#   4. Momentum Heatmap  — Cross-sectional z-score heatmap (sectors + industries)
#
# Usage:
#   Rscript -e "shiny::runApp('app.R', port=8050)"
#   or: shiny::runApp()
# =============================================================================

.sourced_by_app <- TRUE
source("sector_rotation.R")

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(plotly)
  library(DT)
})

# ─────────────────────────────────────────────────────────────────────────────
# Theme & colour constants
# ─────────────────────────────────────────────────────────────────────────────
BG      <- "#0f172a"
CARD_BG <- "#1e293b"
TEXT    <- "#e2e8f0"
MUTED   <- "#94a3b8"
ACCENT  <- "#38bdf8"

dark_theme <- bs_theme(
  bootswatch  = "darkly",
  bg          = BG,
  fg          = TEXT,
  primary     = ACCENT,
  base_font   = font_google("Inter")
)

SIGNAL_COLORS <- c(Overweight = "#22c55e", Hold = "#eab308", Underweight = "#ef4444")

plotly_dark_layout <- function(fig, title = NULL, h = 480, ...) {
  fig %>% layout(
    title       = if (!is.null(title)) list(text=title, font=list(size=18, color=TEXT)) else NULL,
    paper_bgcolor = BG, plot_bgcolor = BG,
    font        = list(color = TEXT),
    xaxis       = list(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis       = list(gridcolor="#1e293b", zerolinecolor="#334155"),
    margin      = list(l=60, r=30, t=if(!is.null(title)) 60 else 30, b=50),
    legend      = list(font=list(color=MUTED), bgcolor="rgba(0,0,0,0)"),
    height      = h,
    ...
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

build_rrg_figure <- function(rrg_df, signals, colors_map,
                              name_map, tail_length = 8L, title = "Relative Rotation Graph") {
  fig <- plot_ly()

  # Quadrant shading
  for (shape_args in list(
    list(x0=100,x1=115,y0=100,y1=115, col="rgba(34,197,94,0.07)"),
    list(x0=100,x1=115,y0=85, y1=100, col="rgba(249,115,22,0.07)"),
    list(x0=85, x1=100,y0=85, y1=100, col="rgba(239,68,68,0.07)"),
    list(x0=85, x1=100,y0=100,y1=115, col="rgba(59,130,246,0.07)")
  )) {
    fig <- fig %>% layout(shapes = list(
      list(type="rect", x0=shape_args$x0, x1=shape_args$x1,
           y0=shape_args$y0, y1=shape_args$y1,
           fillcolor=shape_args$col, line=list(width=0), layer="below")
    ))
  }

  # Crosshairs
  fig <- fig %>%
    add_lines(x=c(100,100), y=c(85,115), line=list(color="#475569",dash="dot",width=1),
              showlegend=FALSE, hoverinfo="skip") %>%
    add_lines(x=c(85,115), y=c(100,100), line=list(color="#475569",dash="dot",width=1),
              showlegend=FALSE, hoverinfo="skip")

  # Quadrant annotations
  for (ann in list(
    list(x=108,y=108,txt="<b>LEADING</b>",   col=QUADRANT_COLORS["Leading"]),
    list(x=108,y=92, txt="<b>WEAKENING</b>",  col=QUADRANT_COLORS["Weakening"]),
    list(x=92, y=92, txt="<b>LAGGING</b>",    col=QUADRANT_COLORS["Lagging"]),
    list(x=92, y=108,txt="<b>IMPROVING</b>",  col=QUADRANT_COLORS["Improving"])
  )) {
    fig <- fig %>% add_annotations(
      x=ann$x, y=ann$y, text=ann$txt, showarrow=FALSE,
      font=list(size=12, color=ann$col), opacity=0.35
    )
  }

  dates       <- sort(unique(rrg_df$date))
  tail_dates  <- tail(dates, tail_length)
  comp_map    <- setNames(signals$composite_norm, signals$ticker)

  for (ticker in unique(rrg_df$ticker)) {
    color <- colors_map[[ticker]] %||% MUTED
    name  <- name_map[[ticker]]  %||% ticker
    sub   <- rrg_df %>% filter(ticker == !!ticker, date %in% tail_dates) %>% arrange(date)
    if (nrow(sub) == 0) next

    # Tail
    fig <- fig %>% add_trace(
      x=sub$rs_ratio, y=sub$rs_momentum, type="scatter", mode="lines",
      line=list(color=color, width=2), opacity=0.5,
      showlegend=FALSE, hoverinfo="skip"
    )

    # Latest point bubble
    latest <- tail(sub, 1)
    comp   <- comp_map[[ticker]] %||% 0.5
    size   <- 12 + comp * 24
    fig    <- fig %>% add_trace(
      x=latest$rs_ratio, y=latest$rs_momentum,
      type="scatter", mode="markers+text",
      marker=list(size=size, color=color, line=list(width=1.5, color="white")),
      text=ticker, textposition="top right",
      textfont=list(size=10, color=color, family="Arial Black"),
      name=sprintf("%s (%s)", name, ticker),
      hovertemplate=paste0(
        "<b>", name, " (", ticker, ")</b><br>",
        "RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br>",
        "Quadrant: ", latest$quadrant, "<br>",
        "Composite: ", round(comp,2), "<extra></extra>"
      )
    )
  }

  fig %>% plotly_dark_layout(title = title) %>%
    layout(xaxis=list(title="RS-Ratio"), yaxis=list(title="RS-Momentum"))
}


build_equity_figure <- function(equity) {
  if (nrow(equity) == 0) {
    return(plot_ly() %>% add_annotations(
      text="No backtest data", xref="paper", yref="paper",
      x=0.5, y=0.5, showarrow=FALSE, font=list(size=18, color=MUTED)
    ) %>% plotly_dark_layout(h=380))
  }
  plot_ly() %>%
    add_trace(x=equity$date, y=equity$port_cumret, type="scatter", mode="lines",
              name="Strategy", line=list(color=ACCENT, width=2.5),
              hovertemplate="Strategy: $%{y:.3f}<extra></extra>") %>%
    add_trace(x=equity$date, y=equity$bm_cumret,   type="scatter", mode="lines",
              name="SPY Benchmark", line=list(color="#64748b", width=1.5, dash="dash"),
              hovertemplate="SPY: $%{y:.3f}<extra></extra>") %>%
    plotly_dark_layout("Backtest: Growth of $1", h=380) %>%
    layout(yaxis=list(title="Cumulative Return ($)"))
}


build_drawdown_figure <- function(equity) {
  if (nrow(equity) == 0) return(plot_ly() %>% plotly_dark_layout(h=200))
  dd <- (equity$port_cumret - cummax(equity$port_cumret)) / cummax(equity$port_cumret) * 100
  plot_ly(x=equity$date, y=dd, type="scatter", mode="lines",
          fill="tozeroy", fillcolor="rgba(239,68,68,0.3)",
          line=list(color="#ef4444", width=1),
          hovertemplate="Drawdown: %{y:.1f}%<extra></extra>", showlegend=FALSE) %>%
    plotly_dark_layout("Drawdown", h=210) %>%
    layout(yaxis=list(title="Drawdown %"))
}


build_portfolio_donut <- function(portfolio) {
  labels <- map_chr(portfolio$ticker, ~ SECTOR_NAMES[[.x]] %||% .x)
  colors <- map_chr(portfolio$ticker, ~ SECTOR_COLORS[[.x]] %||% "#64748b")
  plot_ly(labels=labels, values=portfolio$weight, type="pie",
          marker=list(colors=colors, line=list(color=BG, width=2)),
          textinfo="label+percent",
          textfont=list(size=11, color="white"),
          hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
          hole=0.45) %>%
    plotly_dark_layout("Portfolio Allocation", h=340)
}


build_regime_figure <- function(regime) {
  scores <- regime$scores
  phases <- names(scores)
  cols   <- map_chr(phases, ~ PHASE_COLORS[[.x]] %||% MUTED)

  plot_ly(x=scores, y=phases, type="bar", orientation="h",
          marker=list(color=cols),
          text=sprintf("%.1f%%", scores*100), textposition="outside",
          hovertemplate="<b>%{y}</b>: %{x:.1%}<extra></extra>",
          showlegend=FALSE) %>%
    plotly_dark_layout("Economic Cycle Phase Scores", h=280) %>%
    layout(xaxis=list(title="Score", tickformat=".0%"),
           yaxis=list(categoryorder="array",
                      categoryarray=rev(c("Recovery","Expansion","Slowdown","Contraction"))))
}


build_indicator_figure <- function(regime) {
  ind    <- regime$indicators
  nm     <- names(ind)
  vals   <- unlist(ind)
  cols   <- ifelse(vals >= 0, "#22c55e", "#ef4444")

  plot_ly(x=vals, y=nm, type="bar", orientation="h",
          marker=list(color=cols),
          text=sprintf("%+.3f", vals), textposition="outside",
          hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
          showlegend=FALSE) %>%
    plotly_dark_layout("Regime Indicator Trends (positive = bullish signal)", h=340) %>%
    layout(xaxis=list(title="13-week ratio trend"),
           yaxis=list(categoryorder="total ascending"))
}


build_heatmap_figure <- function(heatmap_df, group_filter = NULL, title = "Momentum Heatmap") {
  lb_cols  <- names(heatmap_df)[grepl("^W\\d+$", names(heatmap_df))]
  col_lbls <- c("1M","2M","3M","6M","1Y")[seq_along(lb_cols)]

  df <- heatmap_df
  if (!is.null(group_filter)) df <- df %>% filter(group %in% group_filter)
  if (nrow(df) == 0) return(plot_ly() %>% plotly_dark_layout(h=400))

  df <- df %>% arrange(group, name)
  z  <- as.matrix(df %>% select(all_of(lb_cols)))

  plot_ly(
    x    = col_lbls,
    y    = df$name,
    z    = z,
    type = "heatmap",
    colorscale = list(
      list(0,   "#ef4444"),
      list(0.5, "#1e293b"),
      list(1,   "#22c55e")
    ),
    zmin = -2.5, zmax = 2.5,
    text = matrix(sprintf("%.2f", z), nrow=nrow(z)),
    hovertemplate = "<b>%{y}</b> | %{x}<br>Z-score: %{z:.2f}<extra></extra>",
    showscale = TRUE,
    colorbar  = list(title="Z-score", tickfont=list(color=TEXT))
  ) %>%
    plotly_dark_layout(title, h = max(300, nrow(df) * 22 + 80)) %>%
    layout(xaxis=list(side="top", title=""),
           yaxis=list(title="", tickfont=list(size=11)))
}

# ─────────────────────────────────────────────────────────────────────────────
# UI helper components
# ─────────────────────────────────────────────────────────────────────────────

metric_box <- function(label, value, color = ACCENT) {
  div(
    style = sprintf(
      "background:%s;border-radius:10px;padding:14px 18px;flex:1;min-width:120px;",
      CARD_BG
    ),
    div(style = sprintf("font-size:10px;color:%s;text-transform:uppercase;letter-spacing:.5px;", MUTED),
        label),
    div(style = sprintf("font-size:22px;font-weight:700;color:%s;margin-top:4px;", color),
        value)
  )
}

signal_badge_html <- function(sig) {
  col <- SIGNAL_COLORS[[sig]] %||% MUTED
  sprintf(
    '<span style="background:%s;color:white;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:600;">%s</span>',
    col, htmltools::htmlEscape(sig)
  )
}

build_signal_table_ui <- function(signals, name_col = "sector_name") {
  tbl <- signals %>% arrange(desc(composite_norm)) %>%
    mutate(
      Sector    = if (name_col %in% names(.)) .[[name_col]] else ticker,
      Ticker    = ticker,
      `RS-Ratio`  = round(rs_ratio, 2),
      `RS-Mom`    = round(rs_momentum, 2),
      Composite   = round(composite_norm, 2),
      Quadrant    = quadrant,
      Signal      = recommendation
    ) %>%
    select(Sector, Ticker, `RS-Ratio`, `RS-Mom`, Composite, Quadrant, Signal)

  datatable(
    tbl,
    escape    = FALSE,
    rownames  = FALSE,
    selection = "none",
    options   = list(
      pageLength = 15, dom = "t", ordering = TRUE,
      columnDefs = list(list(className="dt-center", targets=c(2,3,4,5,6)))
    ),
    class = "compact hover"
  ) %>%
    formatStyle("Signal",
      backgroundColor = styleEqual(
        c("Overweight","Hold","Underweight"),
        c("#14532d","#422006","#450a0a")
      ),
      color = "white", fontWeight = "bold"
    ) %>%
    formatStyle("Quadrant",
      color = styleEqual(
        c("Leading","Weakening","Lagging","Improving"),
        unname(QUADRANT_COLORS[c("Leading","Weakening","Lagging","Improving")])
      ),
      fontWeight = "bold"
    ) %>%
    formatStyle(0, target="row",
      backgroundColor = styleEqual(character(0), character(0)),
      color = TEXT
    )
}

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

ui <- page_navbar(
  title = tags$span(
    style = "font-weight:800;font-size:20px;",
    "Sector & Industry Rotation"
  ),
  theme = dark_theme,
  bg    = CARD_BG,
  fillable = FALSE,

  # ── Tab 1: Sector Overview ────────────────────────────────────────────────
  nav_panel(
    "Sector Overview",
    div(
      style = sprintf("background:%s;min-height:100vh;padding:16px 24px;", BG),

      # Metrics bar
      uiOutput("metrics_bar"),
      br(),

      # RRG + Signal table
      fluidRow(
        column(7, card(
          card_header("Relative Rotation Graph — Sectors vs SPY"),
          plotlyOutput("sector_rrg_plot", height = "500px")
        )),
        column(5, card(
          card_header("Tactical Signals"),
          DTOutput("sector_signal_table")
        ))
      ),
      br(),

      # Portfolio + Equity curve + Drawdown
      fluidRow(
        column(4, card(
          card_header("Portfolio Allocation"),
          plotlyOutput("portfolio_donut", height = "340px")
        )),
        column(8,
          card(card_header("Backtest Equity Curve"),
               plotlyOutput("equity_curve", height = "320px")),
          card(card_header("Drawdown"),
               plotlyOutput("drawdown_chart", height = "210px"))
        )
      )
    )
  ),

  # ── Tab 2: Industry Rotation ──────────────────────────────────────────────
  nav_panel(
    "Industry Rotation",
    div(
      style = sprintf("background:%s;min-height:100vh;padding:16px 24px;", BG),
      fluidRow(
        column(12, card(
          card_header("Relative Rotation Graph — Industries vs SPY"),
          plotlyOutput("industry_rrg_plot", height = "560px")
        ))
      ),
      br(),
      fluidRow(
        column(12, card(
          card_header("Industry Signals"),
          div(
            style = "margin-bottom:10px;",
            selectInput("ind_sector_filter", "Filter by Sector:",
                        choices = c("All", names(INDUSTRY_UNIVERSE)),
                        selected = "All", width = "260px")
          ),
          DTOutput("industry_signal_table")
        ))
      )
    )
  ),

  # ── Tab 3: Economic Cycle ─────────────────────────────────────────────────
  nav_panel(
    "Economic Cycle",
    div(
      style = sprintf("background:%s;min-height:100vh;padding:16px 24px;", BG),
      fluidRow(
        column(4,
          card(
            card_header("Current Phase"),
            uiOutput("regime_phase_indicator"),
            hr(),
            uiOutput("regime_leaders")
          )
        ),
        column(8,
          card(card_header("Phase Scores"),
               plotlyOutput("regime_scores_plot", height = "280px")),
          br(),
          card(card_header("Indicator Trends"),
               plotlyOutput("regime_indicators_plot", height = "340px"))
        )
      )
    )
  ),

  # ── Tab 4: Momentum Heatmap ───────────────────────────────────────────────
  nav_panel(
    "Momentum Heatmap",
    div(
      style = sprintf("background:%s;min-height:100vh;padding:16px 24px;", BG),
      p(style=sprintf("color:%s;font-size:12px;", MUTED),
        "Cross-sectional z-score momentum relative to universe peers. ",
        "Green = above average, red = below average."),
      fluidRow(
        column(6, card(
          card_header("Sector Heatmap"),
          plotlyOutput("sector_heatmap", height = "auto")
        )),
        column(6, card(
          card_header("Industry Heatmap"),
          div(
            style = "margin-bottom:10px;",
            selectInput("heatmap_sector_filter", "Filter by Sector:",
                        choices = c("All", names(INDUSTRY_UNIVERSE)),
                        selected = "All", width = "260px")
          ),
          plotlyOutput("industry_heatmap", height = "auto")
        ))
      )
    )
  ),

  # ── Settings nav item ────────────────────────────────────────────────────
  nav_panel(
    icon("gear"),
    value = "settings",
    div(
      style = sprintf("background:%s;min-height:100vh;padding:16px 24px;", BG),
      card(
        card_header("Analysis Settings"),
        fluidRow(
          column(3,
            numericInput("years",       "History (years)",   value=7,  min=2, max=15, step=1),
            numericInput("rs_lookback", "RS-Ratio Lookback (wks)", value=5, min=3, max=10, step=1)
          ),
          column(3,
            numericInput("mom_lookback","RS-Mom Lookback (wks)", value=2, min=1, max=5, step=1),
            numericInput("sensitivity", "EMA Sensitivity",       value=6, min=3, max=12, step=1)
          ),
          column(3,
            numericInput("top_n",       "Max Overweight Sectors", value=3, min=1, max=6, step=1),
            numericInput("max_weight",  "Max Sector Weight",      value=0.25, min=0.1, max=0.5, step=0.05)
          ),
          column(3,
            numericInput("vol_target",  "Vol Target (ann.)",      value=0.12, min=0.05, max=0.30, step=0.01),
            checkboxInput("offline",    "Use Synthetic Data",     value=FALSE)
          )
        ),
        br(),
        actionButton("refresh_btn", "Refresh Analysis",
                     class="btn-primary btn-lg", icon=icon("rotate-right")),
        br(), br(),
        uiOutput("last_run_info")
      )
    )
  )
)

# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # Compute analysis on load and whenever Refresh is clicked
  analysis <- eventReactive(
    ignoreNULL = FALSE,
    eventExpr  = input$refresh_btn,
    valueExpr  = {
      withProgress(message = "Running analysis...", value = 0, {
        setProgress(0.1, detail = "Fetching prices")
        res <- run_analysis(
          offline       = isTRUE(input$offline),
          years         = input$years         %||% 7L,
          rs_lookback   = input$rs_lookback   %||% 5L,
          mom_lookback  = input$mom_lookback  %||% 2L,
          sensitivity   = input$sensitivity   %||% 6L,
          top_n         = input$top_n         %||% 3L,
          max_weight    = input$max_weight    %||% 0.25,
          vol_target    = input$vol_target    %||% 0.12
        )
        setProgress(1, detail = "Done")
        res
      })
    }
  )

  # ── Sector Overview outputs ───────────────────────────────────────────────
  output$metrics_bar <- renderUI({
    res <- analysis()
    m   <- res$metrics
    badge_col <- if (res$is_offline) "#f97316" else "#22c55e"
    badge_txt <- if (res$is_offline) "SYNTHETIC DATA" else "LIVE DATA"

    div(
      style = "margin-bottom:4px;",
      tags$h2(
        style = sprintf("color:%s;font-weight:800;", TEXT),
        "Sector Rotation Dashboard  ",
        tags$span(
          style = sprintf("background:%s;color:white;padding:3px 11px;border-radius:10px;font-size:12px;font-weight:700;vertical-align:middle;", badge_col),
          badge_txt
        )
      ),
      p(style = sprintf("color:%s;font-size:12px;margin:0;", MUTED),
        sprintf("RRG Tactical Allocation  |  As of %s", format(res$latest_date, "%Y-%m-%d")))
    )
    div(
      style = "display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;",
      metric_box("CAGR",     m[["CAGR (Strategy)"]] %||% "N/A", ACCENT),
      metric_box("Sharpe",   m[["Sharpe Ratio"]]    %||% "N/A", ACCENT),
      metric_box("Max DD",   m[["Max Drawdown"]]    %||% "N/A", "#ef4444"),
      metric_box("Sortino",  m[["Sortino Ratio"]]   %||% "N/A", ACCENT),
      metric_box("Win Rate", m[["Win Rate"]]         %||% "N/A", "#22c55e"),
      metric_box("Ann. Vol", m[["Ann. Volatility"]] %||% "N/A", "#f97316"),
      metric_box("SPY CAGR", m[["CAGR (SPY)"]]      %||% "N/A", "#64748b")
    )
  })

  output$sector_rrg_plot <- renderPlotly({
    res <- analysis()
    build_rrg_figure(res$sector_rrg, res$sector_signals,
                     as.list(SECTOR_COLORS), as.list(SECTOR_NAMES))
  })

  output$sector_signal_table <- renderDT({
    build_signal_table_ui(analysis()$sector_signals, name_col = "sector_name")
  })

  output$portfolio_donut <- renderPlotly({
    build_portfolio_donut(analysis()$portfolio)
  })

  output$equity_curve <- renderPlotly({
    build_equity_figure(analysis()$equity)
  })

  output$drawdown_chart <- renderPlotly({
    build_drawdown_figure(analysis()$equity)
  })

  # ── Industry Rotation outputs ─────────────────────────────────────────────
  output$industry_rrg_plot <- renderPlotly({
    res <- analysis()

    # Build colour + name maps for industries
    ind_colors <- as.list(
      setNames(
        map_chr(INDUSTRY_TICKERS, ~ SECTOR_BASE_COLORS[[INDUSTRY_TO_SECTOR[[.x]]]] %||% MUTED),
        INDUSTRY_TICKERS
      )
    )
    ind_names <- as.list(INDUSTRY_NAMES)

    build_rrg_figure(res$industry_rrg, res$industry_signals,
                     ind_colors, ind_names,
                     tail_length = 6L, title = "Industry Relative Rotation Graph")
  })

  output$industry_signal_table <- renderDT({
    res  <- analysis()
    sigs <- res$industry_signals
    filt <- input$ind_sector_filter %||% "All"
    if (!is.null(filt) && filt != "All")
      sigs <- sigs %>% filter(sector == filt)

    sigs <- sigs %>% mutate(sector_name = paste0(sector, " › ", ind_name))
    build_signal_table_ui(sigs, name_col = "sector_name")
  })

  # ── Economic Cycle outputs ────────────────────────────────────────────────
  output$regime_phase_indicator <- renderUI({
    reg   <- analysis()$regime
    col   <- PHASE_COLORS[[reg$phase]] %||% MUTED
    conf  <- round(reg$confidence * 100, 1)
    div(
      style = "text-align:center;padding:20px 10px;",
      div(style = sprintf(
            "font-size:42px;font-weight:900;color:%s;margin-bottom:6px;", col),
          reg$phase),
      div(style = sprintf("font-size:14px;color:%s;", MUTED),
          sprintf("Confidence: %.1f%%", conf)),
      br(),
      # Mini score pills
      div(style = "display:flex;gap:8px;justify-content:center;flex-wrap:wrap;",
          map(names(reg$scores), function(ph) {
            c_ <- PHASE_COLORS[[ph]] %||% MUTED
            div(style = sprintf(
              "background:%s22;border:1px solid %s;color:%s;border-radius:20px;padding:4px 12px;font-size:12px;",
              c_, c_, c_),
              sprintf("%s %.0f%%", ph, reg$scores[[ph]]*100))
          }))
    )
  })

  output$regime_leaders <- renderUI({
    reg <- analysis()$regime
    leaders <- reg$sector_leaders
    div(
      p(style=sprintf("color:%s;font-size:11px;text-transform:uppercase;letter-spacing:.5px;", MUTED),
        "Historically Leading Sectors"),
      div(style="display:flex;gap:6px;flex-wrap:wrap;",
          map(leaders, function(t) {
            col  <- SECTOR_COLORS[[t]] %||% MUTED
            name <- SECTOR_NAMES[[t]] %||% t
            div(style=sprintf(
              "background:%s22;border:1px solid %s;color:%s;border-radius:6px;padding:4px 10px;font-size:12px;font-weight:600;",
              col, col, col), name)
          }))
    )
  })

  output$regime_scores_plot <- renderPlotly({
    build_regime_figure(analysis()$regime)
  })

  output$regime_indicators_plot <- renderPlotly({
    build_indicator_figure(analysis()$regime)
  })

  # ── Heatmap outputs ───────────────────────────────────────────────────────
  output$sector_heatmap <- renderPlotly({
    hm <- analysis()$heatmap_df %>% filter(group == "Sector")
    build_heatmap_figure(hm, title = "Sector Momentum Z-Scores")
  })

  output$industry_heatmap <- renderPlotly({
    hm   <- analysis()$heatmap_df %>% filter(group != "Sector")
    filt <- input$heatmap_sector_filter %||% "All"
    if (!is.null(filt) && filt != "All") hm <- hm %>% filter(group == filt)
    build_heatmap_figure(hm, title = "Industry Momentum Z-Scores")
  })

  # ── Settings output ───────────────────────────────────────────────────────
  output$last_run_info <- renderUI({
    res <- analysis()
    p(style=sprintf("color:%s;font-size:12px;", MUTED),
      sprintf("Last analysis: %s  |  Data as of: %s  |  Mode: %s",
              format(Sys.time(), "%H:%M:%S"),
              format(res$latest_date, "%Y-%m-%d"),
              if (res$is_offline) "Synthetic" else "Live"))
  })
}

# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
shinyApp(ui = ui, server = server)
