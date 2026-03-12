#!/usr/bin/env Rscript
# =============================================================================
# Sector & Industry Rotation Analysis  —  Relative Rotation Graph (RRG) Engine
# =============================================================================
# Multi-factor RRG model for US equity sectors and sub-industries.
# Provides: data fetching, RRG computation, factor scoring, composite signals,
#           portfolio construction, walk-forward backtest, economic-regime
#           detection, and momentum heatmap.
#
# Usage (standalone):
#   source("sector_rotation.R")
#   result <- run_analysis(offline = FALSE)
#   print_signals(result$signals, result$portfolio)
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidyquant)
  library(TTR)
  library(PerformanceAnalytics)
  library(zoo)
  library(lubridate)
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_TICKERS <- c("XLK","XLF","XLV","XLY","XLP","XLI","XLE","XLU","XLRE","XLB","XLC")
BENCHMARK      <- "SPY"
CASH_PROXY     <- "SHY"

SECTOR_NAMES <- c(
  XLK="Technology", XLF="Financials",    XLV="Health Care",
  XLY="Cons. Discr.", XLP="Cons. Staples", XLI="Industrials",
  XLE="Energy",     XLU="Utilities",     XLRE="Real Estate",
  XLB="Materials",  XLC="Communication"
)

SECTOR_COLORS <- c(
  XLK="#0071C5", XLF="#4CAF50", XLV="#E91E63",
  XLY="#FF9800", XLP="#9C27B0", XLI="#607D8B",
  XLE="#795548", XLU="#00BCD4", XLRE="#F44336",
  XLB="#8BC34A", XLC="#3F51B5"
)

INDUSTRY_UNIVERSE <- list(
  "Technology"             = c(SMH="Semiconductors",  IGV="Software",
                                HACK="Cybersecurity",  CLOU="Cloud Computing"),
  "Financials"             = c(KBE="Banks",            KIE="Insurance",
                                IAI="Broker-Dealers"),
  "Health Care"            = c(IBB="Biotechnology",    IHI="Medical Devices",
                                XPH="Pharmaceuticals", IHF="Healthcare Providers"),
  "Energy"                 = c(XOP="Oil & Gas E&P",    OIH="Oil Services",
                                AMLP="Pipelines/MLPs"),
  "Consumer Discretionary" = c(XRT="Retail",           ITB="Homebuilders",
                                JETS="Airlines"),
  "Consumer Staples"       = c(PBJ="Food & Beverage"),
  "Industrials"            = c(ITA="Aerospace/Defense", XTN="Transportation",
                                PAVE="Infrastructure"),
  "Materials"              = c(GDX="Gold Miners",       COPX="Copper Miners",
                                MOO="Agribusiness"),
  "Real Estate"            = c(REZ="Residential RE"),
  "Communication"          = c(SOCL="Social Media",     PNQI="Internet"),
  "Utilities"              = c(IDU="Diversified Utils",  TAN="Solar Energy")
)

INDUSTRY_TICKERS <- unname(unlist(lapply(INDUSTRY_UNIVERSE, names)))

INDUSTRY_NAMES <- setNames(
  unlist(INDUSTRY_UNIVERSE, use.names = FALSE),
  unlist(lapply(INDUSTRY_UNIVERSE, names))
)

INDUSTRY_TO_SECTOR <- unlist(lapply(names(INDUSTRY_UNIVERSE), function(sec) {
  setNames(rep(sec, length(INDUSTRY_UNIVERSE[[sec]])),
           names(INDUSTRY_UNIVERSE[[sec]]))
}))

REGIME_PROXY_TICKERS <- c("QQQ","IWM","HYG","IWD","TLT")

CYCLE_SECTOR_MAP <- list(
  Recovery    = c("XLF","XLY","XLI","XLB"),
  Expansion   = c("XLK","XLC","XLY","XLI"),
  Slowdown    = c("XLE","XLB","XLP","XLV"),
  Contraction = c("XLU","XLP","XLV","XLRE")
)

PHASE_COLORS <- c(
  Recovery="#3b82f6", Expansion="#22c55e",
  Slowdown="#f97316", Contraction="#ef4444"
)

QUADRANT_COLORS <- c(
  Leading="#22c55e", Weakening="#f97316",
  Lagging="#ef4444", Improving="#3b82f6"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

#' Convert daily price tibble to weekly (last observation per week, ending Fri)
to_weekly <- function(df) {
  df %>%
    arrange(date) %>%
    mutate(week_fri = date + as.integer((5L - wday(date, week_start = 1L)) %% 7L)) %>%
    group_by(week_fri) %>%
    slice_tail(n = 1L) %>%
    ungroup() %>%
    mutate(date = week_fri) %>%
    select(-week_fri) %>%
    arrange(date)
}

#' Generate synthetic daily price data for demonstration
generate_synthetic <- function(tickers, years = 7, seed = 42L) {
  set.seed(seed)
  all_dates <- seq.Date(Sys.Date() - as.integer(years * 365 + 90),
                        Sys.Date(), by = "day")
  all_dates <- all_dates[!weekdays(all_dates) %in% c("Saturday", "Sunday")]
  n <- length(all_dates)

  param_tbl <- tribble(
    ~ticker,  ~drift, ~vol,
    "XLK",    0.14,   0.20,  "XLF",    0.09,   0.19,
    "XLV",    0.10,   0.15,  "XLY",    0.11,   0.21,
    "XLP",    0.07,   0.12,  "XLI",    0.10,   0.18,
    "XLE",    0.04,   0.28,  "XLU",    0.06,   0.14,
    "XLRE",   0.07,   0.18,  "XLB",    0.08,   0.20,
    "XLC",    0.11,   0.22,  "SPY",    0.10,   0.16,
    "SHY",    0.02,   0.02
  )

  mkt_noise <- rnorm(n)

  prices <- map(tickers, function(t) {
    p <- param_tbl %>% filter(ticker == t)
    d <- if (nrow(p) > 0) p$drift[1] else 0.09
    v <- if (nrow(p) > 0) p$vol[1]   else 0.20
    idio <- rnorm(n)
    ret  <- d / 252 + (v / sqrt(252)) * (0.6 * mkt_noise + 0.4 * idio)
    100 * exp(cumsum(ret))
  }) %>% setNames(tickers)

  as_tibble(prices) %>% mutate(date = all_dates, .before = 1)
}

#' Fetch adjusted close prices for a set of tickers.
#' Returns list(weekly, daily) as wide tibbles with a `date` column.
fetch_prices <- function(tickers,
                         benchmark   = BENCHMARK,
                         cash        = CASH_PROXY,
                         extra       = character(0),
                         years       = 7,
                         offline     = FALSE) {
  all_tickers <- unique(c(tickers, benchmark, cash, extra))
  close <- NULL

  if (!offline) {
    tryCatch({
      message(sprintf("[INFO] Fetching %d tickers from Yahoo Finance ...",
                      length(all_tickers)))
      raw <- tq_get(all_tickers,
                    from = Sys.Date() - as.integer(years * 365 + 90),
                    to   = Sys.Date(),
                    get  = "stock.prices") %>%
        select(symbol, date, adjusted)

      if (nrow(raw) > 0) {
        close <- raw %>%
          pivot_wider(names_from = symbol, values_from = adjusted) %>%
          arrange(date)
      }
    }, error = function(e) {
      message(sprintf("[WARN] Download failed: %s", e$message))
    })
  }

  if (is.null(close) || nrow(close) == 0) {
    message("[INFO] Using synthetic data (Yahoo Finance unavailable or --offline).")
    close <- generate_synthetic(all_tickers, years)
  }

  # Forward-fill then backward-fill missing values
  close <- close %>%
    arrange(date) %>%
    mutate(across(-date, ~ zoo::na.locf(., na.rm = FALSE))) %>%
    mutate(across(-date, ~ zoo::na.locf(., fromLast = TRUE, na.rm = FALSE)))

  # Drop columns with > 20 % missing (before fill)
  n_rows <- nrow(close)
  keep <- names(close)[names(close) == "date" |
    map_lgl(names(close)[names(close) != "date"], function(col) {
      sum(!is.na(close[[col]])) >= 0.8 * n_rows
    })]
  close <- close %>% select(all_of(keep))

  daily  <- close
  weekly <- to_weekly(close)

  message(sprintf("[INFO] Weekly: %d rows | %s → %s",
                  nrow(weekly), min(weekly$date), max(weekly$date)))
  list(weekly = weekly, daily = daily)
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CORE RRG COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

#' Assign quadrant label from rs_ratio and rs_momentum scalars
classify_quadrant <- function(rs_ratio, rs_momentum) {
  case_when(
    rs_ratio >= 100 & rs_momentum >= 100 ~ "Leading",
    rs_ratio >= 100 & rs_momentum <  100 ~ "Weakening",
    rs_ratio <  100 & rs_momentum <  100 ~ "Lagging",
    TRUE                                  ~ "Improving"
  )
}

#' Compute RS-Ratio and RS-Momentum for a vector of tickers vs a benchmark.
#'
#' @param weekly  Wide tibble with `date` column and price columns.
#' @param tickers Character vector of sector/industry tickers.
#' @param benchmark Benchmark ticker (must be in `weekly`).
#' @param rs_lookback  Weeks for RS-Ratio (default 5).
#' @param mom_lookback Weeks for RS-Momentum tail (default 2).
#' @param sensitivity  EMA span for smoothing RS-Momentum (default 6).
#' @return Long tibble: date, ticker, rs_ratio, rs_momentum, quadrant.
compute_rrg <- function(weekly,
                        tickers,
                        benchmark    = BENCHMARK,
                        rs_lookback  = 5L,
                        mom_lookback = 2L,
                        sensitivity  = 6L) {
  avail <- tickers[tickers %in% names(weekly)]
  if (!benchmark %in% names(weekly)) {
    warning("Benchmark not in weekly data"); return(tibble())
  }

  bm <- weekly[[benchmark]]

  map_dfr(avail, function(ticker) {
    px       <- weekly[[ticker]]
    rs_line  <- px / bm
    rs_ratio <- (rs_line / lag(rs_line, rs_lookback)) * 100
    rs_roc   <- (rs_ratio / lag(rs_ratio, mom_lookback) - 1) * 100
    rs_mom   <- TTR::EMA(rs_roc, n = sensitivity) + 100

    tibble(
      date        = weekly$date,
      ticker      = ticker,
      rs_ratio    = as.numeric(rs_ratio),
      rs_momentum = as.numeric(rs_mom)
    ) %>%
      filter(!is.na(rs_ratio), !is.na(rs_momentum)) %>%
      mutate(quadrant = classify_quadrant(rs_ratio, rs_momentum))
  })
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FACTOR SCORING
# ─────────────────────────────────────────────────────────────────────────────

#' Cross-sectional z-score a numeric vector.
zscore <- function(x) {
  mu <- mean(x, na.rm = TRUE); sd <- sd(x, na.rm = TRUE)
  if (is.na(sd) || sd == 0) return(rep(0, length(x)))
  (x - mu) / sd
}

#' Compute price-based factor z-scores (momentum, low-vol) for each ticker.
#' Suitable for both live signals and walk-forward backtest slices.
compute_factors <- function(weekly, daily, tickers) {
  avail <- tickers[tickers %in% names(weekly) & tickers %in% names(daily)]

  factor_df <- map_dfr(avail, function(t) {
    w <- weekly[[t]]
    d <- diff(log(daily[[t]]))
    d <- d[!is.na(d)]

    mom_1m <- if (length(w) >= 4)  (tail(w,1) / w[length(w)-3]  - 1) else 0
    mom_3m <- if (length(w) >= 13) (tail(w,1) / w[length(w)-12] - 1) else 0
    vol_20 <- if (length(d) >= 20) sd(tail(d,20)) * sqrt(252) else 0.20

    tibble(ticker = t,
           momentum = 0.5 * mom_1m + 0.5 * mom_3m,
           low_vol  = -vol_20)          # negative vol → low-vol tilt
  })

  factor_df %>%
    mutate(momentum_z  = zscore(momentum),
           low_vol_z   = zscore(low_vol),
           factor_z    = (momentum_z + low_vol_z) / 2,
           top2_factor = (momentum_z + low_vol_z) / 2)
}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPOSITE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

#' Blend RRG z-scores with factor z-scores into a single composite signal.
composite_signal <- function(rrg_latest, factors,
                             rrg_weight = 0.70, factor_weight = 0.30) {
  merged <- rrg_latest %>%
    inner_join(factors %>% select(ticker, factor_z, top2_factor),
               by = "ticker") %>%
    mutate(
      rrg_raw  = (rs_ratio + rs_momentum) / 2,
      rrg_z    = zscore(rrg_raw),
      composite = rrg_weight * rrg_z + factor_weight * top2_factor,
      composite_z    = zscore(composite),
      composite_norm = {
        mn <- min(composite_z, na.rm=TRUE)
        mx <- max(composite_z, na.rm=TRUE)
        if (mx > mn) (composite_z - mn) / (mx - mn) else rep(0.5, n())
      },
      rs_mom_roc = rs_momentum - 100
    ) %>%
    mutate(recommendation = case_when(
      quadrant %in% c("Leading","Improving") & composite_norm > 0.70 & rs_mom_roc > 0 ~ "Overweight",
      quadrant %in% c("Leading","Improving")                                           ~ "Hold",
      quadrant == "Weakening"                                                          ~ "Hold",
      TRUE                                                                              ~ "Underweight"
    ))
  merged
}

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

#' Iteratively cap weights at `cap` and redistribute excess.
cap_weights <- function(w, cap = 0.25) {
  for (i in seq_len(10)) {
    excess  <- sum(pmax(w - cap, 0))
    w       <- pmin(w, cap)
    uncapped <- sum(w < cap)
    if (excess <= 0 || uncapped == 0) break
    w[w < cap] <- w[w < cap] + excess / uncapped
  }
  w
}

#' Build an inverse-vol weighted portfolio from composite signals.
#'
#' Risk rules:
#'   1. Pick top-N Overweight; fall back to Hold if none.
#'   2. Inverse-volatility weights, capped at max_weight.
#'   3. Vol-target scaling; remainder → cash.
#'   4. 100 % cash if all selected sectors have negative 3-month absolute momentum.
construct_portfolio <- function(signals, daily,
                                max_sectors = 3L,
                                max_weight  = 0.25,
                                vol_target  = 0.12) {
  ow <- signals %>% filter(recommendation == "Overweight")
  if (nrow(ow) == 0)
    ow <- signals %>% filter(recommendation == "Hold") %>%
      slice_max(composite_norm, n = max_sectors)
  ow <- ow %>% slice_max(composite_norm, n = max_sectors)

  if (nrow(ow) == 0)
    return(tibble(ticker = CASH_PROXY, weight = 1.0))

  # Absolute-momentum filter
  all_neg <- all(map_lgl(ow$ticker, function(t) {
    if (!t %in% names(daily)) return(TRUE)
    px <- daily[[t]]; px <- px[!is.na(px)]
    if (length(px) < 63) return(TRUE)
    (tail(px,1) / px[length(px)-62] - 1) <= 0
  }))
  if (all_neg) {
    message("[RISK] All sectors negative 3M momentum → 100% CASH")
    return(tibble(ticker = CASH_PROXY, weight = 1.0))
  }

  # Inverse-vol weights
  vols <- map_dbl(ow$ticker, function(t) {
    if (!t %in% names(daily)) return(0.15)
    d <- diff(log(daily[[t]])); d <- d[!is.na(d)]
    if (length(d) >= 20) sd(tail(d,20)) * sqrt(252) else 0.15
  }) %>% pmax(0.01)

  w <- (1 / vols) / sum(1 / vols)
  w <- cap_weights(w, max_weight)

  # Vol-target scaling
  port_vol <- sum(w * vols)
  if (port_vol > 0) w <- w * min(vol_target / port_vol, 1.0)

  result <- tibble(ticker = ow$ticker, weight = round(w, 4))
  cash_w <- 1 - sum(result$weight)
  if (cash_w > 0.001)
    result <- bind_rows(result, tibble(ticker = CASH_PROXY, weight = round(cash_w, 4)))

  result %>% arrange(desc(weight))
}

# ─────────────────────────────────────────────────────────────────────────────
# 7.  WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

#' Monthly walk-forward backtest with transaction costs.
#'
#' @param weekly  Wide weekly price tibble.
#' @param daily   Wide daily price tibble.
#' @param tickers Sector tickers to trade.
#' @param ...     RRG / portfolio params forwarded through.
#' @return list(equity = tibble, metrics = list)
run_backtest <- function(weekly, daily,
                         tickers      = SECTOR_TICKERS,
                         benchmark    = BENCHMARK,
                         cash         = CASH_PROXY,
                         rs_lookback  = 5L,
                         mom_lookback = 2L,
                         sensitivity  = 6L,
                         top_n        = 3L,
                         max_weight   = 0.25,
                         vol_target   = 0.12,
                         tc_bps       = 10,
                         slippage_bps = 5) {

  warmup <- max(rs_lookback + mom_lookback + sensitivity + 10L, 52L)
  if (nrow(weekly) < warmup + 12L) {
    warning("[BACKTEST] Insufficient data."); return(list(equity=tibble(), metrics=list()))
  }

  # Monthly rebalance: last week of each month
  monthly_dates <- weekly %>%
    filter(row_number() > warmup) %>%
    mutate(ym = format(date, "%Y-%m")) %>%
    group_by(ym) %>%
    slice_tail(n = 1) %>%
    ungroup() %>%
    pull(date)

  if (length(monthly_dates) < 2) {
    warning("[BACKTEST] Not enough monthly dates."); return(list(equity=tibble(), metrics=list()))
  }

  message(sprintf("[BACKTEST] %s → %s (%d periods)",
                  min(monthly_dates), max(monthly_dates), length(monthly_dates)-1))

  rrg_full <- compute_rrg(weekly, tickers, benchmark,
                          rs_lookback, mom_lookback, sensitivity)
  prev_w  <- setNames(numeric(length(tickers)), tickers)
  records <- vector("list", length(monthly_dates) - 1L)

  for (i in seq_len(length(monthly_dates) - 1L)) {
    reb   <- monthly_dates[i]
    nxt   <- monthly_dates[i + 1L]

    rrg_snap <- rrg_full %>% filter(date == reb)
    if (nrow(rrg_snap) == 0)
      rrg_snap <- rrg_full %>% filter(date <= reb) %>% filter(date == max(date))
    if (nrow(rrg_snap) == 0) next

    w_sub <- weekly %>% filter(date <= reb)
    d_sub <- daily  %>% filter(date <= reb)

    fac <- compute_factors(w_sub, d_sub, tickers)
    sig <- composite_signal(rrg_snap, fac)

    ow <- sig %>% filter(recommendation == "Overweight")
    if (nrow(ow) == 0)
      ow <- sig %>% filter(recommendation == "Hold") %>% slice_max(composite_norm, n=top_n)
    ow <- ow %>% slice_max(composite_norm, n=top_n)

    # Build weights
    if (nrow(ow) == 0) {
      weights <- setNames(c(1.0), cash)
    } else {
      all_neg <- all(map_lgl(ow$ticker, function(t) {
        if (!t %in% names(d_sub)) return(TRUE)
        px <- d_sub[[t]]; px <- px[!is.na(px)]
        length(px) < 63 || (tail(px,1)/px[length(px)-62] - 1) <= 0
      }))
      if (all_neg) {
        weights <- setNames(c(1.0), cash)
      } else {
        vols <- map_dbl(ow$ticker, function(t) {
          if (!t %in% names(d_sub)) return(0.15)
          d <- diff(log(d_sub[[t]])); d <- d[!is.na(d)]
          if (length(d) >= 20) sd(tail(d,20))*sqrt(252) else 0.15
        }) %>% pmax(0.01)
        w <- cap_weights((1/vols)/sum(1/vols), max_weight)
        pv <- sum(w * vols)
        if (pv > 0) w <- w * min(vol_target / pv, 1.0)
        cash_w <- max(0, 1 - sum(w))
        weights <- c(setNames(w, ow$ticker),
                     if (cash_w > 0.001) setNames(cash_w, cash) else NULL)
      }
    }

    # Period return
    get_px <- function(t, dt) {
      if (!t %in% names(weekly)) return(NA_real_)
      row <- weekly %>% filter(date <= dt) %>% slice_tail(n=1)
      if (nrow(row) == 0) NA_real_ else row[[t]]
    }
    port_ret <- sum(map_dbl(names(weights), function(t) {
      p0 <- get_px(t, reb); p1 <- get_px(t, nxt)
      if (any(is.na(c(p0,p1))) || p0 <= 0) 0 else weights[[t]] * (p1/p0 - 1)
    }))

    # Transaction costs
    all_tks <- union(names(weights), names(prev_w))
    turnover <- sum(abs(map_dbl(all_tks, ~ (weights[[.x]] %||% 0) - (prev_w[[.x]] %||% 0))))
    port_ret <- port_ret - turnover * (tc_bps + slippage_bps) / 1e4

    # Benchmark
    bm0 <- get_px(benchmark, reb); bm1 <- get_px(benchmark, nxt)
    bm_ret <- if (any(is.na(c(bm0,bm1))) || bm0<=0) 0 else bm1/bm0 - 1

    records[[i]] <- tibble(date = nxt, port_return = port_ret, bm_return = bm_ret)
    prev_w <- weights
  }

  equity <- bind_rows(records) %>%
    mutate(port_cumret = cumprod(1 + port_return),
           bm_cumret   = cumprod(1 + bm_return))

  list(equity = equity, metrics = compute_metrics(equity))
}

#' Compute standard portfolio performance metrics.
compute_metrics <- function(equity) {
  if (nrow(equity) == 0) return(list())
  pr   <- equity$port_return
  br   <- equity$bm_return
  n    <- length(pr)
  yrs  <- n / 12

  cum_p  <- tail(equity$port_cumret, 1)
  cum_b  <- tail(equity$bm_cumret,   1)
  cagr_p <- cum_p^(1/yrs) - 1
  cagr_b <- cum_b^(1/yrs) - 1

  excess <- pr - br
  sharpe <- if (sd(excess)>0) mean(excess)/sd(excess)*sqrt(12) else 0

  dd     <- (equity$port_cumret - cummax(equity$port_cumret)) / cummax(equity$port_cumret)
  max_dd <- min(dd)
  calmar <- if (max_dd != 0) cagr_p / abs(max_dd) else 0

  dn_std <- sd(pr[pr < 0]) * sqrt(12)
  sortino <- if (!is.na(dn_std) && dn_std > 0) mean(pr)*12/dn_std else 0

  list(
    `CAGR (Strategy)` = sprintf("%.2f%%", cagr_p * 100),
    `CAGR (SPY)`      = sprintf("%.2f%%", cagr_b * 100),
    `Sharpe Ratio`    = sprintf("%.2f",  sharpe),
    `Sortino Ratio`   = sprintf("%.2f",  sortino),
    `Max Drawdown`    = sprintf("%.2f%%", max_dd  * 100),
    `Calmar Ratio`    = sprintf("%.2f",  calmar),
    `Ann. Volatility` = sprintf("%.2f%%", sd(pr)*sqrt(12)*100),
    `Win Rate`        = sprintf("%.1f%%", mean(pr > 0)*100),
    `Total Return`    = sprintf("%.2f%%", (cum_p - 1)*100),
    `SPY Total Return`= sprintf("%.2f%%", (cum_b - 1)*100),
    Periods           = n
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# 8.  ECONOMIC REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

#' Classify the current economic cycle phase using ETF ratio trends.
#'
#' Proxies (all ETF-based, no external data required):
#'   risk_appetite    XLY / XLP  — cyclicals vs defensives
#'   rate_sensitivity XLU / XLK  — rate-sensitive vs growth (late-cycle signal)
#'   growth_vs_value  QQQ / IWD  — growth vs value (QQQ/IWD or XLK/XLV)
#'   credit           HYG / SHY  — high-yield vs short-treasury (risk-on proxy)
#'   breadth          IWM / SPY  — small-cap vs large-cap (broad participation)
#'   market_trend     SPY vs 26-week MA
#'
#' @return Named list: phase, confidence, scores, indicators, sector_leaders
detect_economic_regime <- function(weekly, lookback_weeks = 13L) {

  ratio_trend <- function(t1, t2) {
    if (!all(c(t1, t2) %in% names(weekly))) return(0)
    r <- weekly[[t1]] / weekly[[t2]]
    r <- r[!is.na(r)]
    if (length(r) < lookback_weeks + 2L) return(0)
    r <- tail(r, lookback_weeks)
    as.numeric(tail(r,1) / r[1] - 1)
  }

  above_ma <- function(ticker, weeks = 26L) {
    if (!ticker %in% names(weekly)) return(0)
    px <- weekly[[ticker]]; px <- px[!is.na(px)]
    if (length(px) < weeks) return(0)
    diff_val <- tail(px,1) - mean(tail(px, weeks))
    as.numeric(sign(diff_val) * min(abs(diff_val / tail(px,1)), 1))
  }

  gv_proxy <- if (all(c("QQQ","IWD") %in% names(weekly))) c("QQQ","IWD") else c("XLK","XLV")
  cr_proxy <- if (all(c("HYG","SHY") %in% names(weekly))) c("HYG","SHY") else c("XLF","XLU")
  br_proxy <- if ("IWM" %in% names(weekly))               c("IWM","SPY") else c("XLI","XLK")

  ind <- list(
    risk_appetite    = ratio_trend("XLY","XLP"),
    rate_sensitivity = ratio_trend("XLU","XLK"),
    growth_vs_value  = ratio_trend(gv_proxy[1], gv_proxy[2]),
    credit           = ratio_trend(cr_proxy[1], cr_proxy[2]),
    breadth          = ratio_trend(br_proxy[1], br_proxy[2]),
    market_trend     = above_ma("SPY", 26L),
    momentum_factor  = ratio_trend("XLK","XLP"),
    defensives_trend = ratio_trend("XLV","SPY")
  )

  ra <- ind$risk_appetite; rs <- ind$rate_sensitivity
  gv <- ind$growth_vs_value; cr <- ind$credit
  br <- ind$breadth;          mt <- ind$market_trend
  mo <- ind$momentum_factor;  df <- ind$defensives_trend

  scores <- c(
    Recovery    = (max(ra,0)*0.25 + max(cr,0)*0.25 +
                     ifelse(mt>0,0.5,0)*0.25 + max(-gv,0)*0.25),
    Expansion   = (max(ra,0)*0.25 + max(gv,0)*0.25 +
                     max(br,0)*0.25 + max(mo,0)*0.25),
    Slowdown    = (max(rs,0)*0.25 + max(-ra,0)*0.20 +
                     max(-gv,0)*0.25 + max(df,0)*0.30),
    Contraction = (max(-ra,0)*0.30 + max(-cr,0)*0.25 +
                     ifelse(mt<0,0.5,0)*0.25 + max(-br,0)*0.20)
  )

  total <- sum(scores); if (total > 0) scores <- scores / total
  phase <- names(which.max(scores))

  list(
    phase          = phase,
    confidence     = round(scores[[phase]], 4),
    scores         = round(scores, 4),
    indicators     = ind,
    sector_leaders = CYCLE_SECTOR_MAP[[phase]]
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# 9.  MOMENTUM HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

#' Build cross-sectional z-score momentum table.
#'
#' @param weekly   Wide weekly tibble.
#' @param tickers  Tickers to include (sectors and/or industries).
#' @param lookbacks Integer vector of lookback windows in weeks.
#' @return Tibble: ticker, name, group, and one column per lookback (z-scored).
build_momentum_heatmap <- function(weekly,
                                   tickers  = SECTOR_TICKERS,
                                   lookbacks = c(4L, 8L, 13L, 26L, 52L)) {
  avail <- tickers[tickers %in% names(weekly)]
  all_names <- c(SECTOR_NAMES, INDUSTRY_NAMES)

  raw <- map_dfr(avail, function(t) {
    px <- weekly[[t]]; px <- px[!is.na(px)]
    rets <- map_dbl(lookbacks, function(lb) {
      if (length(px) > lb) as.numeric(tail(px,1) / px[length(px)-lb] - 1) else NA_real_
    })
    tibble(ticker = t, !!!setNames(rets, paste0("W", lookbacks)))
  })

  # Cross-sectional z-score each lookback column
  for (col in paste0("W", lookbacks)) {
    raw[[col]] <- zscore(raw[[col]])
  }

  raw %>%
    mutate(
      name  = all_names[ticker],
      group = case_when(
        ticker %in% SECTOR_TICKERS   ~ "Sector",
        ticker %in% INDUSTRY_TICKERS ~ INDUSTRY_TO_SECTOR[ticker],
        TRUE                          ~ "Other"
      )
    ) %>%
    select(ticker, name, group, everything())
}

# ─────────────────────────────────────────────────────────────────────────────
# 10.  HIGH-LEVEL ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

#' Run the full sector + industry rotation analysis pipeline.
#'
#' @param offline    Logical — skip Yahoo Finance and use synthetic data.
#' @param years      Years of history to fetch.
#' @param rs_lookback  RRG RS-Ratio lookback (weeks).
#' @param mom_lookback RRG RS-Momentum tail (weeks).
#' @param sensitivity  EMA smoothing span.
#' @param top_n        Max sectors to overweight.
#' @param max_weight   Per-sector weight cap.
#' @param vol_target   Annualised portfolio volatility target.
#' @param rrg_weight   Weight of RRG component in composite.
#' @param factor_weight Weight of factor component in composite.
#' @return Named list with all computed artefacts.
run_analysis <- function(offline      = FALSE,
                         years        = 7L,
                         rs_lookback  = 5L,
                         mom_lookback = 2L,
                         sensitivity  = 6L,
                         top_n        = 3L,
                         max_weight   = 0.25,
                         vol_target   = 0.12,
                         rrg_weight   = 0.70,
                         factor_weight = 0.30) {

  # --- Fetch all data in one call ---
  message("[INFO] Fetching data ...")
  all_extra <- unique(c(INDUSTRY_TICKERS, REGIME_PROXY_TICKERS))
  data <- fetch_prices(SECTOR_TICKERS, extra = all_extra,
                       years = years, offline = offline)
  weekly <- data$weekly; daily <- data$daily

  # --- Sector RRG & signals ---
  message("[INFO] Computing sector RRG ...")
  sector_rrg <- compute_rrg(weekly, SECTOR_TICKERS, BENCHMARK,
                             rs_lookback, mom_lookback, sensitivity)
  latest_date   <- max(sector_rrg$date)
  sector_latest <- sector_rrg %>% filter(date == latest_date)

  sector_factors  <- compute_factors(weekly, daily, SECTOR_TICKERS)
  sector_signals  <- composite_signal(sector_latest, sector_factors,
                                      rrg_weight, factor_weight) %>%
    mutate(sector_name = SECTOR_NAMES[ticker])
  portfolio       <- construct_portfolio(sector_signals, daily,
                                         top_n, max_weight, vol_target)

  # --- Industry RRG & signals ---
  message("[INFO] Computing industry RRG ...")
  ind_tickers <- INDUSTRY_TICKERS[INDUSTRY_TICKERS %in% names(weekly)]
  industry_rrg    <- compute_rrg(weekly, ind_tickers, BENCHMARK,
                                  rs_lookback, mom_lookback, sensitivity)
  ind_latest_date <- if (nrow(industry_rrg) > 0) max(industry_rrg$date) else latest_date
  industry_latest <- industry_rrg %>% filter(date == ind_latest_date)

  industry_factors <- compute_factors(weekly, daily, ind_tickers)
  industry_signals <- composite_signal(industry_latest, industry_factors,
                                        rrg_weight, factor_weight) %>%
    mutate(sector   = INDUSTRY_TO_SECTOR[ticker],
           ind_name = INDUSTRY_NAMES[ticker])

  # --- Economic regime ---
  message("[INFO] Detecting economic regime ...")
  regime <- detect_economic_regime(weekly)

  # --- Momentum heatmap ---
  message("[INFO] Building heatmap ...")
  heatmap_df <- build_momentum_heatmap(weekly,
                                        c(SECTOR_TICKERS, ind_tickers))

  # --- Backtest ---
  message("[INFO] Running backtest ...")
  bt <- run_backtest(weekly, daily, SECTOR_TICKERS, BENCHMARK, CASH_PROXY,
                     rs_lookback, mom_lookback, sensitivity,
                     top_n, max_weight, vol_target)

  list(
    weekly          = weekly,
    daily           = daily,
    sector_rrg      = sector_rrg,
    sector_signals  = sector_signals,
    industry_rrg    = industry_rrg,
    industry_signals = industry_signals,
    portfolio       = portfolio,
    regime          = regime,
    heatmap_df      = heatmap_df,
    equity          = bt$equity,
    metrics         = bt$metrics,
    latest_date     = latest_date,
    is_offline      = offline
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# 11.  CONSOLE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

print_signals <- function(signals, portfolio) {
  cat(strrep("=", 90), "\n")
  cat("  SECTOR ROTATION — TACTICAL ALLOCATION SIGNALS\n")
  cat(strrep("=", 90), "\n")
  cat(sprintf("%-16s %-7s %9s %9s %10s %-12s %-12s\n",
              "Sector","Ticker","RS-Ratio","RS-Mom","Composite","Quadrant","Signal"))
  cat(strrep("-", 90), "\n")

  for (i in seq_len(nrow(signals %>% arrange(desc(composite_norm))))) {
    row <- (signals %>% arrange(desc(composite_norm)))[i,]
    marker <- if (row$recommendation == "Overweight") ">>>" else "   "
    name   <- substr(SECTOR_NAMES[row$ticker] %||% row$ticker, 1, 14)
    cat(sprintf("%s %-13s %-7s %8.2f %8.2f %9.2f %-12s %-12s\n",
                marker, name, row$ticker,
                row$rs_ratio, row$rs_momentum,
                row$composite_norm, row$quadrant, row$recommendation))
  }
  cat(strrep("-", 90), "\n\n  PORTFOLIO WEIGHTS:\n", strrep("-", 40), "\n")
  for (i in seq_len(nrow(portfolio))) {
    t    <- portfolio$ticker[i]; w <- portfolio$weight[i]
    name <- substr(SECTOR_NAMES[t] %||% t, 1, 14)
    bar  <- strrep("#", as.integer(w * 40))
    cat(sprintf("  %-14s (%s) %6.1f%%  %s\n", name, t, w*100, bar))
  }
  cat(strrep("-", 40), "\n")
}

print_backtest <- function(metrics) {
  if (length(metrics) == 0) return(invisible(NULL))
  cat(strrep("=", 50), "\n  BACKTEST PERFORMANCE SUMMARY\n", strrep("=", 50), "\n")
  for (nm in names(metrics))
    cat(sprintf("  %-24s %12s\n", nm, as.character(metrics[[nm]])))
  cat(strrep("=", 50), "\n")
}

# Run when executed as a script (not when sourced from app.R)
if (!interactive() && !exists(".sourced_by_app")) {
  result <- run_analysis(offline = "--offline" %in% commandArgs(trailingOnly=TRUE))
  print_signals(result$sector_signals, result$portfolio)
  print_backtest(result$metrics)
  cat(sprintf("\n[REGIME] Current phase: %s (confidence %.1f%%)\n",
              result$regime$phase, result$regime$confidence * 100))
  cat("[REGIME] Leading sectors:", paste(result$regime$sector_leaders, collapse=", "), "\n")
}
