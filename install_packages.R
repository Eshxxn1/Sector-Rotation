#!/usr/bin/env Rscript
# =============================================================================
# install_packages.R — Install all required R packages
# =============================================================================
# Run once before launching the dashboard:
#   Rscript install_packages.R
# =============================================================================

pkgs <- c(
  # Core data manipulation & time series
  "tidyverse",     # dplyr, ggplot2, tidyr, purrr, etc.
  "lubridate",     # date/time helpers
  "zoo",           # irregular time series (na.locf)

  # Financial data & analysis
  "tidyquant",     # Yahoo Finance via tq_get()
  "TTR",           # technical indicators (EMA, SMA, RSI …)
  "PerformanceAnalytics",  # portfolio metrics

  # Interactive dashboard
  "shiny",         # web application framework
  "bslib",         # Bootstrap themes + modern UI components
  "plotly",        # interactive Plotly charts
  "DT",            # interactive DataTables
  "htmltools"      # HTML helpers
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s ...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(sprintf("%-25s already installed", pkg))
  }
}

invisible(lapply(pkgs, install_if_missing))
message("\nAll packages ready. Launch the dashboard with:")
message('  shiny::runApp("app.R", port = 8050)')
message('or:')
message('  Rscript -e "shiny::runApp(\'app.R\', port=8050)"')
