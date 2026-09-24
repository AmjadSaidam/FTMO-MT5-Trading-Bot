# FTMO MT5 Trading Bot

The repository contains an end-to-end trading strategy (portfolio) pipeline, from backtesting to deployment code, intended to be used to pass the one-step FTMO prop firm evaluation [1].

![Repo Header](/images/repo_header.png)

DISCLAIMER: THIS REPOSITORY AND ITS CONTENTS DO NOT CONSTITUTE FINANCIAL ADVICE. THE LIVE CODE IS PAPER-TRADING COMPATIBLE, AND PERSONAL PAPER TRADING AND TESTING IS HIGHLY RECOMMENDED BEFORE ANY CAPITAL OUTLAY. AS OF WRITING, I AM CURRENTLY TRADING THIS LIVE ON TWO ACCOUNTS.

## Contents

- [Repository Tree](#repository-tree)
- [Medium Blog](#medium-blog)
- [Setup](#setup)
  - [0. Pre-Setup](#0-pre-setup)
  - [1. VPS](#1-vps)
  - [2. Installing Python and Git](#2-installing-python-and-git)
  - [3. Cloning](#3-cloning)
  - [4. Installing Dependencies](#4-installing-dependencies)
  - [5. Environment Variables](#5-environment-variables)
  - [6. Task Scheduler](#6-task-scheduler)
- [Development](#development)
- [Appendix](#appendix)

## Repository Tree

```
Trading Strategy One/
├── app/
│   └── dashboard.py                    # Streamlit dashboard for monitoring account/strategy performance (run: streamlit run app/dashboard.py)
├── code/                                # Core backtesting, strategy, and analytics library
│   ├── analytics.py                     # Trade-outcome probability statistics (e.g. probability of a losing streak)
│   ├── backtest_engine.py               # Vectorised backtest engine and position-sizing logic
│   ├── challenge_validation.py          # FTMO prop-firm rule/constraint validation (PropFirmValidation)
│   ├── optimisation.py                  # Grid-search optimisation of strategy ATR/risk-reward parameters
│   ├── performance_metrics.py           # Strategy performance statistics (e.g. Sharpe ratio)
│   ├── permutation_testing.py           # Permutation-based statistical significance testing of strategy returns
│   ├── regime_filter.py                 # Market regime classification (trend/range detection via rolling R²)
│   ├── risk_management.py               # ATR, consecutive-loss guards, and the ML-based volatility/position risk model
│   ├── strategy_indicators.py           # Technical indicators (VWAP, Bollinger Bands, cumulative volume delta, z-score, etc.)
│   ├── trading_strategies.py            # The 3 signal-generating strategy functions, one per traded symbol
│   └── walk_forward_analysis.py         # Walk-forward analysis (in-sample/out-of-sample split) logic
├── data/                                 # Historical OHLCV data used for backtesting (git-ignored, populated locally by data_loaders/)
├── data_loaders/                         # Scripts for pulling historical data from external sources
│   ├── binance_data.py                  # Historical OHLCV downloader from Binance
│   ├── databento_data.py                # Historical equity data downloader from Databento
│   ├── kaggle_data.py                   # BTCUSDT historical dataset downloader from Kaggle
│   └── massive_data.py                  # Bulk historical data downloader
├── enums/
│   ├── challenge_tradeable_assets.py    # FTMO tradeable asset lists (per FTMO's published symbol list)
│   └── utility.py                       # Shared constants (e.g. supported timeframe interval strings)
├── images/                               # README screenshots (Task Scheduler setup, repo header)
├── live_trading/                         # Live execution engine deployed on the VPS
│   ├── live_loop.py                     # Main polling loop: pulls MT5 data, evaluates strategies, sends/manages orders
│   ├── logging_config.py                # Logging setup (rotating text log + JSONL trade-event log)
│   ├── mt5_connector.py                 # Thin wrapper around the MetaTrader5 Python API (connect, fetch data, send/close orders)
│   └── mt5_errors.py                    # Custom exception types for MT5 connection/order/data errors
├── notebooks/
│   ├── backtest.ipynb                   # Backtesting and strategy analysis notebook
│   └── results/                         # Saved notebook outputs (currently empty)
├── paper/                                # Reserved for paper-trading records (currently empty)
├── pine/                                 # Reserved for TradingView Pine Script strategy ports (currently empty)
├── results/                              # Backtest result outputs (currently empty)
├── tests/                                # Pytest suite for the live trading module
│   ├── conftest.py                      # Shared pytest fixtures and sys.path setup
│   ├── test_live_loop.py                # Tests for live_loop.py polling/exception-handling logic
│   ├── test_logging_config.py           # Tests for logging_config.py
│   ├── test_mt5_connector.py            # Tests for mt5_connector.py
│   └── test_mt5_errors.py               # Tests for mt5_errors.py
├── .env                                  # Local environment variables (git-ignored, not committed)
├── .gitignore                            # Git-ignored paths (caches, logs, .env, etc.)
├── README.md                             # This file
├── requirements.txt                      # Python dependencies for the live trading deployment
└── run_strategy.bat                      # Windows batch script that launches the live loop under the venv's Python, restarting it on exit/crash
```

## Medium Blog

I am currently blogging my experience deploying this live strategy on Medium [2]. Topics covered (but not limited to) include strategy motivation, backtesting analysis, live performance evaluation, and FTMO evaluation progress.

## Setup

The recommended implementation involves hosting the live strategy code indefinitely on a Windows VPS. Importantly, the signal-to-order logic uses the MetaTrader5 Python package [3], which relies on an Interprocess Communication (IPC) link to the MetaTrader terminal. This package only ships wheels for Windows, so deployment is only possible on a Windows operating system. Alternative deployment methods include local hosting (on your own PC) or manual trading. The latter overcomes the OS restriction, although it is not recommended, as it requires you to input every signal manually, which is extremely difficult given trades can fire at any time. The setup below proceeds with a VPS deployment.

### 0. Pre-Setup

Create an FTMO account [4]. You can either start with the free demo paper trading account (recommended) or purchase the real evaluation challenge account. The setup process is identical either way, with the only differences being the MT5 deployment server and the demo account rules. *All commands below use PowerShell syntax.*

### 1. VPS

There are many VPS hosting solutions; some examples include Microsoft Azure [5] and Amazon Web Services (AWS) [6].

### 2. Installing Python and Git

The commands below install Python and Git globally (accessible from any directory).

```
winget install --id Python.Python.3.12 -e --source winget --scope machine
winget install --id Git.Git -e --source winget --scope machine
```

### 3. Cloning

To copy this repository's contents to your local machine:

```
$repo = "REPO_PATH" # alternatively, cd into the target directory first and just git clone
git clone "https://github.com/AmjadSaidam/FTMO-MT5-Trading-Bot.git" $repo
cd $repo
```

Note: without setting the `$repo` variable, the clone will create a `FTMO-MT5-Trading-Bot` folder in the current directory. To confirm the download was successful, run:

```
ls -la # output should contain all repo contents
```

### 4. Installing Dependencies

Install `uv` for fast Python package installs:

```
pip install uv
```

To avoid dependency conflicts, create a virtual environment pinned to the Python version used in development, for consistency:

```
uv venv --python 3.12
```

To activate the virtual environment, run `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (Command Prompt). If a script-loading error is raised, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` and re-attempt activation. To confirm the virtual environment is active, run `$env:VIRTUAL_ENV` — it should print the path to `.venv`.

We are now ready to install all required repo dependencies:

```
uv pip install -r requirements.txt
```

### 5. Environment Variables

These environment variables are user-specific and must be set for account authentication; only once they are set can a connection be established to poll data and send orders.

```
# Information can be sourced from your FTMO account dashboard
setx META_PATH "PATH_TO_MT5_TERMINAL" # file terminal64.exe
setx META_LOGIN "FTMO_ACCOUNT_LOGIN"
setx META_PASSWORD "FTMO_ACCOUNT_MASTER_PASSWORD"
setx META_SERVER "FTMO_ACCOUNT_SERVER"
setx META_INITIAL_ACCOUNT_BALANCE "FTMO_ACCOUNT_INITIAL_BALANCE"
```

These are backtest and model environment variables; they should match the values used during the Walk Forward Analysis (WFA) backtest. *Only change these values if you understand each variable's purpose and effect. It is highly recommended to re-run the WFA and only update these values if doing so improves performance.*

```
setx META_WFA_IN_SAMPLE_DAYS "20"
setx META_ML_FEATURE_WINDOW "30" # in bars (default 5min)
setx META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP "2"
setx META_TOTAL_RISK_ALLOCATION "0.01"
```

These are default FTMO account environment variables and should not be changed. They mirror the account constraints listed by FTMO [7].

```
setx META_MIN_VOLUME "0.01"
setx META_MAX_FOREX_LEVERAGE "100.0"
setx META_MAX_OTHER_LEVERAGE "3.33"
```

To re-assign a value, simply re-run the command with the new value. `setx` only affects new sessions, so to validate that an environment variable has been created, close and reopen PowerShell in the project directory and run `$env:<VARIABLE_NAME>` (e.g. `$env:META_LOGIN`), which should print the assigned value.

### 6. Task Scheduler

This step is the most critical, and ensures our script can run 24/7, restarting automatically after a computer reboot, crash, or unhandled error. First, open Task Scheduler and click "Task Scheduler Library" → "Create Task". Configure as follows:

**General**

![General](/images/task_scheduler_general.png)

**Triggers**

Select "New" → "At Startup".

**Action**

Select "New", and under "Program/script" click "Browse" and select the `run_strategy.bat` file.

**Conditions**

![Conditions](/images/task_scheduler_conditions.png)

**Settings**

![Settings](/images/task_scheduler_settings.png)

## Development

- Re-run backtests on FTMO's own data using the MetaTrader5 package [NOT IMPLEMENTED]

## Appendix

[1] - [FTMO One Step Challenge](https://trader.ftmo.com/start-challenge)

[2] - [Medium Blog](https://medium.com/@amjadsaidama)

[3] - [MetaTrader Python Package Docs](https://www.mql5.com/en/docs/python_metatrader5)

[4] - [FTMO Sign-Up](https://sso.ftmo.com/auth/realms/FTMO-Global/protocol/openid-connect/auth?redirect_uri=https%3A%2F%2Ftrader.ftmo.com%2Fapi%2Fauth%2Fcallback&scope=openid+email+profile&state=dca7c7b4-d3a6-4347-9e43-6a3f0840776e&client_id=ftmo-trader&response_type=code)

[5] - [Microsoft Azure](https://azure.microsoft.com/en-gb?ocid=cmm4r4ppnhp)

[6] - [AWS](https://aws.amazon.com/free/?trk=ea4f1bda-c329-47b3-a6d0-278045ca3aea&sc_channel=ps&ef_id=Cj0KCQjwlNPVBhCMARIsAPZ5RqhpVYKenMTYFY8V0exvBR-OSIQG_LbHO3uJ2i0Xnrq1FTorX9ZFswMaAkjSEALw_wcB:G:s&gads_camp=23528572796&gads_ag=192827516116&gads_ad=795841353781&gads_kw=amazon%20web%20services&gads_matchtype=e&gads_network=g&gads_device=c&gads_geo=9046082&gad_campaignid=23528572796&gbraid=0AAAAADjHtp9jRVT44JCOHgGp3azYfxg-L&gclid=Cj0KCQjwlNPVBhCMARIsAPZ5RqhpVYKenMTYFY8V0exvBR-OSIQG_LbHO3uJ2i0Xnrq1FTorX9ZFswMaAkjSEALw_wcB)

[7] - [FTMO account constraints](https://ftmo.com/en/symbols/?utm_source=google&utm_medium=cpc&utm_campaign=enx_uk_lower_q1-2026_purchase_Brand&utm_content=enx_uk_uk_keyword_mix_exact%2Bphrase&utm_id=23304324622&gad_source=1&gad_campaignid=23304324622&gbraid=0AAAAAobYLd8SxMoBLxYGjsMYYYpO-Gx1E&gclid=Cj0KCQjwlNPVBhCMARIsAPZ5RqjjKD3DQ0YpzNtD47Orvr7vyI0cgA5gatLDEXczV9pzpwWtI5KgL74aAr3pEALw_wcB)
