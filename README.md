# Does Exchange Fragmentation Distort Cryptocurrency Risk Metrics? Evidence from Cross-Venue Volatility and Stablecoin Peg Fragility  
Replication Code and Data

This repository contains the replication code, cleaned datasets, and output figures underlying the empirical analysis in the paper:

**“Does Exchange Fragmentation Distort Cryptocurrency Risk Metrics? Evidence from Cross-Venue Volatility and Stablecoin Peg Fragility”**

by S. Aljaloud and W. Alghassab - submitted to https://www.mdpi.com/journal/ijfs

Repository URL:

##  Repository Structure
crypto-fragmentation-risk/
├── data/
│ └── clean/
│ ├── depeg_predictions.csv
│ ├── panel_overlap_btc_eth.csv
│ ├── panel_overlap_btc_eth_3venues.csv
│ ├── spreads_overlap_btc_eth.csv
│ ├── spreads_overlap_btc_eth_3venues.csv
│ ├── spread_anomalies.csv
│ ├── usdt_depeg_dataset.csv
│ └── usdt_kraken.csv
├── outputs/
│ ├── calibration_logistic.png
│ ├── logit_coef_bootstrap_10bps.csv
│ ├── spread_rel_timeseries_BTC.png
│ ├── spread_rel_timeseries_ETH.png
│ ├── spread_rel_zscore_BTC.png
│ ├── spread_rel_zscore_ETH.png
│ ├── spread_summary_3venues.csv
│ └── threshold_robustness.csv
├── src/
│ ├── bootstrap_logit_inference.py
│ ├── build_depeg_dataset.py
│ ├── build_depeg_dataset_02.py
│ ├── calibration_plot.py
│ ├── config.py
│ ├── logistic_coefficients.py
│ ├── make_clean_panel.py
│ ├── make_clean_panel_3venues.py
│ ├── plot_spreads.py
│ ├── pull_binance_daily.py
│ ├── pull_coinbase_daily.py
│ ├── pull_kraken_daily.py
│ ├── threshold_robustness.py
│ ├── train_depeg_models.py
│ ├── var_comparison.py
│ ├── volatility_comparison.py
│ └── volatility_without_anomalies.py
├── README.md
├── requirements.txt
└── LICENSE

##  Cleaned Data (`data/clean/`)

The following cleaned datasets are provided:

- `depeg_predictions.csv`  
- `panel_overlap_btc_eth.csv`  
- `panel_overlap_btc_eth_3venues.csv`  
- `spreads_overlap_btc_eth.csv`  
- `spreads_overlap_btc_eth_3venues.csv`  
- `spread_anomalies.csv`  
- `usdt_depeg_dataset.csv`  
- `usdt_kraken.csv`

These files represent harmonized, analysis-ready data constructed from raw exchange API price data as described in the paper.

**Note:** Raw price data from exchanges are not included due to size and reproducibility via public APIs. Scripts in `src/` document how to reconstruct raw panels.

##  Output Figures and Summaries (`outputs/`)

The following outputs correspond to figures and tables in the paper:

- `calibration_logistic.png`  
- `spread_rel_timeseries_BTC.png`  
- `spread_rel_timeseries_ETH.png`  
- `spread_rel_zscore_BTC.png`  
- `spread_rel_zscore_ETH.png`  
- `spread_summary_3venues.csv`  
- `threshold_robustness.csv`  
- `logit_coef_bootstrap_10bps.csv`

These illustrate spread series, calibration diagnostics, and threshold robustness results.

##  Analysis Code (`src/`)

Python scripts to reproduce the key steps of the analysis are provided:

- `bootstrap_logit_inference.py` — performs walk-forward bootstrapped micro-depeg prediction  
- `build_depeg_dataset.py` — constructs the stablecoin micro-depeg panel  
- `build_depeg_dataset_02.py` — alternate dataset build  
- `calibration_plot.py` — generates calibration diagnostics  
- `config.py` — configuration settings  
- `logistic_coefficients.py` — extracts logistic coefficient summaries  
- `make_clean_panel.py` — builds cleaned panel data for BTC/ETH (two venues)  
- `make_clean_panel_3venues.py` — builds cleaned panel across three venues  
- `plot_spreads.py` — generates spread time-series plots  
- `pull_binance_daily.py` — retrieves Binance daily data  
- `pull_coinbase_daily.py` — retrieves Coinbase daily data  
- `pull_kraken_daily.py` — retrieves Kraken daily data  
- `threshold_robustness.py` — robustness evaluation across thresholds  
- `train_depeg_models.py` — trains logistic micro-depeg models  
- `var_comparison.py` — computes variance decomposition and comparison  
- `volatility_comparison.py` — compares volatility measures  
- `volatility_without_anomalies.py` — volatility excluding anomalies  

Scripts are organized to reflect the workflow described in the paper. See the Usage section below for execution order.

##  Requirements

Install with:
pip install -r requirements.txt

##  Contact

For questions: s.aljaloud@uoh.edu.sa