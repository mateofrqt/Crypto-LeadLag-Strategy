import pandas as pd
import os
from portfolio import LeadLagPortfolio
from csv_transformer import load_and_filter_csv
import matplotlib.pyplot as plt
from portfolio import RiskConfig
# Read daily price panel from the csv file located in the data folder of the repo
# Filter to keep only data from 2026-01-07 onwards to have more cryptos available

start_date='2026-02-18'
end_date='2026-02-21'

daily_price_panel = load_and_filter_csv('/Users/mateofourquet/Desktop/data/data/Hyperliquid_ALL_COINS_1h.csv',
                                            start_date=start_date,
                                            end_date=end_date,
                                            )

# For csv already in the right format
# daily_price_panel = pd.read_csv("/Users/mateofourquet/Desktop/LeadLag/scr/prices_2026-01-25_20h_to_2026-01-26_04h.csv", index_col=0, parse_dates=True)

leadlag_port = LeadLagPortfolio(price_panel=daily_price_panel) 

# 2. CONFIGURER LE RISK MANAGEMENT AVANT les backtests
config = RiskConfig(
    enabled=False,
    target_volatility=0.25,
    max_leverage=10.0,
    max_drawdown=-0.15
)
leadlag_port.set_risk_config(config)

# Generate matrices and networks
leadlag_port.generate_matrices_and_networks(window_size=30, min_assets=40)

# Cluster directed networks using Hermitian algorithm
leadlag_port.cluster_directed_nets(k_min=3, k_max=10)

# Consider X% for leaders and followers selection
leadlag_port.find_cp_leaders_followers(selection_percentile=0.1)

# Backtest the clustered portfolio
leadlag_port.backtest_cp()
leadlag_port.backtest_gcp()
leadlag_port.backtest_gp()


fig_gp_cp_perf = leadlag_port.plot_portfolio_performance(rf=0.02, start_dt=start_date, end_dt=end_date,
                                                          cp=True, gcp=True, fig_size=(10,6))

os.makedirs('/Users/mateofourquet/Desktop/data/results', exist_ok=True)
fig_gp_cp_perf.savefig(f'/Users/mateofourquet/Desktop/data/results/portfolio_perf_with_risk_management_vol50percent.png')
print(leadlag_port.gcp_data[['PRet_Raw', 'VolScalar', 'PRet']].describe())
plt.show()
