import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from levy import Levy
from hermitian import Hermitian

from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """
    Configuration pour le risk management intégré au backtest.
    
    Attributes:
    -----------
    enabled : bool
        Active/désactive le risk management
    target_volatility : float
        Volatilité annualisée cible (ex: 0.15 = 15%)
    vol_lookback : int
        Nombre de périodes pour calculer la vol réalisée
    vol_floor : float
        Vol minimum pour éviter le sur-levier
    vol_cap : float
        Vol maximum
    max_leverage : float
        Levier maximum autorisé
    min_leverage : float
        Levier minimum (0 = peut aller flat)
    max_drawdown : float
        Drawdown max avant circuit breaker (ex: -0.15 = -15%)
    cooldown_periods : int
        Périodes flat après déclenchement du circuit breaker
    periods_per_year : float
        Pour annualisation (365.25 pour daily, 365.25*24 pour hourly)
    """
    enabled: bool = True
    target_volatility: float = 0.15
    vol_lookback: int = 30
    vol_floor: float = 0.05
    vol_cap: float = 1.0
    max_leverage: float = 5
    min_leverage: float = 0.0
    max_drawdown: float = -0.15
    cooldown_periods: int = 5
    periods_per_year: float = 365.25


# =============================================================================
# RISK MANAGER CLASS
# =============================================================================

class RiskManager:
    """
    A class to manage volatility targeting and circuit breaker.
    
    Usage:
    ------
    >>> config = RiskConfig(target_volatility=0.15, max_leverage=2.0)
    >>> risk_mgr = RiskManager(config)
    >>> 
    >>> for i in range(n_periods):
    >>>     raw_ret = calculate_raw_return(...)
    >>>     result = risk_mgr.apply(raw_ret)
    >>>     adjusted_ret = result['adjusted_ret']
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset l'état pour un nouveau backtest."""
        self.return_accumulator = []
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.cooldown_counter = 0
        self.circuit_breaker_active = False
    
    def apply(self, raw_ret: float) -> Dict:
        """
        Applique vol targeting + circuit breaker à un return brut.
        
        Parameters:
        -----------
        raw_ret : float
            Return brut de la période
            
        Returns:
        --------
        Dict avec:
            - adjusted_ret: return après ajustement
            - vol_scalar: multiplicateur appliqué
            - realized_vol: vol réalisée annualisée
            - circuit_breaker: True si flat forcé
        """
        cfg = self.config
        
        # 1. VOLATILITY TARGETING
        if len(self.return_accumulator) >= cfg.vol_lookback:
            hist = pd.Series(self.return_accumulator)
            realized_vol = hist.iloc[-cfg.vol_lookback:].std() * np.sqrt(cfg.periods_per_year)
            realized_vol = np.clip(realized_vol, cfg.vol_floor, cfg.vol_cap)
            vol_scalar = np.clip(cfg.target_volatility / realized_vol,
                                 cfg.min_leverage, cfg.max_leverage)
        else:
            realized_vol = np.nan
            vol_scalar = 1.0
        
        # 2. APPLY SCALING
        adjusted_ret = raw_ret * vol_scalar
        
        # 3. CIRCUIT BREAKER
        self.current_equity *= (1 + adjusted_ret)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        current_dd = (self.current_equity - self.peak_equity) / self.peak_equity
        
        circuit_breaker = False
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            circuit_breaker = True
            adjusted_ret = 0.0
        elif current_dd <= cfg.max_drawdown:
            self.circuit_breaker_active = True
            self.cooldown_counter = cfg.cooldown_periods
            circuit_breaker = True
            adjusted_ret = 0.0
        else:
            self.circuit_breaker_active = False
        
        # 4. ACCUMULATE
        self.return_accumulator.append(adjusted_ret)
        
        return {
            'adjusted_ret': adjusted_ret,
            'vol_scalar': vol_scalar,
            'realized_vol': realized_vol,
            'circuit_breaker': circuit_breaker
        }


# =============================================================================
# LEAD-LAG PORTFOLIO CLASS
# =============================================================================

class LeadLagPortfolio():
    """
    A class for managing lead-lag portfolio analysis.

    Attributes:
    - price_panel (pd.DataFrame): DataFrame containing historical prices.
    - s (dict): Dictionary to store "S" matrices.
    - g (dict): Dictionary to store "G" directed graphs.
    - dt_cluster_dict (dict): Dictionary to store cluster information.
    - selection_pct (float): Percentage for asset selection.
    - global_ranking (pd.DataFrame): Global ranking of assets.
    - global_lfs (pd.Series): Global lead-lag factor series.
    - clustered_lfs (pd.Series): Clustered lead-lag factor series.
    - return_panel (pd.DataFrame): DataFrame containing returns.
    - gp_stats (pd.DataFrame): Global portfolio statistics.
    - cp_stats (pd.DataFrame): Clustered portfolio statistics.
    - risk_config (RiskConfig): Configuration for risk management.
    """
    def __init__(self, price_panel: pd.DataFrame):
        
        self.price_panel = price_panel   
        self.return_panel = pd.DataFrame()     
        
        # Initialize S matrix and directed network dicts
        self.s = {}  
        self.g = {}  
        
        # Initialize global portfolio items
        self.global_scores = pd.DataFrame()
        self.gp_leaders_followers = pd.Series()
        self.gp_data = pd.DataFrame()
        
        # Initialize clustered portfolio items
        self.dt_cluster_dict = {}
        self.cp_leaders_followers = {}
        self.cp_data = pd.DataFrame()
        self.gcp_data = pd.DataFrame()
        
        # Risk management
        self.risk_config: Optional[RiskConfig] = None
    
    
    def set_risk_config(self, config: RiskConfig):
        """
        Configure les paramètres de risk management pour les backtests.
        
        Parameters:
        -----------
        config : RiskConfig
            Configuration du risk management
            
        Example:
        --------
        >>> config = RiskConfig(target_volatility=0.15, max_leverage=2.0)
        >>> portfolio.set_risk_config(config)
        """
        self.risk_config = config
        print(f"Risk config set: target_vol={config.target_volatility:.0%}, "
              f"max_leverage={config.max_leverage}x, max_dd={config.max_drawdown:.0%}")
        
    
    # =========================================================================
    # EXISTING METHODS (unchanged)
    # =========================================================================
    
    def generate_matrices_and_networks(self, window_size: int = 30, min_assets: int = 40, show_progress: bool = True):
        """
        Generates lead-lag score matrices and relevant directed network for each index in the price panel 
        and stores each matrix and directed network in a dictionary with the index value as key.

        Parameters:
        - window_size (int): Size of the rolling window.
        - min_assets (int): Minimum number of assets required in the window.
        - show_progress (bool): Flag to show the progress bar.

        Note: Make sure to have the 'tqdm' library installed for the progress bar to work.
        """
        # Run a rolling window and generate lead-lag score matrices based on levy area
        for i in tqdm(range(1, len(self.price_panel) - window_size), 
                      desc='Generating Lead-Lag Scoring Matrices and Directed Networks', 
                      disable=not show_progress):
            
            # Slice the rolling window
            window_df = self.price_panel.iloc[i - 1:i + window_size, :]

            # Drop assets with missing data inside the window
            window_df = window_df.dropna(axis=1)

            # If more than min_assets are in the window, then continue
            if window_df.shape[1] >= min_assets:
                
                # Find the last index of the window as the key
                window_index = window_df.index[-1]

                # Generate Levy matrix, "S" for the rolling window and store
                levy_ll = Levy(price_panel=window_df)
                s_matrix = levy_ll.generate_levy_matrix()
                self.s[window_index] = s_matrix

                # Generate adjacency matrix, "A", convert to directed network, "G" and store
                a_matrix = np.maximum(s_matrix, 0)
                directed_net = nx.from_pandas_adjacency(a_matrix, create_using=nx.DiGraph)
                self.g[window_index] = directed_net
                

    def _calculate_return_panel(self):
        """
        Calculate the return panel based on the available scoring matrix "S".
        If the scoring matrix is not available, generate scores before calculating returns.
        """
        # Check if the scoring matrix is available
        if not self.s:
            # If not available, generate 
            self.generate_matrices_and_networks()

        # Calculate returns
        return_panel = self.price_panel.pct_change()
        
        # Find the earliest index from the scoring matrix "S" keys and filter the return panel
        earliest_idx = min(self.s.keys())
        self.return_panel = return_panel[earliest_idx:]
            

    def calculate_global_scores(self):
        """
        Calculate global scores for each asset based on the available scoring matrix "S".
        
        For every index in the S matrix, average every asset's lead-lag score with respect to all other assets
        as that asset's global score for that index. The scores are used for sorting assets from most likely 
        to be a leader to most likely to be a follower in constructing the global portfolio.
        """
        # Check if the scoring matrix is available
        if not self.s:
            # If not available, generate 
            self.generate_matrices_and_networks()
            
        global_scores = pd.DataFrame()
        for idx, s_matrix in self.s.items():
            dt_scores = pd.DataFrame({idx: s_matrix.mean(axis=1)}).T
            global_scores = pd.concat([global_scores, dt_scores])
            
        self.global_scores = global_scores
        

    def find_gp_leaders_followers(self, selection_percentile: float = 0.2):
        """
        Identify leaders and followers based on global ranking scores and the specified percentile.

        Parameters:
        - selection_percentile (float): The percentile used to determine leaders and followers.
        """
        # Check if the global scores are available
        if self.global_scores.empty:
            # If not, generate
            self.calculate_global_scores()
        
        # Utility function to find global leaders and followers for each index
        def identify_leaders_followers(row):
            non_nan_values = row.dropna()
            num_values = len(non_nan_values)
            follower_assets = non_nan_values.nsmallest(int(selection_percentile * num_values)).index
            leader_assets = non_nan_values.nlargest(int(selection_percentile * num_values)).index
            return follower_assets, leader_assets
        
        self.gp_leaders_followers = self.global_scores.apply(identify_leaders_followers, axis=1)
        
        
    def cluster_directed_nets(self, 
                              k_min: int = 3,
                              k_max: int = 10,
                              kmeans_init: str = 'k-means++',
                              kmeans_n_init: int = 10,
                              kmeans_random_state: int = 42,
                              show_progress: bool = True):
        """
        Clusters directed networks using the Hermitian clustering approach and stores the results.

        Parameters:
        - k_min (int): The minimum number of clusters for each network.
        - k_max (int): The maximum number of clusters for each network.
        - kmeans_init (str): Initialization method for KMeans. Default is 'k-means++'.
        - kmeans_n_init (int): Number of times KMeans will be run with different centroid seeds. Default is 10.
        - kmeans_random_state (int): Random state for KMeans. Default is 42.
        - show_progress (bool): Flag to show the progress bar.

        Note: Make sure to have the 'tqdm' library installed for the progress bar to work.
        """
        # Check if the directed networks are available
        if not self.g:
            # If not available, generate 
            self.generate_matrices_and_networks()

        # A dict of dicts, datetime and cluster number are keys, values are assets in each cluster    
        dt_cluster_dict = {}
        for dt, dt_g in tqdm(self.g.items(), 
                             desc='Clustering Lead-Lag Networks Using Hermitian Algorithm', 
                             disable=not show_progress):
            # Do Clustering
            clusterer = Hermitian(directed_net=dt_g)    
            dt_cluster_dict[dt] = clusterer.cluster_hermitian_opt(k_min=k_min,
                                                                  k_max=k_max, 
                                                                  kmeans_init=kmeans_init,
                                                                  kmeans_n_init=kmeans_n_init,
                                                                  kmeans_random_state=kmeans_random_state)
            
        self.dt_cluster_dict = dt_cluster_dict
        

    def find_cp_leaders_followers(self, selection_percentile: float = 0.2):
        """
        Identify leaders and followers based on the clustered lead-lag networks and the specified percentile.
        
        Parameters:
        - selection_percentile (float): The percentile used to determine leaders and followers.
        """        
        # Check availability of the backtest data
        if not self.dt_cluster_dict:
            raise ValueError("Clustered networks data is empty. Run 'cluster_directed_nets()' method with your desired parameters and try again.")
        
        dt_cl_lfs_dict = {}
        for dt, cluster_dict in self.dt_cluster_dict.items():

            # Slice score matrix for a given datetime/index
            dt_s = self.s[dt]

            # Do inter-cluster ranking and identify leaders and followers for each cluster for a given datetime
            cl_lfs_dict = {}
            for c_no, assets in cluster_dict.items():

                # Slice cluster assets from the dt_s matrix and calculate mean score of each asset
                dt_cluster_s = dt_s.loc[assets, assets]
                dt_cluster_score = dt_cluster_s.mean(axis=1)

                # Add clusters leaders/followers to a dict
                leaders = dt_cluster_score.nlargest(int(selection_percentile*len(dt_cluster_score))).index.tolist()
                followers = dt_cluster_score.nsmallest(int(selection_percentile*len(dt_cluster_score))).index.tolist()
                cl_lfs_dict[c_no] = {'leaders': leaders, 'followers': followers}

            # Add datetime'index clusters leader follower data
            dt_cl_lfs_dict[dt] = cl_lfs_dict
                
        self.cp_leaders_followers = dt_cl_lfs_dict
    
    
    # =========================================================================
    # HELPER: PROCESS BACKTEST RESULTS
    # =========================================================================
    
    def _build_backtest_dataframe(self, 
                                   entry_dt: list,
                                   exit_dt: list,
                                   raw_returns: list,
                                   risk_results: list,
                                   extra_columns: Dict = None) -> pd.DataFrame:
        """
        Construit le DataFrame de backtest avec les colonnes risk management.
        
        Parameters:
        -----------
        entry_dt, exit_dt : list
            Dates d'entrée et sortie
        raw_returns : list
            Returns bruts
        risk_results : list
            Liste de dicts retournés par RiskManager.apply()
        extra_columns : dict
            Colonnes supplémentaires (LRet, FRet, MktRet, etc.)
        """
        data = {
            'Entry': entry_dt,
            'Exit': exit_dt,
            'PRet_Raw': raw_returns,
            'VolScalar': [r['vol_scalar'] for r in risk_results],
            'RealizedVol': [r['realized_vol'] for r in risk_results],
            'CircuitBreaker': [r['circuit_breaker'] for r in risk_results],
        }
        
        if extra_columns:
            data.update(extra_columns)
        
        df = pd.DataFrame(data)
        
        # Final adjusted return
        df['PRet'] = df.apply(
            lambda x: 0.0 if x['CircuitBreaker'] else x['PRet_Raw'] * x['VolScalar'],
            axis=1
        )
        
        # Cumulative PnL
        df['PnL_Raw'] = (df['PRet_Raw'] + 1).cumprod()
        df['PnL'] = (df['PRet'] + 1).cumprod()
        
        # Set date index
        df['Date'] = df['Exit']
        df = df.set_index('Date')
        
        return df
    
    
    # =========================================================================
    # BACKTEST GP
    # =========================================================================

    def backtest_gp(self, selection_percentile: float = 0.2, show_progress: bool = True):
        """
        Performs the walk-forward backtest for the global portfolio by considering leaders and followers 
        identified based on global ranking scores.
        
        If risk_config is set (via set_risk_config), applies:
        - Volatility targeting: adjusts exposure based on realized vol vs target vol
        - Circuit breaker: goes flat if drawdown exceeds threshold

        Parameters:
        - selection_percentile (float): The percentile used to identify leaders and followers. Defaults to 0.2.
        - show_progress (bool): If True, display a progress bar during backtesting. Defaults to True.

        Returns:
        None: The backtest results are stored in the 'gp_data' attribute.
        """
        # Check if the global leaders/followers are available or not
        if self.gp_leaders_followers.empty:
            self.find_gp_leaders_followers(selection_percentile=selection_percentile)
        
        # Check if the return panel is available or not
        if self.return_panel.empty:
            self._calculate_return_panel()
        
        # Init risk manager
        risk_mgr = RiskManager(self.risk_config) if self.risk_config and self.risk_config.enabled else None
        
        # Lists
        leaders_ret = []
        followers_ret = []
        mkt_ret = []
        entry_dt = []
        exit_dt = []
        raw_returns = []
        risk_results = []
        
        # Find leaders, followers and market return, entry and exit datetimes for global portfolio
        for i in tqdm(range(len(self.gp_leaders_followers)),
                      desc='Generating Backtest Results for Global Portfolio', 
                      disable=not show_progress):
            
            # Entry and Exit datetime for positions
            entry_dt.append(self.return_panel.index[i])
            exit_dt.append(self.return_panel.index[i+1])
            
            # Global leaders and followers at time T
            followers = self.gp_leaders_followers.iloc[i][0].tolist()
            leaders = self.gp_leaders_followers.iloc[i][1].tolist()

            # Leaders return at time T, will be used for buy/sell signal of the followers at time T+1
            leader_ret = self.return_panel.iloc[i][leaders].mean()
            leaders_ret.append(leader_ret)
    
            # Followers and market return at time T+1, will be used to calculate GP performance
            follower_ret = self.return_panel.iloc[i+1][followers].mean()
            followers_ret.append(follower_ret)
            
            market_ret = self.return_panel.iloc[i+1].dropna().mean()
            mkt_ret.append(market_ret)
            
            # Raw return
            raw_ret = (follower_ret - market_ret) if leader_ret > 0 else (market_ret - follower_ret)
            raw_returns.append(raw_ret)
            
            # Risk management
            if risk_mgr:
                risk_results.append(risk_mgr.apply(raw_ret))
            else:
                risk_results.append({
                    'adjusted_ret': raw_ret,
                    'vol_scalar': 1.0,
                    'realized_vol': np.nan,
                    'circuit_breaker': False
                })
        
        # Build dataframe
        self.gp_data = self._build_backtest_dataframe(
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            raw_returns=raw_returns,
            risk_results=risk_results,
            extra_columns={
                'LRet': leaders_ret,
                'FRet': followers_ret,
                'MktRet': mkt_ret,
            }
        )
        
        # Add extra PnL columns
        self.gp_data['FPnL'] = (self.gp_data['FRet'] + 1).cumprod()
        self.gp_data['MktPnL'] = (self.gp_data['MktRet'] + 1).cumprod()
        
        
    # =========================================================================
    # BACKTEST CP
    # =========================================================================
    
    def _calc_clusters_weighted_return(self, t, t_plus_one):
        """
        Calculates all clusters portfolios as a single weighted portfolio
        """
        # Return and weight of each cluster's portfolio on the requested datetime
        cluster_returns = []
        cluster_weights = []
        
        # Iterate over the clusters, leaders/followers list of the given datetime/index
        for c_no, lf_dict in self.cp_leaders_followers[t].items():

            # Extract leaders/followers from lf_dict of the cluster
            cluster_leaders = lf_dict['leaders']
            cluster_followers = lf_dict['followers']
            
            # If there is at least one leader/follower, then continue
            if len(cluster_followers) > 0:
                cluster_universe = self.dt_cluster_dict[t][c_no]
                
                # Calculate leaders return at time T, followers and cluster universe return at time T+1
                cluster_leaders_ret = self.return_panel.loc[t][cluster_leaders].mean()
                cluster_followers_ret = self.return_panel.loc[t_plus_one][cluster_followers].mean()
                cluster_universe_ret = self.return_panel.loc[t_plus_one][cluster_universe].mean()
                
                # Clustered portfolio return
                if cluster_leaders_ret > 0:
                    cluster_return = cluster_followers_ret - cluster_universe_ret
                else:
                    cluster_return = cluster_universe_ret - cluster_followers_ret
                    
                # Consider number of followers as that clusters's weight
                cluster_weight = len(cluster_followers) 
                cluster_weights.append(cluster_weight) 
                cluster_returns.append(cluster_return)
                    
        cp_return = np.dot(cluster_returns, cluster_weights) / sum(cluster_weights)
        return cp_return 
        
        
    def backtest_cp(self, selection_percentile: float = 0.2, show_progress: bool = True):
        """
        Performs the walk-forward backtest for the clustered portfolio.
        
        If risk_config is set (via set_risk_config), applies:
        - Volatility targeting: adjusts exposure based on realized vol vs target vol
        - Circuit breaker: goes flat if drawdown exceeds threshold

        Parameters:
        - selection_percentile (float): The percentile used to identify leaders and followers. Defaults to 0.2.
        - show_progress (bool): If True, display a progress bar during backtesting. Defaults to True.

        Returns:
        None: The backtest results are stored in the 'cp_data' attribute.
        """
        # Check if the global leaders/followers are available or not
        if not self.cp_leaders_followers:
            self.find_cp_leaders_followers(selection_percentile=selection_percentile)
        
        # Check if the return panel is available or not
        if self.return_panel.empty:
            self._calculate_return_panel()
            self._calculate_return_panel()
        
        # Init risk manager
        risk_mgr = RiskManager(self.risk_config) if self.risk_config and self.risk_config.enabled else None
            
        # Lists
        entry_dt = []
        exit_dt = []
        raw_returns = []
        risk_results = []
        
        # Find leaders, followers and market return, entry and exit datetimes for global portfolio
        dt_list = list(self.cp_leaders_followers.keys())
        
        for i in tqdm(range(len(dt_list)-1),
                      desc='Generating Backtest Results for Clustered Portfolio', 
                      disable=not show_progress):
            
            # Entry and Exit datetime for positions 
            entry_dt.append(self.return_panel.index[i])
            exit_dt.append(self.return_panel.index[i+1])
            
            # Raw return from clusters
            raw_ret = self._calc_clusters_weighted_return(t=dt_list[i], t_plus_one=dt_list[i+1])
            raw_returns.append(raw_ret)
            
            # Risk management
            if risk_mgr:
                risk_results.append(risk_mgr.apply(raw_ret))
            else:
                risk_results.append({
                    'adjusted_ret': raw_ret,
                    'vol_scalar': 1.0,
                    'realized_vol': np.nan,
                    'circuit_breaker': False
                })
        
        # Build dataframe
        self.cp_data = self._build_backtest_dataframe(
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            raw_returns=raw_returns,
            risk_results=risk_results
        )
    
    
    # =========================================================================
    # BACKTEST GCP
    # =========================================================================
    
    def backtest_gcp(self, selection_percentile: float = 0.2, show_progress: bool = True):
        """
        Performs the walk-forward backtest for the global clustered portfolio by considering leaders and followers 
        identified based on global ranking scores. 
        
        If risk_config is set (via set_risk_config), applies:
        - Volatility targeting: adjusts exposure based on realized vol vs target vol
        - Circuit breaker: goes flat if drawdown exceeds threshold

        Parameters:
        - selection_percentile (float): The percentile used to identify leaders and followers. Defaults to 0.2.
        - show_progress (bool): If True, display a progress bar during backtesting. Defaults to True.

        Returns:
        None: The backtest results are stored in the 'gcp_data' attribute.
        
        New columns when risk_config is enabled:
        - PRet_Raw: raw return before risk adjustment
        - VolScalar: the leverage/scalar applied (target_vol / realized_vol)
        - RealizedVol: annualized realized volatility
        - CircuitBreaker: True if position was forced flat
        - PnL_Raw: cumulative PnL without risk management
        """
        # Check if the global leaders/followers are available or not
        if not self.cp_leaders_followers:
            print('find_cp_leaders_followers is empty')
            self.find_cp_leaders_followers(selection_percentile=selection_percentile)
        
        # Check if the return panel is available or not
        if self.return_panel.empty:
            self._calculate_return_panel()
        
        # Init risk manager
        risk_mgr = RiskManager(self.risk_config) if self.risk_config and self.risk_config.enabled else None
            
        # Lists
        leaders_ret = []
        followers_ret = []
        mkt_ret = []
        entry_dt = []
        exit_dt = []
        raw_returns = []
        risk_results = []
        
        dt_list = list(self.cp_leaders_followers.keys())
        
        for i in tqdm(range(len(self.cp_leaders_followers)),
                        desc='Generating Backtest Results for Global Clustered Portfolio', 
                        disable=not show_progress):
            
            # Entry and Exit datetime for positions            
            entry_dt.append(self.return_panel.index[i])
            exit_dt.append(self.return_panel.index[i+1])
            
            # Aggregate leaders and followers from all clusters
            leaders = [leader for leaders_list in self.cp_leaders_followers[dt_list[i]].values() 
                       for leader in leaders_list['leaders']]
            followers = [follower for follower_list in self.cp_leaders_followers[dt_list[i]].values() 
                         for follower in follower_list['followers']]
            
            leader_ret = self.return_panel.iloc[i][leaders].mean()
            leaders_ret.append(leader_ret)

            follower_ret = self.return_panel.iloc[i+1][followers].mean()
            followers_ret.append(follower_ret)
            
            market_ret = self.return_panel.iloc[i+1].dropna().mean()
            mkt_ret.append(market_ret)
            
            # Raw return
            raw_ret = (follower_ret - market_ret) if leader_ret > 0 else (market_ret - follower_ret)
            raw_returns.append(raw_ret)
            
            # Risk management
            if risk_mgr:
                risk_results.append(risk_mgr.apply(raw_ret))
            else:
                risk_results.append({
                    'adjusted_ret': raw_ret,
                    'vol_scalar': 1.0,
                    'realized_vol': np.nan,
                    'circuit_breaker': False
                })
        
        # Build dataframe
        self.gcp_data = self._build_backtest_dataframe(
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            raw_returns=raw_returns,
            risk_results=risk_results,
            extra_columns={
                'LRet': leaders_ret,
                'FRet': followers_ret,
                'MktRet': mkt_ret,
            }
        )
        
        # Add extra PnL columns
        self.gcp_data['FPnL'] = (self.gcp_data['FRet'] + 1).cumprod()
        self.gcp_data['MktPnL'] = (self.gcp_data['MktRet'] + 1).cumprod()
            
            
    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================
        
    def plot_portfolio_performance(self, rf: float = 0.02, start_dt: str = '2019-09-01', 
                                   end_dt: str = '2023-10-30', gcp: bool = True, 
                                   cp: bool = False, fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plots portfolio performance over time.
        """
        if self.gp_data.empty:
            raise ValueError("Global portfolio backtest data is empty. Run 'backtest_gp()' method and try again.")
        
        if cp and self.cp_data.empty:
            raise ValueError("Clustered portfolio backtest data is empty. Run 'backtest_cp()' method and try again.")
        
        if gcp and self.gcp_data.empty:
            raise ValueError("Global Clustered portfolio backtest data is empty. Run 'backtest_gcp()' method and try again.")

        # Copy global portfolio backtest data
        gp_data = self.gp_data.copy()
        gp_data = gp_data[(gp_data['Entry'] >= start_dt) & (gp_data['Entry'] <= end_dt)]
        gp_data['PnL'] = (gp_data['PRet'] + 1).cumprod()
        gp_data['DD'] = (gp_data['PnL'] - gp_data['PnL'].cummax()) / gp_data['PnL'].cummax()
        
        gp_ann_vol = np.std(gp_data['PRet']) * np.sqrt(365.25)
        gp_ann_ret = (gp_data['PnL'].iloc[-1]) ** (365.25 / len(gp_data)) - 1
        gp_ann_sr = (gp_ann_ret - rf) / gp_ann_vol
        gp_max_dd = gp_data['DD'].min()
        gp_metrics = ['GP', f'{gp_ann_ret*100:.1f}%', f'{gp_ann_vol*100:.1f}%',f'{gp_max_dd*100:.1f}%',f'{gp_ann_sr:.2f}']
        metrics = [gp_metrics]
            
        if cp:
            cp_data = self.cp_data.copy()
            cp_data = cp_data[(cp_data['Entry'] >= start_dt) & (cp_data['Entry'] <= end_dt)]
            cp_data['PnL'] = (cp_data['PRet'] + 1).cumprod()
            cp_data['DD'] = (cp_data['PnL'] - cp_data['PnL'].cummax()) / cp_data['PnL'].cummax()
            
            cp_ann_vol = np.std(cp_data['PRet']) * np.sqrt(365.25)
            cp_ann_ret = (cp_data['PnL'].iloc[-1]) ** (365.25 / len(cp_data)) - 1
            cp_ann_sr = (cp_ann_ret - rf) / cp_ann_vol
            cp_max_dd = cp_data['DD'].min()
            cp_metrics = ['CP', f'{cp_ann_ret*100:.1f}%', f'{cp_ann_vol*100:.1f}%',f'{cp_max_dd*100:.1f}%',f'{cp_ann_sr:.2f}']
            metrics.append(cp_metrics)
            
        if gcp:
            gcp_data = self.gcp_data.copy()
            gcp_data = gcp_data[(gcp_data['Entry'] >= start_dt) & (gcp_data['Entry'] <= end_dt)]
            gcp_data['PnL'] = (gcp_data['PRet'] + 1).cumprod()
            gcp_data['DD'] = (gcp_data['PnL'] - gcp_data['PnL'].cummax()) / gcp_data['PnL'].cummax()
            
            gcp_ann_vol = np.std(gcp_data['PRet']) * np.sqrt(365.25)
            gcp_ann_ret = (gcp_data['PnL'].iloc[-1]) ** (365.25 / len(gcp_data)) - 1
            gcp_ann_sr = (gcp_ann_ret - rf) / gcp_ann_vol
            gcp_max_dd = gcp_data['DD'].min()
            gcp_metrics = ['GCP', f'{gcp_ann_ret*100:.1f}%', f'{gcp_ann_vol*100:.1f}%',f'{gcp_max_dd*100:.1f}%',f'{gcp_ann_sr:.2f}']
            metrics.append(gcp_metrics)
            
        # Plot
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[1])
        ax0 = plt.subplot(gs[0], sharex=ax1)

        ax0.plot(gp_data.index, gp_data['PnL'], linewidth=1, label='GP')
        if cp:
            ax0.plot(cp_data.index, cp_data['PnL'], linewidth=1, label='CP')
        if gcp:
            ax0.plot(gcp_data.index, gcp_data['PnL'], linewidth=1, label='GCP')
            
        table = ax0.table(cellText=metrics, loc='lower right', 
                          colLabels=['Portfolio', r'$Ret_{ann}$', r'$Vol_{ann}$', r'$DD_{max}$', r'$SR_{ann}$'], 
                          cellLoc='center', colColours=['#f3f3f3']*5)
        table.auto_set_font_size(True)
        table.scale(0.4, 1.4)
        
        ax0.set_ylabel('Portfolio PnL')
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.legend(loc='upper left')
        
        ax1.plot(gp_data.index, gp_data['DD']*100, linewidth=1)
        if cp:
            ax1.plot(cp_data.index, cp_data['DD']*100, linewidth=1)
        if gcp:
            ax1.plot(gcp_data.index, gcp_data['DD']*100, linewidth=1)
        ax1.set_ylabel('DD (%)')
        
        plt.suptitle('Portfolio Performance Over Time', fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    
    def plot_monthly_returns(self, portfolio: str = 'GP', start_dt: str = '2019-09-01', 
                             end_dt: str = '2023-10-30', fig_size: Tuple[int, int] = (10, 4)) -> plt.Figure:
        """
        Plots a heatmap of monthly and year-to-date (YTD) returns for a given portfolio.
        """
        if portfolio == 'GP':
            portfolio_data = self.gp_data.copy()
            heatmap_title = 'Global'
        elif portfolio == 'CP':
            portfolio_data = self.cp_data.copy()
            heatmap_title = 'Clustered'
        elif portfolio == 'GCP':
            portfolio_data = self.gcp_data.copy()
            heatmap_title = 'Global Clustered'
        else:
            raise ValueError("Invalid entry for portfolio type. Portfolio type must be 'GP', 'GCP', or 'CP'.")
        
        portfolio_data = portfolio_data[(portfolio_data['Entry'] >= start_dt) & (portfolio_data['Entry'] <= end_dt)]

        monthly_ret = pd.DataFrame(portfolio_data['PRet'].resample('ME').agg(lambda x: (1 + x).prod() - 1))
        monthly_ret['Year'] = monthly_ret.index.year
        monthly_ret['Month'] = monthly_ret.index.strftime('%b')
        heatmap_data = monthly_ret.pivot_table(values='PRet', index='Year', columns='Month', margins_name='All')

        heatmap_data.fillna(0, inplace=True)
        heatmap_data_percentage = heatmap_data + 1
        heatmap_data['YTD'] = (heatmap_data_percentage.cumprod(axis=1).iloc[:, -1] - 1)
        heatmap_data.replace(0, np.nan, inplace=True)

        heatmap_data = 100 * heatmap_data
        
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','YTD']
        heatmap_data = heatmap_data.reindex(month_order, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [12, 1]}, figsize=fig_size, sharey=True)

        sns.heatmap(heatmap_data.drop(columns=['YTD']), annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5, center=0, ax=ax1)
        sns.heatmap(heatmap_data[['YTD']], annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5, center=0, ax=ax2)

        ax2.set_xlabel('')
        ax2.set_ylabel('')

        plt.tight_layout()
        plt.suptitle(f'{heatmap_title} Portfolio Monthly and YTD Returns', fontsize=11, fontweight='bold', y=1.05);
        
        return fig
