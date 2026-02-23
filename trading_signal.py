import pandas as pd
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm
from hermitian import Hermitian
from levy import Levy
from csv_transformer import load_and_filter_csv

# Configuration par défaut
DEFAULT_CONFIG = {
    "rolling_window": 30,
    "k_min": 2,
    "k_max": 10,
    "selection_percentile": 0.15,
    "use_permutation_test": False, # Désactivé par défaut pour la rapidité en live
    "n_permutations": 30,
    "permutation_confidence": 0.95,
    "fixed_noise_threshold": 1.1,
    "filter_min_weight": True,
    "min_weight": 0.03
}

def compute_noise_threshold(window_data, n_permutations=30, confidence=0.95):
    """Permutation test for Lévy score significance (Bennett et al., 2022).
    
    Shuffle the temporal ordering of returns to destroy lead-lag structure,
    recompute Lévy scores, and use the distribution of shuffled scores
    to determine a significance threshold.
    
    Under H0 (no lead-lag), the row-mean Lévy scores should be ~0.
    We reject H0 for assets whose |score| exceeds the confidence percentile
    of the null distribution.
    """
    null_scores = []
    returns_matrix = window_data.pct_change().dropna()
    
    for i in range(n_permutations):
        shuffled = returns_matrix.apply(np.random.permutation, axis=0)
        shuffled.index = returns_matrix.index
        shuffled_prices = (1 + shuffled).cumprod()
        shuffled_prices = pd.concat([window_data.iloc[[0]], shuffled_prices])
        
        levy_shuffled = Levy(price_panel=shuffled_prices)
        levy_matrix_shuffled = levy_shuffled.generate_levy_matrix()
        
        abs_mean_scores = levy_matrix_shuffled.mean(axis=1).abs()
        null_scores.extend(abs_mean_scores.values)
    
    return np.percentile(null_scores, confidence * 100)

def generate_live_signals(csv_path, config=DEFAULT_CONFIG):
    """
    Fonction principale appelée par le bot de trading.
    Retourne un DataFrame avec les actifs à trader, leurs poids et la direction.
    """
    # 1. Load and check data
    price_panel = load_and_filter_csv(csv_path)
    if len(price_panel) < config["rolling_window"]:
        logging.warning("Not enough data for rolling window.")
        return pd.DataFrame()

    window_data = price_panel.iloc[-config["rolling_window"]:].dropna(axis=1)

    # 2. Calculate levy matrix and scores
    levy = Levy(price_panel=window_data)
    levy_matrix = levy.generate_levy_matrix()
    mean_scores = levy_matrix.mean(axis=1)

    # 3. Noise threshold
    if config["use_permutation_test"]:
        threshold = compute_noise_threshold(window_data, config["n_permutations"], config["permutation_confidence"])
    else:
        threshold = config["fixed_noise_threshold"]
    
    significant_mask = mean_scores.abs() >= threshold
    significant_scores = mean_scores[significant_mask]
    
    if len(significant_scores) < 4: # Sécurité
        significant_scores = mean_scores

    # 4. Clustering et GCP (Global-Clustered Portfolio)
    adjacency_matrix = np.maximum(levy_matrix, 0)
    directed_net = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph)
    
    clusterer = Hermitian(directed_net=directed_net)

    ##Pour le CP 
    ##cluster_dict = clusterer.cluster_hermitian_opt(k_min=config["k_min"], k_max=config["k_max"])
    
    # 5. Sélection Leaders/Followers Globaux
    n_select = max(2, int(config["selection_percentile"] * len(significant_scores)))
    global_leaders = significant_scores.nlargest(n_select).index.tolist()
    global_followers = significant_scores.nsmallest(n_select).index.tolist()
    
    # 6. Signal et Poids
    returns = window_data.pct_change().iloc[-1]
    global_signal = returns[global_leaders].mean()
    direction = "LONG" if global_signal > 0 else "SHORT"
    
    follower_scores = mean_scores[global_followers].abs()
    weights = follower_scores / follower_scores.sum() if follower_scores.sum() != 0 else 1.0/len(global_followers)
    
    # 7. Formatage de la sortie
    final_output = pd.DataFrame({
        'ticker': global_followers,
        'weight': weights,
        'direction': [direction] * len(global_followers),
        'levy_score': mean_scores[global_followers].values
    })

    if config["filter_min_weight"]:
        final_output = final_output[final_output['weight'] >= config["min_weight"]]
        final_output['weight'] = final_output['weight'] / final_output['weight'].sum() # Renormalisation

    return final_output

## NE MARCHE QUE POUR LE GCP CAR LES CLUSTERS SONT DECONSTRUITS