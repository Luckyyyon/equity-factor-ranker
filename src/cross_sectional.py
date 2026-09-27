import pandas as pd
import numpy as np
from scipy.stats import mstats

def winsorize_series(series: pd.Series, lower_quantile: float = 0.025, upper_quantile: float = 0.975) -> pd.Series:
    """
    Winsorize a pandas series capping values outside [lower_quantile, upper_quantile].
    Handles NaNs safely.
    """
    valid_mask = series.notna()
    if valid_mask.sum() < 3:
        return series
    
    clean_vals = series[valid_mask]
    lower_bound = clean_vals.quantile(lower_quantile)
    upper_bound = clean_vals.quantile(upper_quantile)
    
    winsorized = series.clip(lower=lower_bound, upper=upper_bound)
    return winsorized

def calculate_z_score(series: pd.Series) -> pd.Series:
    """
    Standardizes a series to zero mean and unit variance.
    Z = (X - mean) / std
    """
    valid_mask = series.notna()
    if valid_mask.sum() < 2:
        return pd.Series(0.0, index=series.index)
    
    mean = series[valid_mask].mean()
    std = series[valid_mask].std(ddof=1)
    
    if std == 0 or pd.isna(std):
        z = pd.Series(0.0, index=series.index)
    else:
        z = (series - mean) / std
    
    # Fill remaining NaNs with 0 (neutral score)
    return z.fillna(0.0)

def standardize_factors(
    df: pd.DataFrame, 
    factor_cols: list, 
    sector_neutral: bool = True,
    sector_col: str = 'Sector',
    lower_q: float = 0.025, 
    upper_q: float = 0.975
) -> pd.DataFrame:
    """
    Performs winsorization and z-score normalization on specified factor columns.
    If sector_neutral is True, normalizes factors relative to sector peers.
    """
    result_df = df.copy()
    z_cols = []

    for col in factor_cols:
        z_col_name = f"Z_{col}"
        z_cols.append(z_col_name)
        
        # Step 1: Winsorization (overall or by sector)
        if sector_neutral and sector_col in df.columns:
            # Winsorize within sector
            winsorized = df.groupby(sector_col)[col].transform(
                lambda s: winsorize_series(s, lower_q, upper_q)
            )
            # Z-score within sector
            z_scores = df.groupby(sector_col)[col].transform(
                lambda s: calculate_z_score(winsorize_series(s, lower_q, upper_q))
            )
        else:
            winsorized = winsorize_series(df[col], lower_q, upper_q)
            z_scores = calculate_z_score(winsorized)
            
        result_df[z_col_name] = z_scores

    return result_df
