import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def analyze_decile_dispersion(
    scored_df: pd.DataFrame, 
    prices_dict: Dict[str, pd.Series],
    decile_pct: float = 0.20
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Evaluates Top vs Bottom Decile (or Quintile) historical cumulative price dispersion spread.
    
    Parameters:
      - scored_df: DataFrame containing ticker index sorted by 'Percentile_Rank'
      - prices_dict: Dict mapping ticker to historical price Series
      - decile_pct: Fraction for top/bottom baskets (default top 20% vs bottom 20%)
      
    Returns:
      - Summary metrics DataFrame (Cumulative Return, Volatility, Sharpe, Max Drawdown)
      - Cumulative returns time-series DataFrame for Top Basket, Bottom Basket, Benchmark (Equal Weighted), Spread (Top - Bottom)
    """
    n = len(scored_df)
    if n < 5:
        return pd.DataFrame(), {}

    k = max(1, int(np.ceil(n * decile_pct)))
    
    top_tickers = list(scored_df.index[:k])
    bottom_tickers = list(scored_df.index[-k:])
    all_tickers = list(scored_df.index)

    # Reconstruct daily price table
    price_matrix = pd.DataFrame()
    for t in all_tickers:
        if t in prices_dict and len(prices_dict[t]) > 0:
            price_matrix[t] = prices_dict[t]

    price_matrix = price_matrix.dropna(how='all').fillna(method='ffill').dropna()
    
    if price_matrix.empty or len(price_matrix) < 10:
        return pd.DataFrame(), {}

    # Calculate daily simple returns
    daily_returns = price_matrix.pct_change().dropna()

    # Equal weighted basket returns
    top_daily = daily_returns[top_tickers].mean(axis=1) if any(t in daily_returns.columns for t in top_tickers) else pd.Series(0, index=daily_returns.index)
    bottom_daily = daily_returns[bottom_tickers].mean(axis=1) if any(t in daily_returns.columns for t in bottom_tickers) else pd.Series(0, index=daily_returns.index)
    benchmark_daily = daily_returns.mean(axis=1)

    # Spread daily return (Long Top / Short Bottom zero-cost spread)
    spread_daily = top_daily - bottom_daily

    # Cumulative Returns
    top_cum = (1 + top_daily).cumprod()
    bottom_cum = (1 + bottom_daily).cumprod()
    benchmark_cum = (1 + benchmark_daily).cumprod()
    spread_cum = (1 + spread_daily).cumprod()

    ts_df = pd.DataFrame({
        'Top Basket (Long)': top_cum,
        'Bottom Basket (Short)': bottom_cum,
        'Universe Benchmark (EQW)': benchmark_cum,
        'Long/Short Spread': spread_cum
    })

    # Summary performance metrics
    def get_stats(ret_series: pd.Series, cum_series: pd.Series) -> Dict:
        tot_ret = cum_series.iloc[-1] - 1.0 if len(cum_series) > 0 else 0.0
        ann_vol = ret_series.std() * np.sqrt(252) if len(ret_series) > 1 else 0.0
        ann_ret = (1 + tot_ret) ** (252 / max(len(ret_series), 1)) - 1.0 if len(ret_series) > 0 else 0.0
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0.0
        
        # Max Drawdown
        peak = cum_series.cummax()
        dd = (cum_series - peak) / peak
        max_dd = dd.min() if len(dd) > 0 else 0.0

        return {
            'Total Return': tot_ret,
            'Annualized Return': ann_ret,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio (Rf=3%)': sharpe,
            'Max Drawdown': max_dd
        }

    summary_df = pd.DataFrame({
        'Top Basket (Long)': get_stats(top_daily, top_cum),
        'Bottom Basket (Short)': get_stats(bottom_daily, bottom_cum),
        'Benchmark (EQW)': get_stats(benchmark_daily, benchmark_cum),
        'Long/Short Spread': get_stats(spread_daily, spread_cum)
    }).T

    return summary_df, ts_df
