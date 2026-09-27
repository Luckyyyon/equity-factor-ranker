import pytest
import pandas as pd
import numpy as np
from src.cross_sectional import winsorize_series, calculate_z_score, standardize_factors
from src.composite_scorer import compute_composite_scores

def test_winsorize_series():
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, -50.0])
    winsorized = winsorize_series(data, lower_quantile=0.1, upper_quantile=0.9)
    assert winsorized.max() < 100.0
    assert winsorized.min() > -50.0

def test_calculate_z_score_properties():
    data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    z = calculate_z_score(data)
    
    # Check mean close to 0 and std close to 1
    np.testing.assert_almost_equal(z.mean(), 0.0, decimal=5)
    np.testing.assert_almost_equal(z.std(ddof=1), 1.0, decimal=5)

def test_no_nan_leaks():
    df = pd.DataFrame({
        'Sector': ['Tech', 'Tech', 'Industrial', 'Industrial', 'Tech'],
        'Value_EBITDA_EV': [0.1, np.nan, 0.15, 0.08, 0.12],
        'Quality_ROIC': [0.2, 0.15, np.nan, 0.10, 0.25],
        'Mom_6M': [0.05, 0.12, 0.20, np.nan, 0.08]
    }, index=['AAPL', 'MSFT', 'CAT', 'HON', 'NVDA'])
    
    factor_cols = ['Value_EBITDA_EV', 'Quality_ROIC', 'Mom_6M']
    std_df = standardize_factors(df, factor_cols, sector_neutral=True)
    
    for col in factor_cols:
        z_col = f"Z_{col}"
        assert not std_df[z_col].isna().any(), f"NaN leak found in {z_col}"

def test_composite_scorer():
    df = pd.DataFrame({
        'Z_Value_EBITDA_EV': [1.0, -1.0, 0.5],
        'Z_Value_FCF_Yield': [1.2, -0.8, 0.2],
        'Z_Value_Earnings_Yield': [0.8, -1.2, 0.4],
        'Z_Quality_ROIC': [1.5, -1.5, 0.0],
        'Z_Quality_Op_Margin': [1.1, -1.0, 0.1],
        'Z_Quality_Fin_Health': [0.9, -0.7, 0.3],
        'Z_Mom_12M_1M': [2.0, -2.0, 0.0],
        'Z_Mom_6M': [1.8, -1.8, 0.2]
    }, index=['A', 'B', 'C'])

    scored = compute_composite_scores(df)
    assert 'Percentile_Rank' in scored.columns
    assert 'Composite_Z' in scored.columns
    assert scored.loc['A', 'Percentile_Rank'] > scored.loc['C', 'Percentile_Rank']
    assert scored.loc['C', 'Percentile_Rank'] > scored.loc['B', 'Percentile_Rank']
