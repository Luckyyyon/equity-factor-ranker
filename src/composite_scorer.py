import pandas as pd
import numpy as np
from scipy.stats import rankdata
from typing import Dict

DEFAULT_FACTOR_WEIGHTS = {
    # Category weights
    'Value': 0.35,
    'Quality': 0.35,
    'Momentum': 0.30,
    
    # Sub-factor weights inside category
    'Value_EBITDA_EV': 0.40,
    'Value_FCF_Yield': 0.40,
    'Value_Earnings_Yield': 0.20,
    
    'Quality_ROIC': 0.40,
    'Quality_Op_Margin': 0.35,
    'Quality_Fin_Health': 0.25,
    
    'Mom_12M_1M': 0.60,
    'Mom_6M': 0.40
}

def compute_composite_scores(
    df: pd.DataFrame,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Computes category composite Z-scores and overall Composite Factor Score & Percentile Rank.
    """
    if weights is None:
        weights = DEFAULT_FACTOR_WEIGHTS

    res_df = df.copy()

    # Category Z-score calculations
    # 1. Value Composite Z
    v_cols = ['Z_Value_EBITDA_EV', 'Z_Value_FCF_Yield', 'Z_Value_Earnings_Yield']
    v_w = [weights.get('Value_EBITDA_EV', 0.4), weights.get('Value_FCF_Yield', 0.4), weights.get('Value_Earnings_Yield', 0.2)]
    v_w_sum = sum(v_w) or 1.0
    v_norm_w = [w / v_w_sum for w in v_w]
    res_df['Value_Composite_Z'] = sum(res_df[c] * w for c, w in zip(v_cols, v_norm_w) if c in res_df.columns)

    # 2. Quality Composite Z
    q_cols = ['Z_Quality_ROIC', 'Z_Quality_Op_Margin', 'Z_Quality_Fin_Health']
    q_w = [weights.get('Quality_ROIC', 0.4), weights.get('Quality_Op_Margin', 0.35), weights.get('Quality_Fin_Health', 0.25)]
    q_w_sum = sum(q_w) or 1.0
    q_norm_w = [w / q_w_sum for w in q_w]
    res_df['Quality_Composite_Z'] = sum(res_df[c] * w for c, w in zip(q_cols, q_norm_w) if c in res_df.columns)

    # 3. Momentum Composite Z
    m_cols = ['Z_Mom_12M_1M', 'Z_Mom_6M']
    m_w = [weights.get('Mom_12M_1M', 0.6), weights.get('Mom_6M', 0.4)]
    m_w_sum = sum(m_w) or 1.0
    m_norm_w = [w / m_w_sum for w in m_w]
    res_df['Momentum_Composite_Z'] = sum(res_df[c] * w for c, w in zip(m_cols, m_norm_w) if c in res_df.columns)

    # Main Category weights
    cat_val_w = weights.get('Value', 0.35)
    cat_qual_w = weights.get('Quality', 0.35)
    cat_mom_w = weights.get('Momentum', 0.30)
    cat_sum = cat_val_w + cat_qual_w + cat_mom_w or 1.0

    cat_val_w /= cat_sum
    cat_qual_w /= cat_sum
    cat_mom_w /= cat_sum

    # Overall Composite Score Z
    res_df['Composite_Z'] = (
        cat_val_w * res_df['Value_Composite_Z'] +
        cat_qual_w * res_df['Quality_Composite_Z'] +
        cat_mom_w * res_df['Momentum_Composite_Z']
    )

    # Percentile Rank (0 to 100)
    n = len(res_df)
    if n > 0:
        ranks = rankdata(res_df['Composite_Z'], method='average')
        res_df['Percentile_Rank'] = (ranks - 1) / (n - 1 if n > 1 else 1) * 100.0
    else:
        res_df['Percentile_Rank'] = 0.0

    # Sort descending by Percentile Rank
    res_df = res_df.sort_values(by='Percentile_Rank', ascending=False)

    return res_df
