import pandas as pd
import numpy as np

def compute_raw_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes/formats raw fundamental and price factors from loaded universe dataframe.
    Ensures correct sign direction so higher values are ALWAYS better / more attractive.

    Raw Factors computed:
    1. Value:
       - EBITDA_to_EV: Inverted EV/EBITDA (Higher = Cheaper)
       - FCF_Yield: Free Cash Flow Yield (Higher = Cheaper/Cash Generative)
       - Earnings_Yield: 1 / P/E (Higher = Cheaper)
    2. Quality:
       - ROIC: Return on Invested Capital (Higher = Higher Quality)
       - Operating_Margin: Operating Income / Revenue (Higher = Higher Quality)
       - Financial_Health: Inverted Net Debt to EBITDA (-1 * Net_Debt_to_EBITDA, Higher = Lower Leverage)
    3. Momentum:
       - Mom_12M_1M: 12M minus 1M price return (Higher = Stronger Momentum)
       - Mom_6M: Trailing 6M price return (Higher = Stronger Momentum)
    """
    factors_df = pd.DataFrame(index=df.index)

    # Sector & Metadata
    factors_df['Sector'] = df['Sector']
    factors_df['Name'] = df['Name']
    factors_df['MarketCap'] = df['MarketCap']

    # 1. Value Factors
    factors_df['Value_EBITDA_EV'] = df['EBITDA_to_EV']
    factors_df['Value_FCF_Yield'] = df['FCF_Yield']
    factors_df['Value_Earnings_Yield'] = df['Earnings_Yield']

    # 2. Quality Factors
    factors_df['Quality_ROIC'] = df['ROIC']
    factors_df['Quality_Op_Margin'] = df['Operating_Margin']
    # Invert Net Debt / EBITDA so higher value represents lower leverage / better health
    factors_df['Quality_Fin_Health'] = -1.0 * df['Net_Debt_to_EBITDA']

    # 3. Momentum Factors
    factors_df['Mom_12M_1M'] = df['Mom_12M_1M']
    factors_df['Mom_6M'] = df['Mom_6M']

    return factors_df
