import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import concurrent.futures

# Default sample universe (S&P 500 representative tickers across tech, industrial, healthcare, consumer, finance)
DEFAULT_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'JNJ', 'JPM', 'PG',
    'V', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'PFE', 'DIS', 'AVGO', 'CSCO',
    'CVX', 'ABBV', 'PEP', 'KO', 'MRK', 'TMO', 'COST', 'WMT', 'MCD', 'ACN',
    'ADBE', 'LIN', 'LLY', 'NKE', 'ORCL', 'CRM', 'AMD', 'TXN', 'PM', 'HON',
    'INTC', 'UNP', 'LOW', 'SPGI', 'QCOM', 'CAT', 'GS', 'MS', 'RTX', 'AMAT'
]

def get_ticker_info_and_history(ticker_symbol: str) -> Dict:
    """
    Fetch fundamental info and historical OHLCV data for a given ticker.
    Returns a dictionary of raw metric fields and price series.
    """
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info or {}
        
        # Historical price data for momentum (1 year + 1 month roughly 280 trading days)
        hist = t.history(period="1y")
        
        if hist.empty or len(hist) < 30:
            return None

        # Extract sector and industry
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        short_name = info.get("shortName", ticker_symbol)
        market_cap = info.get("marketCap", np.nan)

        # Financial Ratios & Raw Numbers
        # Value inputs
        ev = info.get("enterpriseValue", np.nan)
        ebitda = info.get("ebitda", np.nan)
        operating_cashflow = info.get("operatingCashflow", np.nan)
        free_cashflow = info.get("freeCashflow", np.nan)
        pe_ratio = info.get("forwardPE", info.get("trailingPE", np.nan))
        
        # Quality inputs
        return_on_assets = info.get("returnOnAssets", np.nan)
        return_on_equity = info.get("returnOnEquity", np.nan)
        operating_margins = info.get("operatingMargins", np.nan)
        total_debt = info.get("totalDebt", np.nan)
        total_cash = info.get("totalCash", np.nan)
        
        # Compute ROIC proxy if detailed balance sheet isn't direct in info
        # ROIC proxy: ebitda / (market_cap + total_debt - total_cash) or returnOnEquity fallback
        enterprise_val = market_cap + total_debt - total_cash if (pd.notnull(market_cap) and pd.notnull(total_debt) and pd.notnull(total_cash)) else ev
        roic = (ebitda / enterprise_val) if (pd.notnull(ebitda) and pd.notnull(enterprise_val) and enterprise_val > 0) else return_on_equity

        # Inverted EV/EBITDA (EBITDA / EV)
        ebitda_to_ev = (ebitda / ev) if (pd.notnull(ebitda) and pd.notnull(ev) and ev > 0) else np.nan
        
        # FCF Yield = FCF / Market Cap
        fcf_yield = (free_cashflow / market_cap) if (pd.notnull(free_cashflow) and pd.notnull(market_cap) and market_cap > 0) else np.nan
        
        # Earnings Yield = 1 / P/E
        earnings_yield = (1.0 / pe_ratio) if (pd.notnull(pe_ratio) and pe_ratio > 0) else np.nan

        # Net Debt / EBITDA (Inverted for ranking later)
        net_debt = (total_debt - total_cash) if (pd.notnull(total_debt) and pd.notnull(total_cash)) else np.nan
        net_debt_to_ebitda = (net_debt / ebitda) if (pd.notnull(net_debt) and pd.notnull(ebitda) and ebitda > 0) else np.nan

        # Momentum calculations
        # 12M - 1M Momentum: Return from t-252 to t-21
        # 6M Momentum: Return over trailing 126 trading days
        prices = hist['Close']
        n_days = len(prices)
        
        mom_12m_1m = np.nan
        mom_6m = np.nan
        
        if n_days >= 252:
            p_start = prices.iloc[-252]
            p_mid = prices.iloc[-21]
            if p_start > 0:
                mom_12m_1m = (p_mid - p_start) / p_start
        elif n_days >= 126:
            p_start = prices.iloc[0]
            p_mid = prices.iloc[-21] if n_days >= 21 else prices.iloc[-1]
            if p_start > 0:
                mom_12m_1m = (p_mid - p_start) / p_start

        if n_days >= 126:
            p_start_6m = prices.iloc[-126]
            p_current = prices.iloc[-1]
            if p_start_6m > 0:
                mom_6m = (p_current - p_start_6m) / p_start_6m
        elif n_days >= 60:
            p_start_6m = prices.iloc[0]
            p_current = prices.iloc[-1]
            if p_start_6m > 0:
                mom_6m = (p_current - p_start_6m) / p_start_6m

        return {
            'Ticker': ticker_symbol,
            'Name': short_name,
            'Sector': sector,
            'Industry': industry,
            'MarketCap': market_cap,
            'EBITDA_to_EV': ebitda_to_ev,
            'FCF_Yield': fcf_yield,
            'Earnings_Yield': earnings_yield,
            'ROIC': roic,
            'Operating_Margin': operating_margins,
            'Net_Debt_to_EBITDA': net_debt_to_ebitda,
            'Mom_12M_1M': mom_12m_1m,
            'Mom_6M': mom_6m,
            'Prices': prices
        }
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def load_universe_data(tickers: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Parallel fetch fundamental metrics and prices for list of tickers.
    Returns:
       - DataFrame of cross-sectional fundamental and factor metrics
       - Dict of historical price series per ticker
    """
    if tickers is None or len(tickers) == 0:
        tickers = DEFAULT_UNIVERSE
        
    records = []
    prices_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(get_ticker_info_and_history, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            if data is not None:
                prices_dict[data['Ticker']] = data.pop('Prices')
                records.append(data)

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('Ticker', inplace=True)
    return df, prices_dict
