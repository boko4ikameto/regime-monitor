import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import yfinance as yf
from fredapi import Fred

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import dump

from dotenv import load_dotenv
load_dotenv()

# Technical indicators
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator


@dataclass
class Config:
    # Post-2020 domain only
    start: str = "2020-01-01"
    end: str = datetime.today().strftime("%Y-%m-%d")

    # PCA target
    pca_variance: float = 0.95

    # Output directory for the post-2020 model artifacts
    out_dir: str = "out_pca_2020"

    # Yahoo Finance tickers (proxy for S&P and Gold)
    yf_tickers = [
        "SPY",
        "GLD",
    ]

    # FRED series IDs (unchanged)
    fred_series = [
        "UNRATE",
        "FEDFUNDS",
        "T10Y2Y",
        "BAMLH0A0HYM2",
        "DCOILWTICO",
        "DEXUSEU",
        "VIXCLS",
    ]

    # Tickers used for OHLCV-based technical indicators (match focus assets)
    ta_tickers = ["SPY", "GLD"]


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns (especially with group_by variations).
    Normalize to single-level columns using level 0 (Open/High/Low/Close/Volume).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def yf_download_adjclose(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted (auto_adjust=True) close prices for multiple tickers.
    threads=False avoids sqlite/cache locking issues on Windows.
    """
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )

    closes = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")].rename(t)
    else:
        # single ticker fallback
        closes[tickers[0]] = df["Close"].rename(tickers[0])

    out = pd.concat(closes.values(), axis=1)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def yf_download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker, normalize columns.
    threads=False for stability on Windows.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    df.index = pd.to_datetime(df.index)
    df = normalize_yf_columns(df)
    return df


def fred_download_daily(series_ids: list[str], start: str, end: str, api_key: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    data = {}
    for sid in series_ids:
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        s.index = pd.to_datetime(s.index)
        data[sid] = s.rename(sid)

    # sort=False to silence pandas concat warning and make behavior explicit
    df = pd.concat(data.values(), axis=1, sort=False)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert to business-day calendar and forward/back fill
    bdays = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="B")
    df = df.reindex(bdays)
    df = df.ffill().bfill()
    return df


def calc_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)


def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    # annualized realized volatility, sqrt(252)
    return returns.rolling(window).std() * np.sqrt(252)


def rolling_momentum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    # cumulative return over window
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def rolling_max_drawdown(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    roll_max = prices.rolling(window).max()
    dd = prices / roll_max - 1.0
    return dd.rolling(window).min()


def add_ta_features(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    volume: pd.Series | None,
) -> pd.DataFrame:
    """
    Compute a compact set of technical indicators.
    Returns a DataFrame with generic column names; caller should add a prefix.
    """
    feats = {}

    # RSI
    feats["RSI_14"] = RSIIndicator(close=close, window=14).rsi()

    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    feats["MACD"] = macd.macd()
    feats["MACD_signal"] = macd.macd_signal()
    feats["MACD_diff"] = macd.macd_diff()

    # Stoch RSI
    st = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
    feats["StochRSI"] = st.stochrsi()
    feats["StochRSI_K"] = st.stochrsi_k()
    feats["StochRSI_D"] = st.stochrsi_d()

    # ADX family requires high/low/close
    if high is not None and low is not None:
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        feats["ADX_14"] = adx.adx()
        feats["DIp_14"] = adx.adx_pos()
        feats["DIn_14"] = adx.adx_neg()

    # OBV requires volume
    if volume is not None:
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        feats["OBV"] = obv.on_balance_volume()

    return pd.DataFrame(feats, index=close.index)


def build_feature_matrix(cfg: Config) -> pd.DataFrame:
    # Business day index for the whole period
    bdays = pd.date_range(start=pd.to_datetime(cfg.start), end=pd.to_datetime(cfg.end), freq="B")

    # 1) Yahoo closes for multiple tickers
    closes = yf_download_adjclose(cfg.yf_tickers, cfg.start, cfg.end)
    closes = closes.reindex(bdays).ffill().bfill()

    rets = calc_returns(closes)

    # 2) Engineered features: returns, vol, momentum, drawdowns
    feats = {}

    for c in closes.columns:
        feats[f"{c}_ret_1d"] = rets[c]

    for w in (21, 63, 126):
        rv = rolling_vol(rets, w)
        for c in rv.columns:
            feats[f"{c}_realvol_{w}d"] = rv[c]

    for w in (21, 63, 126):
        mom = rolling_momentum(rets, w)
        for c in mom.columns:
            feats[f"{c}_mom_{w}d"] = mom[c]

    for w in (63, 126):
        mdd = rolling_max_drawdown(closes, w)
        for c in mdd.columns:
            feats[f"{c}_mdd_{w}d"] = mdd[c]

    X = pd.DataFrame(feats, index=closes.index)

    # 3) Add macro series from FRED
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

    fred_key = os.getenv("FRED_API_KEY")
    if fred_key is None or fred_key == "":
        raise RuntimeError("Missing FRED_API_KEY. Put it into a .env file or environment variable.")

    try:
        fred_df = fred_download_daily(cfg.fred_series, cfg.start, cfg.end, api_key=fred_key)
        X = X.join(fred_df, how="left")
    except Exception as e:
        print("[WARN] FRED download failed, continuing without FRED series:", repr(e))

    # 4) Add technical indicators from OHLCV for selected tickers
    for t in cfg.ta_tickers:
        ohlcv = yf_download_ohlcv(t, cfg.start, cfg.end)
        ohlcv = ohlcv.reindex(X.index).ffill().bfill()

        if "Close" not in ohlcv.columns:
            continue

        close = ohlcv["Close"]
        high = ohlcv["High"] if "High" in ohlcv.columns else None
        low = ohlcv["Low"] if "Low" in ohlcv.columns else None
        volume = ohlcv["Volume"] if "Volume" in ohlcv.columns else None

        ta_df = add_ta_features(close=close, high=high, low=low, volume=volume)
        ta_df = ta_df.add_prefix(f"{t}_")

        X = X.join(ta_df, how="left")

    # 5) Clean and finalize
    X = X.replace([np.inf, -np.inf], np.nan)

    # Forward-fill to avoid losing full rows due to isolated missing values
    X = X.ffill()

    # Drop remaining rows with NaNs (early-window effects or any residual missing)
    X = X.dropna(axis=0, how="any")

    return X


def run_pca(cfg: Config) -> None:
    ensure_out_dir(cfg.out_dir)

    X = build_feature_matrix(cfg)

    pipe = Pipeline(
        steps=[
            ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))),
            ("pca", PCA(n_components=cfg.pca_variance, svd_solver="full", random_state=42)),
        ]
    )

    Z = pipe.fit_transform(X.values)
    pca: PCA = pipe.named_steps["pca"]

    n_comp = int(pca.n_components_)
    cols = [f"PC{str(i + 1).zfill(2)}" for i in range(n_comp)]
    Z_df = pd.DataFrame(Z, index=X.index, columns=cols)

    # Save outputs
    X.to_parquet(os.path.join(cfg.out_dir, "features_raw.parquet"))
    Z_df.to_parquet(os.path.join(cfg.out_dir, "pca_components.parquet"))
    Z_df.to_csv(os.path.join(cfg.out_dir, "pca_components.csv"), float_format="%.8f")

    dump(pipe, os.path.join(cfg.out_dir, "pca_pipeline.joblib"))

    summary = {
        "rows": int(X.shape[0]),
        "features_in": int(X.shape[1]),
        "pca_components": n_comp,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "start": cfg.start,
        "end": cfg.end,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    pd.Series(summary).to_json(os.path.join(cfg.out_dir, "run_summary.json"), indent=2)

    print("DONE")
    print(pd.Series(summary))


if __name__ == "__main__":
    run_pca(Config())