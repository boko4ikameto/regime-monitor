import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).item())


def _expected_duration(A: np.ndarray) -> np.ndarray:
    diag = np.diag(A).astype(float)
    out = np.empty_like(diag, dtype=float)
    for i, aii in enumerate(diag):
        if np.isclose(aii, 1.0):
            out[i] = np.inf
        else:
            out[i] = 1.0 / (1.0 - aii)
    return out


def read_schema_columns(parquet_path: str) -> list[str]:
    schema = pq.read_schema(parquet_path)
    cols = [n for n in schema.names if not n.startswith("__index_level_")]
    return cols


def read_parquet_via_pyarrow(parquet_path: str) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "index" in df.columns:
            maybe_idx = pd.to_datetime(df["index"], errors="coerce")
            if maybe_idx.notna().all():
                df = df.drop(columns=["index"])
                df.index = maybe_idx
        elif "__index_level_0__" in df.columns:
            maybe_idx = pd.to_datetime(df["__index_level_0__"], errors="coerce")
            if maybe_idx.notna().all():
                df = df.drop(columns=["__index_level_0__"])
                df.index = maybe_idx

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    return df


def infer_yf_tickers_from_schema(cols: list[str]) -> list[str]:
    tickers = set()

    anchored_suffixes = [
        "_ret_1d",
        "_realvol_21d",
        "_realvol_63d",
        "_realvol_126d",
        "_mom_21d",
        "_mom_63d",
        "_mom_126d",
        "_mdd_63d",
        "_mdd_126d",
    ]

    for c in cols:
        for sfx in anchored_suffixes:
            if c.endswith(sfx):
                tickers.add(c[:-len(sfx)])
                break

    return sorted(tickers)


def infer_ta_tickers_from_schema(cols: list[str]) -> list[str]:
    ta_suffixes = [
        "_RSI_14",
        "_MACD",
        "_MACD_signal",
        "_MACD_diff",
        "_StochRSI",
        "_StochRSI_K",
        "_StochRSI_D",
        "_ADX_14",
        "_DIp_14",
        "_DIn_14",
        "_OBV",
    ]

    tickers = set()
    for c in cols:
        for sfx in ta_suffixes:
            if c.endswith(sfx):
                tickers.add(c[:-len(sfx)])
                break

    return sorted(tickers)


def infer_date_bounds_from_stored_features(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) == 0:
        return None, None

    start = pd.to_datetime(df.index.min()).strftime("%Y-%m-%d")
    end = pd.to_datetime(df.index.max()).strftime("%Y-%m-%d")
    return start, end


def main():
    schema_path = "out_pca/features_raw.parquet"
    pca_path = "out_pca/pca_pipeline.joblib"
    hmm_path = "out_hmm/hmm7_model.joblib"
    outdir = "site"

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Missing schema file: {schema_path}")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"Missing PCA pipeline: {pca_path}")
    if not os.path.exists(hmm_path):
        raise FileNotFoundError(f"Missing HMM model: {hmm_path}")

    schema_cols = read_schema_columns(schema_path)
    schema_df = read_parquet_via_pyarrow(schema_path)

    pca_pipeline = joblib.load(pca_path)
    hmm_model = joblib.load(hmm_path)

    from pipeline_pca54 import Config, build_feature_matrix

    cfg = Config()

    cfg.yf_tickers = infer_yf_tickers_from_schema(schema_cols)
    cfg.ta_tickers = infer_ta_tickers_from_schema(schema_cols)

    inferred_start, _ = infer_date_bounds_from_stored_features(schema_df)
    if inferred_start is not None:
        cfg.start = inferred_start

    cfg.end = datetime.today().strftime("%Y-%m-%d")

    X = build_feature_matrix(cfg)

    if len(X) == 0:
        raise ValueError("Feature matrix is empty after build_feature_matrix().")

    missing_cols = [c for c in schema_cols if c not in X.columns]
    extra_cols = [c for c in X.columns if c not in schema_cols]

    X_aligned = X.reindex(columns=schema_cols)

    if missing_cols:
        raise RuntimeError(
            "Feature schema mismatch. Missing columns: "
            + ", ".join(missing_cols[:50])
        )

    if X_aligned.isna().any().any():
        nan_cols = X_aligned.columns[X_aligned.isna().any()].tolist()
        raise RuntimeError(
            "NaNs detected after schema alignment in columns: "
            + ", ".join(nan_cols[:50])
        )

    Z = pca_pipeline.transform(X_aligned.values)
    proba = hmm_model.predict_proba(Z)

    dt_last = X_aligned.index[-1]
    p_t = proba[-1]
    state = int(np.argmax(p_t))

    A = hmm_model.transmat_
    p_next = p_t @ A
    exp_dur = _expected_duration(A)

    os.makedirs(outdir, exist_ok=True)

    snapshot = {
        "asof_date": str(pd.to_datetime(dt_last).date()),
        "asof_timestamp": str(dt_last),
        "mode": "live",
        "current_regime": state,
        "regime_probability": _safe_float(p_t[state]),
        "all_regime_probabilities": [_safe_float(x) for x in p_t],
        "next_day_probabilities": [_safe_float(x) for x in p_next],
        "expected_durations": [_safe_float(x) for x in exp_dur],
        "schema_check": {
            "extra_cols_ignored_count": int(len(extra_cols)),
            "missing_cols_count": int(len(missing_cols)),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(os.path.join(outdir, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print("Snapshot generated successfully.")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
