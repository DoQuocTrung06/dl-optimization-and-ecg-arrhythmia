import io
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import *

COLS = ["id_series", "date"] + FEATURES + ["time_step"]


def load_raw():
    if DATA_PATH.exists():
        lines = DATA_PATH.read_text().split("@DATA")[1].strip().splitlines()
        df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, quotechar="'", names=COLS)
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml(data_id=46209, as_frame=True).frame
        df.columns = COLS
        df = df.astype({c: float for c in FEATURES})
    df["date"] = pd.to_datetime(df["date"])
    return df.drop(columns=["id_series", "time_step"]).sort_values("date").set_index("date")


def time_features(idx):
    h = idx.hour.values + idx.minute.values / 60
    d = idx.dayofweek.values
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], 1)


def make_windows(arr, y, start, end, window, horizon):
    j = np.arange(max(start, window - 1 + horizon), end)
    e = j - horizon
    idx = e[:, None] + np.arange(-window + 1, 1)[None, :]
    return arr[idx], y[j], j


@lru_cache(maxsize=2)
def get_data(window=WINDOW, time_feats=False, horizon=HORIZON):
    df = load_raw()
    vals = df[FEATURES].values.astype(np.float32)
    sc = StandardScaler().fit(vals[:TRAIN_END])
    arr = sc.transform(vals).astype(np.float32)
    y = arr[:, TARGET_IDX].copy()
    if time_feats:
        arr = np.concatenate([arr, time_features(df.index).astype(np.float32)], 1)
    out = {}
    for name, (a, b) in dict(train=(0, TRAIN_END), val=(TRAIN_END, VAL_END), test=(VAL_END, TEST_END)).items():
        X, yy, j = make_windows(arr, y, a, b, window, horizon)
        out[name] = dict(X=X, y=yy, idx=j, base=X[:, -1, TARGET_IDX].copy())
    out["mu"], out["sd"] = float(sc.mean_[TARGET_IDX]), float(sc.scale_[TARGET_IDX])
    out["dates"] = df.index
    return out
