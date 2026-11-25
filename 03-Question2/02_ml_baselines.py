import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.config import *
from src.data import get_data
from src.features import make_features
from src.utils import metrics, slug

# Data
d = get_data(WINDOW, False)
mu, sd = d["mu"], d["sd"]
Ftr, names = make_features(d["train"]["X"])
Fva, _ = make_features(d["val"]["X"])
Fte, _ = make_features(d["test"]["X"])
ytr = d["train"]["y"]
yv = d["val"]["y"] * sd + mu
yt = d["test"]["y"] * sd + mu
print("features:", Ftr.shape, Fva.shape, Fte.shape)

rows = []


def save(name, pt):
    np.savez(PRED / f"{slug(name)}.npz", y_true=yt, y_pred=pt, idx=d["test"]["idx"])


# Naive
pv = d["val"]["X"][:, -1, TARGET_IDX] * sd + mu
pt = d["test"]["X"][:, -1, TARGET_IDX] * sd + mu
mt = metrics(yt, pt)
rows.append(dict(name="Naive", params="-", val_rmse=metrics(yv, pv)["rmse"], test_mae=mt["mae"],
                 test_rmse=mt["rmse"], test_r2=mt["r2"], time_s=0.0))
save("Naive", pt)
print("Naive", mt)


def tune(name, make, grid, residual=False):
    best = None
    btr = d["train"]["base"] if residual else 0.0
    bva = d["val"]["base"] if residual else 0.0
    bte = d["test"]["base"] if residual else 0.0
    for p in grid:
        t0 = time.time()
        m = make(**p).fit(Ftr, ytr - btr)
        t = time.time() - t0
        v = metrics(yv, (m.predict(Fva) + bva) * sd + mu)["rmse"]
        print(f"  {name} {p} val_rmse={v:.3f} ({t:.0f}s)", flush=True)
        if best is None or v < best[0]:
            best = (v, p, m, t)
    v, p, m, t = best
    pt = (m.predict(Fte) + bte) * sd + mu
    mt = metrics(yt, pt)
    rows.append(dict(name=name, params=str(p), val_rmse=v, test_mae=mt["mae"], test_rmse=mt["rmse"],
                     test_r2=mt["r2"], time_s=t))
    save(name, pt)
    print(name, mt, flush=True)
    return m, p


# Ridge
tune("Ridge", lambda **p: Ridge(**p), [dict(alpha=a) for a in (0.1, 1, 10, 100)])

# Random Forest
rf, rf_p = tune("Random Forest",
          lambda **p: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, **p),
          [dict(max_depth=12, min_samples_leaf=5), dict(max_depth=None, min_samples_leaf=5)])
pd.DataFrame(dict(feature=names, importance=rf.feature_importances_)).sort_values(
    "importance", ascending=False).to_csv(TAB / "rf_importance.csv", index=False)

# Gradient boosting
try:
    from xgboost import XGBRegressor
    GB_NAME = "XGBoost"
    gb_make = lambda **p: XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       tree_method="hist", n_jobs=-1, random_state=0, **p)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    GB_NAME = "HistGradientBoosting"
    gb_make = lambda **p: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0, **p)
_, gb_p = tune(GB_NAME, gb_make, [dict(max_depth=4), dict(max_depth=6)])

# Residual target (predict OT(t+H) - OT(t))
tune("Ridge (residual)", lambda **p: Ridge(**p), [dict(alpha=a) for a in (0.1, 1, 10, 100)], residual=True)
tune("Random Forest (residual)",
     lambda **p: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, **p), [rf_p], residual=True)
tune(GB_NAME + " (residual)", gb_make, [gb_p], residual=True)

pd.DataFrame(rows).to_csv(TAB / "ml_results.csv", index=False)
print(pd.DataFrame(rows).drop(columns="params").round(3).to_string(index=False))
