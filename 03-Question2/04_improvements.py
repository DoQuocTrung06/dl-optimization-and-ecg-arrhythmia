import shutil
import pandas as pd

from src.config import *
from src.train import run_config
from src.utils import slug

base = pd.read_csv(TAB / "dl_results.csv").set_index("name").loc["CNN-LSTM"].to_dict()
base["name"] = "CNN-LSTM (base)"
base["delta_test_rmse"] = 0.0
results = [base]

EXPS = [
    ("+ Residual target", dict(residual=True)),
    ("+ Time features", dict(time_feats=True)),
    ("+ BiLSTM", dict(opts=dict(bidir=True))),
    ("+ Attention", dict(opts=dict(attn=True))),
    ("+ Dropout 0.2", dict(opts=dict(dropout=0.2))),
    ("+ Huber loss", dict(opts=dict(loss="huber"))),
    ("Window 48", dict(window=48)),
    ("Window 192", dict(window=192)),
]

singles = {}
for name, kw in EXPS:
    s, _ = run_config(name, "cnn_lstm", **kw)
    s["delta_test_rmse"] = s["test_rmse"] - base["test_rmse"]
    singles[name] = (s, kw)
    results.append(s)

# Combine every change that lowered validation RMSE
good = {n: v for n, v in singles.items() if v[0]["val_rmse"] < base["val_rmse"]}
print("Improving on val:", list(good))
if len(good) >= 2:
    cfg = dict(window=WINDOW, time_feats=False, residual=False, opts={})
    wins = {n: v for n, v in good.items() if n.startswith("Window")}
    if wins:
        cfg["window"] = min(wins.items(), key=lambda kv: kv[1][0]["val_rmse"])[1][1]["window"]
    for n, (s, kw) in good.items():
        if n.startswith("Window"):
            continue
        for k, v in kw.items():
            if k == "opts":
                cfg["opts"].update(v)
            else:
                cfg[k] = v
    s, _ = run_config("Combined", "cnn_lstm", **cfg)
    s["delta_test_rmse"] = s["test_rmse"] - base["test_rmse"]
    results.append(s)

df = pd.DataFrame(results)
best = df.loc[df["val_rmse"].idxmin()]
df["selected"] = df["name"] == best["name"]
df.to_csv(TAB / "improvements.csv", index=False)

if best["name"] == "CNN-LSTM (base)":
    shutil.copy(PRED / "cnn_lstm.npz", PRED / "best_improved.npz")
else:
    shutil.copy(PRED / f"{slug(best['name'])}.npz", PRED / "best_improved.npz")
print("Selected by val RMSE:", best["name"])
print(df[["name", "val_rmse", "test_mae", "test_rmse", "test_r2", "delta_test_rmse", "selected"]].round(3).to_string(index=False))
