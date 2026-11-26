import shutil
import pandas as pd

from src.config import *
from src.train import run_config
from src.utils import slug

base = pd.read_csv(TAB / "dl_results.csv").set_index("name").loc["CNN-LSTM"].to_dict()
base["name"] = "CNN-LSTM (base)"
base["delta_val_rmse"] = 0.0
results = [base]
configs = {"CNN-LSTM (base)": None}

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
    # Select improvements using validation only; do not access test here.
    s, _ = run_config(name, "cnn_lstm", evaluate_test=False, **kw)
    s["delta_val_rmse"] = s["val_rmse"] - base["val_rmse"]
    singles[name] = (s, kw)
    configs[name] = kw
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
    s, _ = run_config("Combined", "cnn_lstm", evaluate_test=False, **cfg)
    s["delta_val_rmse"] = s["val_rmse"] - base["val_rmse"]
    configs["Combined"] = cfg
    results.append(s)

df = pd.DataFrame(results)
best = df.loc[df["val_rmse"].idxmin()]
df["selected"] = df["name"] == best["name"]

if best["name"] == "CNN-LSTM (base)":
    shutil.copy(PRED / "cnn_lstm.npz", PRED / "best_improved.npz")
else:
    # Evaluate test only after the validation-selected configuration is frozen.
    final_s, _ = run_config(best["name"], "cnn_lstm", evaluate_test=True,
                            **configs[best["name"]])
    for col in ["test_mae", "test_mae_std", "test_rmse", "test_rmse_std",
                "test_r2", "test_r2_std", "time_s", "epochs"]:
        df.loc[df["selected"], col] = final_s[col]
    shutil.copy(PRED / f"{slug(best['name'])}.npz", PRED / "best_improved.npz")
df.to_csv(TAB / "improvements.csv", index=False)
print("Selected by val RMSE:", best["name"])
print(df[["name", "val_rmse", "delta_val_rmse", "selected", "test_mae",
          "test_rmse", "test_r2"]].round(3).to_string(index=False))
