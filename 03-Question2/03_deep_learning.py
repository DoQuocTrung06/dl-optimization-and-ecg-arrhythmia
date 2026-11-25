import pandas as pd

from src.config import *
from src.train import run_config

ARCHS = [("LSTM", "lstm"), ("1D-CNN", "cnn"), ("CNN-LSTM", "cnn_lstm")]

summaries, runs = [], []
for name, arch in ARCHS:
    s, df = run_config(name, arch)
    summaries.append(s)
    runs.append(df)

pd.DataFrame(summaries).to_csv(TAB / "dl_results.csv", index=False)
pd.concat(runs).to_csv(TAB / "dl_runs.csv", index=False)
print(pd.DataFrame(summaries)[["name", "params", "val_rmse", "test_mae", "test_rmse", "test_r2", "time_s", "epochs"]]
      .round(3).to_string(index=False))
