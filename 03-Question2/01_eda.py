import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import *
from src.data import load_raw

df = load_raw()
splits = dict(train=(0, TRAIN_END), val=(TRAIN_END, VAL_END), test=(VAL_END, TEST_END))

# Overview
print(df.shape, df.index.min(), df.index.max())
print("missing:", int(df.isna().sum().sum()))
print("gaps:", int((df.index.to_series().diff().dropna() != pd.Timedelta("15min")).sum()))
df.describe().T.to_csv(TAB / "describe.csv")

# Split info
info = []
for k, (a, b) in splits.items():
    s = df.iloc[a:b]
    info.append(dict(split=k, rows=len(s), start=s.index[0], end=s.index[-1],
                     OT_mean=s.OT.mean(), OT_std=s.OT.std(), OT_min=s.OT.min(), OT_max=s.OT.max()))
info = pd.DataFrame(info)
info.to_csv(TAB / "split_info.csv", index=False)
print(info)

colors = dict(train="tab:blue", val="tab:orange", test="tab:green")

# OT timeline
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df.OT, lw=0.4, color="gray")
for k, (a, b) in splits.items():
    ax.axvspan(df.index[a], df.index[b - 1], color=colors[k], alpha=0.15, label=k)
ax.set_title("Oil temperature (OT) and data split"); ax.set_ylabel("OT (°C)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "eda_01_ot_timeline.png", dpi=150); plt.close(fig)

# Load channels (daily mean)
daily = df.resample("D").mean()
fig, axes = plt.subplots(len(FEATURES), 1, figsize=(14, 12), sharex=True)
for ax, c in zip(axes, FEATURES):
    ax.plot(daily.index, daily[c], lw=0.8); ax.set_ylabel(c)
axes[0].set_title("Daily mean of each variable")
fig.tight_layout(); fig.savefig(FIG / "eda_02_variables_daily.png", dpi=150); plt.close(fig)

# Correlation
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix")
fig.tight_layout(); fig.savefig(FIG / "eda_03_correlation.png", dpi=150); plt.close(fig)

# Daily profile
fig, ax = plt.subplots(figsize=(8, 4))
prof = df.OT.groupby(df.index.hour).agg(["mean", "std"])
ax.plot(prof.index, prof["mean"]); ax.fill_between(prof.index, prof["mean"] - prof["std"], prof["mean"] + prof["std"], alpha=0.2)
ax.set_xlabel("Hour of day"); ax.set_ylabel("OT (°C)"); ax.set_title("Average OT by hour")
fig.tight_layout(); fig.savefig(FIG / "eda_04_hourly_profile.png", dpi=150); plt.close(fig)

# Monthly boxplot
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x=df.index.strftime("%Y-%m"), y=df.OT.values, ax=ax, fliersize=1)
ax.set_xlabel("Month"); ax.set_ylabel("OT (°C)"); ax.set_title("OT by month")
plt.setp(ax.get_xticklabels(), rotation=60)
fig.tight_layout(); fig.savefig(FIG / "eda_05_monthly_box.png", dpi=150); plt.close(fig)

# Autocorrelation
x = df.OT.values - df.OT.values.mean()
lags = np.arange(1, 4 * 96 + 1)
acf = np.array([np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in lags])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lags, acf)
for d in range(1, 5):
    ax.axvline(96 * d, color="red", ls="--", lw=0.7)
ax.set_xlabel("Lag (15-min steps)"); ax.set_ylabel("ACF"); ax.set_title("Autocorrelation of OT")
fig.tight_layout(); fig.savefig(FIG / "eda_06_acf.png", dpi=150); plt.close(fig)

# Distribution shift
fig, ax = plt.subplots(figsize=(8, 4))
for k, (a, b) in splits.items():
    sns.kdeplot(df.OT.iloc[a:b], ax=ax, label=k, color=colors[k], fill=True, alpha=0.2)
ax.set_xlabel("OT (°C)"); ax.set_title("OT distribution per split"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "eda_07_split_distribution.png", dpi=150); plt.close(fig)

print("EDA done")
