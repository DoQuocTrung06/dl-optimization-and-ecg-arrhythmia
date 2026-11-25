import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import *
from src.data import get_data
from src.utils import slug

ml = pd.read_csv(TAB / "ml_results.csv")
dl = pd.read_csv(TAB / "dl_results.csv")
imp = pd.read_csv(TAB / "improvements.csv")
dates = get_data()["dates"]

# Final comparison table
rows = []
for _, r in ml.iterrows():
    rows.append(dict(model=r["name"], group="Traditional ML" if r["name"] != "Naive" else "Baseline", key=slug(r["name"]),
                     mae=f"{r.test_mae:.3f}", rmse=f"{r.test_rmse:.3f}", r2=f"{r.test_r2:.3f}",
                     time_s=f"{r.time_s:.1f}", rmse_v=r.test_rmse))
for _, r in dl.iterrows():
    rows.append(dict(model=r["name"], group="Deep learning", key=slug(r["name"]),
                     mae=f"{r.test_mae:.3f} ± {r.test_mae_std:.3f}", rmse=f"{r.test_rmse:.3f} ± {r.test_rmse_std:.3f}",
                     r2=f"{r.test_r2:.3f} ± {r.test_r2_std:.3f}", time_s=f"{r.time_s:.0f}", rmse_v=r.test_rmse))
b = imp[imp["selected"]].iloc[0]
rows.append(dict(model=f"Best improved ({b['name']})", group="Improved", key="best_improved",
                 mae=f"{b.test_mae:.3f} ± {b.test_mae_std:.3f}", rmse=f"{b.test_rmse:.3f} ± {b.test_rmse_std:.3f}",
                 r2=f"{b.test_r2:.3f} ± {b.test_r2_std:.3f}", time_s=f"{b.time_s:.0f}", rmse_v=b.test_rmse))
final = pd.DataFrame(rows)
final.drop(columns=["key", "rmse_v"]).to_csv(TAB / "final_comparison.csv", index=False)
cols = ["model", "group", "mae", "rmse", "r2", "time_s"]
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
for _, r in final.iterrows():
    md += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
(TAB / "final_comparison.md").write_text(md)
print(md)

P = {r["model"]: np.load(PRED / f"{r['key']}.npz") for _, r in final.iterrows()}
ml_best = final[final.group == "Traditional ML"].sort_values("rmse_v").iloc[0]["model"]
dl_best = final[final.group == "Deep learning"].sort_values("rmse_v").iloc[0]["model"]
imp_name = final[final.group == "Improved"].iloc[0]["model"]
show = ["Naive", ml_best, dl_best, imp_name]
d0 = P[imp_name]
tdates = dates[d0["idx"]]
y = d0["y_true"]

# Fig 1: model comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
f = final.copy()
f["mae_v"] = f["mae"].str.split(" ").str[0].astype(float)
f = f.sort_values("rmse_v", ascending=False)
axes[0].barh(f["model"], f["rmse_v"], color="tab:blue"); axes[0].set_title("Test RMSE (°C)")
axes[1].barh(f["model"], f["mae_v"], color="tab:orange"); axes[1].set_title("Test MAE (°C)")
axes[1].set_yticklabels([])
fig.tight_layout(); fig.savefig(FIG / "res_01_model_comparison.png", dpi=150); plt.close(fig)

# Fig 2: predicted vs actual
n = 96 * 7
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(tdates[:n], y[:n], "k", lw=1.5, label="Actual")
for m, ls in zip(show, ["--", "-", "-", "-"]):
    ax.plot(tdates[:n], P[m]["y_pred"][:n], ls, lw=1, label=m)
ax.set_ylabel("OT (°C)"); ax.set_title("First 7 days of test set"); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "res_02_pred_vs_actual.png", dpi=150); plt.close(fig)

# Fig 3: scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, m in zip(axes, show[1:]):
    ax.scatter(y, P[m]["y_pred"], s=2, alpha=0.3)
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], "r--"); ax.set_title(m, fontsize=9); ax.set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
fig.tight_layout(); fig.savefig(FIG / "res_03_scatter.png", dpi=150); plt.close(fig)

# Fig 4: residual histogram
fig, ax = plt.subplots(figsize=(8, 4.5))
for m in show:
    ax.hist(P[m]["y_pred"] - y, bins=80, alpha=0.4, label=m, range=(-8, 8))
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Residual (pred - actual, °C)"); ax.legend(fontsize=8)
ax.set_title("Residual distribution (test)")
fig.tight_layout(); fig.savefig(FIG / "res_04_residuals.png", dpi=150); plt.close(fig)

# Fig 5-7: error breakdown
err = pd.DataFrame({m: np.abs(P[m]["y_pred"] - y) for m in show})
err["hour"] = tdates.hour
err["month"] = tdates.strftime("%Y-%m")
err["level"] = pd.qcut(y, 5, duplicates="drop")
for key, title, fn in [("hour", "MAE by hour of day", "res_05_error_hour"),
                       ("month", "MAE by month", "res_06_error_month"),
                       ("level", "MAE by actual OT level (quintiles)", "res_07_error_level")]:
    g = err.groupby(key, observed=True)[show].mean()
    ax = g.plot(kind="line", marker="o", figsize=(10, 4)) if key == "hour" else g.plot(kind="bar", figsize=(10, 4))
    ax.set_ylabel("MAE (°C)"); ax.set_title(title); plt.xticks(rotation=45 if key != "hour" else 0)
    plt.tight_layout(); plt.savefig(FIG / f"{fn}.png", dpi=150); plt.close()
    g.round(3).to_csv(TAB / f"{fn}.csv")

# Fig 8: loss curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (_, r) in zip(axes, dl.iterrows()):
    h = json.loads((HIST / f"{slug(r['name'])}.json").read_text())
    ax.plot(h["loss"], label="train"); ax.plot(h["val_loss"], label="val")
    ax.set_title(r["name"]); ax.set_xlabel("Epoch"); ax.legend()
axes[0].set_ylabel("MSE loss (scaled)")
fig.tight_layout(); fig.savefig(FIG / "res_08_loss_curves.png", dpi=150); plt.close(fig)

# Fig 9: improvements
fig, ax = plt.subplots(figsize=(9, 5))
im = imp.iloc[::-1]
ax.barh(im["name"], im["test_rmse"], xerr=im["test_rmse_std"], color=["tab:green" if s else "tab:gray" for s in im["selected"]])
ax.axvline(imp.iloc[0]["test_rmse"], color="r", ls="--", lw=0.8)
ax.set_xlabel("Test RMSE (°C)"); ax.set_title("Effect of each improvement (green = selected by val RMSE)")
fig.tight_layout(); fig.savefig(FIG / "res_09_improvements.png", dpi=150); plt.close(fig)

# Fig 10: RF importance
imp_rf = pd.read_csv(TAB / "rf_importance.csv").head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp_rf["feature"], imp_rf["importance"]); ax.set_title("Random Forest feature importance (top 15)")
fig.tight_layout(); fig.savefig(FIG / "res_10_rf_importance.png", dpi=150); plt.close(fig)

# Error summary + worst cases
summ = []
for m in show:
    e = P[m]["y_pred"] - y
    summ.append(dict(model=m, bias=e.mean(), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                     p95_abs=np.percentile(np.abs(e), 95), max_abs=np.abs(e).max()))
pd.DataFrame(summ).round(3).to_csv(TAB / "error_summary.csv", index=False)
print(pd.DataFrame(summ).round(3).to_string(index=False))
w = pd.DataFrame(dict(date=tdates, actual=y, pred=d0["y_pred"], abs_err=np.abs(d0["y_pred"] - y)))
w.sort_values("abs_err", ascending=False).head(15).to_csv(TAB / "worst_errors.csv", index=False)
print("Analysis done")
