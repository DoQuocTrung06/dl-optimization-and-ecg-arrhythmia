# ETTm1 - Oil temperature forecasting (CNN / LSTM)

Task: predict OT (transformer oil temperature) 24h ahead (96 steps) from the previous 24h (96 steps) of 7 variables.
Split by time: 12 / 4 / 4 months (train / val / test). Scaler fitted on train only.

## Run
```
pip install -r requirements.txt
python run_all.py          # full run (3 seeds, up to 40 epochs)
python run_all.py --fast   # quick smoke test
```
Or one step at a time: `01_eda.py`, `02_ml_baselines.py`, `03_deep_learning.py`, `04_improvements.py`, `05_analysis.py`.

## Outputs
- `results/tables/`: describe, split_info, ml_results, dl_results, improvements, final_comparison (.csv/.md), error tables
- `results/figures/`: eda_*.png, res_*.png
