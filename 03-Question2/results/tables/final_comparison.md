| model | group | mae | rmse | r2 | time_s |
|---|---|---|---|---|---|
| Naive | Baseline | 1.530 | 1.967 | 0.609 | 0.0 |
| Ridge | Traditional ML | 1.809 | 2.328 | 0.452 | 0.0 |
| Random Forest | Traditional ML | 4.387 | 5.032 | -1.558 | 30.8 |
| XGBoost | Traditional ML | 4.241 | 4.833 | -1.360 | 3.0 |
| Ridge (residual) | Traditional ML | 1.799 | 2.318 | 0.457 | 0.0 |
| Random Forest (residual) | Traditional ML | 3.542 | 4.289 | -0.859 | 46.4 |
| XGBoost (residual) | Traditional ML | 3.853 | 4.518 | -1.063 | 2.9 |
| LSTM | Deep learning | 3.994 ± 0.199 | 4.441 ± 0.200 | -0.996 ± 0.180 | 154 |
| 1D-CNN | Deep learning | 4.834 ± 0.707 | 5.258 ± 0.653 | -1.836 ± 0.709 | 19 |
| CNN-LSTM | Deep learning | 4.615 ± 0.349 | 4.998 ± 0.305 | -1.533 ± 0.306 | 51 |
| Best improved (+ Residual target) | Improved | 3.203 ± 0.148 | 3.689 ± 0.128 | -0.377 ± 0.095 | 52 |
