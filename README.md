# Deep Learning Optimization and ECG Arrhythmia Classification

This repository contains PyTorch implementations and experimental evaluations for two deep learning studies:

1. **MLP Optimization and Regularization (`01-mlp-optimization-dry-bean`)**: Analysis of Multilayer Perceptron (MLP) training via Backpropagation on the UCI Dry Bean dataset (13,611 raw samples, 16 geometric features, 7 classes). Experiments compare five optimizers (Gradient Descent, SGD / Mini-batch SGD, SGD with Momentum, RMSprop, Adam) and regularization techniques (L1/L2 regularization, Dropout, Early Stopping).
2. **ECG Arrhythmia Classification (`02-ecg-arrhythmia-cnn-lstm`)**: Comparative evaluation of traditional machine learning baselines (Random Forest, SVM/XGBoost) against deep learning architectures (1D-CNN, LSTM, BiLSTM, and Hybrid CNN-BiLSTM) on the MIT-BIH Arrhythmia dataset (5 heartbeat classes, 187 time steps), including class-imbalance handling, ablation studies, and error analysis.

## Repository Structure

```text
├── 01-mlp-optimization-dry-bean/   # MLP experiments, learning curves, and output figures
├── 02-ecg-arrhythmia-cnn-lstm/     # ECG EDA, traditional ML baselines, CNN/LSTM models, error analysis
├── data/
│   ├── raw/                        # Local raw dataset files (ignored by Git; see Data Acquisition)
│   └── processed/                  # Saved Train/Validation/Test splits (SEED=42)
├── docs/                           # Technical report and presentation materials
├── requirements.txt                # Shared Python dependencies
└── README.md
```

## Data Acquisition

Raw data files are excluded from Git tracking due to file size. Download them into `data/raw/` before executing the notebooks:

1. **Dry Bean Dataset (Part 1: Tabular MLP)**:
   - Source: UCI Machine Learning Repository (ID: 602). Loaded via `ucimlrepo` inside the notebook or downloaded manually from [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
2. **MIT-BIH Arrhythmia Dataset (Part 2: ECG Sequence Modeling)**:
   - Source: Kaggle (`shayanfazeli/heartbeat`). Place `mitbih_train.csv` and `mitbih_test.csv` into `data/raw/`.

## Environment Setup

- **Python**: 3.12+
- **Framework**: PyTorch

### Option A: Local CPU (Windows / macOS / Linux)

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: Local NVIDIA GPU (CUDA 12.4)

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Option C: Google Colab

Open any `.ipynb` file in Colab and enable `Runtime -> Change runtime type -> T4 GPU`. PyTorch and core scientific packages are pre-installed; run `pip install ucimlrepo` if executing Part 1 on Colab.

## Reproducibility Protocol

- **Random Seed**: Fixed at `SEED = 42` across `random`, `numpy`, and `torch`.
- **Data Splitting**:
  - *Dry Bean*: Stratified `70% Train / 15% Validation / 15% Test` after duplicate removal.
  - *MIT-BIH ECG*: Official `mitbih_test.csv` is strictly reserved as the final test set; `mitbih_train.csv` is split via Stratified `80% Train / 20% Validation`.
- **Evaluation Rule**: Hyperparameter tuning and model selection are performed exclusively on the Validation set. The Test set is evaluated once for final reporting.

## Execution Order

1. `01-mlp-optimization-dry-bean/mlp_optimization_study.ipynb` (runs independently)
2. `02-ecg-arrhythmia-cnn-lstm/01_eda_and_baselines.ipynb` (generates shared splits and baseline benchmarks)
3. `02-ecg-arrhythmia-cnn-lstm/02_hybrid_cnn_bilstm_analysis.ipynb` (trains improved architectures and performs final test evaluation and error analysis)

## References

- Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.
- Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG Heartbeat Classification: A Deep Transferable Representation. *IEEE International Conference on Healthcare Informatics (ICHI)*.

## Contributors

- **Do Quoc Trung**: MLP Backpropagation theory, optimizer and regularization experiments, project integration.
- **Le Hoang Tri**: ECG exploratory analysis, traditional ML baselines, and base CNN/LSTM configurations.
- **Nguyen Quoc Huy**: Hybrid CNN-BiLSTM improvements, ablation study, final test evaluation, and error analysis.