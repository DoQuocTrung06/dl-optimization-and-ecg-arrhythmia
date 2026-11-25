import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "ETTm1.arff"
RES = ROOT / "results"
FIG, TAB, PRED, HIST = RES / "figures", RES / "tables", RES / "preds", RES / "history"
for d in (FIG, TAB, PRED, HIST):
    d.mkdir(parents=True, exist_ok=True)

FAST = os.environ.get("FAST", "0") == "1"

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_IDX = 6

WINDOW = 96
HORIZON = 96
TRAIN_END, VAL_END, TEST_END = 34560, 46080, 57600

SEEDS = [0] if FAST else [0, 1, 2]
EPOCHS = 3 if FAST else 40
BATCH = 256
PATIENCE = 6
