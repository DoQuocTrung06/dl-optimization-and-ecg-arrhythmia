import numpy as np
from .config import FEATURES


def make_features(X):
    ot = X[:, :, 6]
    f, n = [], []
    for i, c in enumerate(FEATURES):
        f.append(X[:, -1, i]); n.append(f"{c}_last")
    for k in (1, 4, 12, 24, 48, 95):
        f.append(ot[:, -1 - k]); n.append(f"OT_lag{k}")
    for w in (4, 24, 96):
        f.append(ot[:, -w:].mean(1)); n.append(f"OT_mean{w}")
        f.append(ot[:, -w:].std(1)); n.append(f"OT_std{w}")
    for w in (24, 96):
        for i, c in enumerate(FEATURES[:6]):
            f.append(X[:, -w:, i].mean(1)); n.append(f"{c}_mean{w}")
    f.append(ot.max(1)); n.append("OT_max96")
    f.append(ot.min(1)); n.append("OT_min96")
    return np.stack(f, 1), n
