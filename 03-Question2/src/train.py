import json
import time

import numpy as np
import pandas as pd

from .config import *
from .data import get_data
from .utils import metrics, slug


def _fit_predict(arch, d, ytr, yva, seed, opts, epochs, evaluate_test):
    import tensorflow as tf
    from .models import build_model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    Xtr = d["train"]["X"]
    model = build_model(arch, Xtr.shape[1], Xtr.shape[2], **opts)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    t0 = time.time()
    h = model.fit(Xtr, ytr, validation_data=(d["val"]["X"], yva),
                  epochs=epochs, batch_size=BATCH, callbacks=cb, verbose=0)
    t = time.time() - t0
    pv = model.predict(d["val"]["X"], batch_size=1024, verbose=0).ravel()
    pt = (model.predict(d["test"]["X"], batch_size=1024, verbose=0).ravel()
          if evaluate_test else None)
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    return pv, pt, hist, model.count_params(), t


def run_config(name, arch, window=WINDOW, time_feats=False, residual=False, opts=None,
               seeds=SEEDS, epochs=EPOCHS, evaluate_test=True):
    """Train one configuration; disable test evaluation while tuning."""
    opts = opts or {}
    d = get_data(window, time_feats)
    mu, sd = d["mu"], d["sd"]
    yv = d["val"]["y"] * sd + mu
    yt = d["test"]["y"] * sd + mu if evaluate_test else None
    bv = d["val"]["base"] if residual else 0.0
    bt = d["test"]["base"] if residual and evaluate_test else 0.0
    ytr = d["train"]["y"] - (d["train"]["base"] if residual else 0.0)
    yva = d["val"]["y"] - bv
    rows = []
    for s in seeds:
        pv, pt, hist, npar, t = _fit_predict(
            arch, d, ytr, yva, s, opts, epochs, evaluate_test)
        pv = (pv + bv) * sd + mu
        mv = metrics(yv, pv)
        mt = None
        if evaluate_test:
            pt = (pt + bt) * sd + mu
            mt = metrics(yt, pt)
        rows.append(dict(
            name=name, seed=s, val_rmse=mv["rmse"],
            test_mae=mt["mae"] if mt else np.nan,
            test_rmse=mt["rmse"] if mt else np.nan,
            test_r2=mt["r2"] if mt else np.nan,
            time_s=t, epochs=len(hist["loss"]), params=npar))
        msg = f"[{name}] seed={s} val_rmse={mv['rmse']:.3f}"
        if mt:
            msg += (f" test_rmse={mt['rmse']:.3f}"
                    f" test_mae={mt['mae']:.3f} r2={mt['r2']:.3f}")
        print(f"{msg} ({t:.0f}s)", flush=True)
        if s == seeds[0]:
            if evaluate_test:
                np.savez(PRED / f"{slug(name)}.npz", y_true=yt, y_pred=pt,
                         idx=d["test"]["idx"], seed=s)
            (HIST / f"{slug(name)}.json").write_text(
                json.dumps(hist), encoding="utf-8")
        if evaluate_test:
            np.savez(PRED / f"{slug(name)}_seed{s}.npz", y_true=yt, y_pred=pt,
                     idx=d["test"]["idx"], seed=s)
    df = pd.DataFrame(rows)
    summ = dict(name=name, arch=arch, window=window, time_feats=time_feats, residual=residual, opts=json.dumps(opts),
                params=npar, val_rmse=df.val_rmse.mean(),
                val_rmse_std=df.val_rmse.std(ddof=0),
                test_mae=df.test_mae.mean(), test_mae_std=df.test_mae.std(ddof=0),
                test_rmse=df.test_rmse.mean(), test_rmse_std=df.test_rmse.std(ddof=0),
                test_r2=df.test_r2.mean(), test_r2_std=df.test_r2.std(ddof=0),
                time_s=df.time_s.mean(), epochs=df.epochs.mean())
    return summ, df
