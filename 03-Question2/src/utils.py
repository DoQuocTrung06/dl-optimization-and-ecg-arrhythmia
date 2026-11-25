import re
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, p):
    return dict(mae=mean_absolute_error(y, p),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=r2_score(y, p))


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
