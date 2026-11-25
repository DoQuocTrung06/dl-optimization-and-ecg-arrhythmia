import tensorflow as tf
from tensorflow.keras import layers, models


def _lstm(units, seq, bidir):
    l = layers.LSTM(units, return_sequences=seq)
    return layers.Bidirectional(l) if bidir else l


def _attention(x):
    s = layers.Dense(1, activation="tanh")(x)
    w = layers.Softmax(axis=1)(s)
    c = layers.Dot(axes=1)([w, x])
    return layers.Flatten()(c)


def build_model(arch, n_steps, n_feat, bidir=False, attn=False, dropout=0.0, loss="mse", lr=1e-3):
    inp = layers.Input((n_steps, n_feat))
    x = inp
    if arch == "lstm":
        x = _lstm(64, attn, bidir)(x)
        if attn:
            x = _attention(x)
    elif arch == "cnn":
        x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
    elif arch == "cnn_lstm":
        x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = _lstm(64, attn, bidir)(x)
        if attn:
            x = _attention(x)
    else:
        raise ValueError(arch)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    loss_fn = tf.keras.losses.Huber(1.0) if loss == "huber" else "mse"
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss_fn)
    return m
