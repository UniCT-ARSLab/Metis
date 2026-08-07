"""Zero-initialized RESIDUAL actor for the M6.1 canary (approved spec).

The base BC clone actor is FROZEN. A separate small residual net produces a bounded correction in the
NORMALIZED [-1, 1] joint-velocity action space (Godot scales [-1,1] -> +-0.5 rad/s):

    delta  = DELTA_SCALE * tanh(residual(obs))            # DELTA_SCALE = 0.05 (normalized, NOT rad)
    action = clip(bc_action(obs) + delta, -1, 1)

The residual's OUTPUT layer is zero-initialized, so at update 0 delta == 0 and action == bc_action
EXACTLY (a preflight gate checks this). The clip can only shrink |action - bc|, never grow it, so the
hard per-batch bound max|action - bc| <= DELTA_SCALE (+ tiny epsilon) holds by construction; a runtime
assertion enforces it anyway. The critic v6 accepts actions in [-1,1] (action_low/high = -/+1), so the
normalized action feeds the frozen critic directly.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

DELTA_SCALE = 0.05
DEVIATION_TOLERANCE = 0.050001    # hard per-batch bound on max|action - bc_action|


def build_residual_actor(obs_dim, action_size, hidden=(256, 256)):
    """Small MLP whose OUTPUT layer is zero-initialized -> residual(obs) == 0 at init."""
    inputs = keras.Input(shape=(obs_dim,))
    x = inputs
    for h in hidden:
        x = keras.layers.Dense(h, activation="relu")(x)
    outputs = keras.layers.Dense(action_size, activation="linear",
                                 kernel_initializer="zeros", bias_initializer="zeros")(x)
    return keras.Model(inputs, outputs, name="residual_actor")


def residual_delta(residual_model, obs, *, delta_scale=DELTA_SCALE, training=False):
    """delta = delta_scale * tanh(residual(obs)) in [-delta_scale, delta_scale]."""
    return delta_scale * tf.tanh(residual_model(obs, training=training))


def combined_action(bc_action, residual_model, obs, *, delta_scale=DELTA_SCALE, training=False):
    """action = clip(bc_action(obs) + delta, -1, 1). `bc_action` is the FROZEN BC actor's output in
    [-1,1]. Returns (action, delta)."""
    bc = bc_action(obs)
    delta = residual_delta(residual_model, obs, delta_scale=delta_scale, training=training)
    action = tf.clip_by_value(bc + delta, -1.0, 1.0)
    return action, delta


def max_action_deviation(action, bc):
    """max_ij |action - bc| as a python float (the hard per-batch safety bound)."""
    return float(tf.reduce_max(tf.abs(action - bc)).numpy())


def assert_within_bound(action, bc, *, tol=DEVIATION_TOLERANCE):
    """Hard guard: the residual may never move the action more than DELTA_SCALE from the clone."""
    dev = max_action_deviation(action, bc)
    if not (dev <= tol):
        raise AssertionError(f"residual deviation {dev} exceeds bound {tol} (canary safety violation)")
    return dev


def residual_is_zero(residual_model, obs, *, atol=0.0):
    """True iff residual(obs) is exactly zero (preflight gate at update 0)."""
    r = np.asarray(residual_model(obs, training=False))
    return bool(np.all(np.abs(r) <= atol))


ANCHOR_COEF = 10.0


def canary_actor_loss(residual_model, bc_action_fn, critic_min_q_fn, obs, *, q_weight,
                      anchor_coef=ANCHOR_COEF, delta_scale=DELTA_SCALE):
    """Canary objective (approved):  L = -q_weight * mean(min(Q1,Q2)(s, action)) + 10 * mean(delta^2).

    `bc_action_fn(obs)` = FROZEN BC clone output in [-1,1]; `critic_min_q_fn(obs, action)` = FROZEN twin
    critics' min-Q. Only `residual_model` is trainable -- callers apply gradients to it ALONE, so the
    base actor and critics never change (their weights are not passed to apply_gradients). Returns
    (loss, delta, action, telemetry)."""
    obs_t = tf.convert_to_tensor(obs, tf.float32)
    bc = tf.stop_gradient(bc_action_fn(obs_t))            # base actor frozen (defensive stop_gradient)
    delta = residual_delta(residual_model, obs_t, delta_scale=delta_scale, training=True)
    action = tf.clip_by_value(bc + delta, -1.0, 1.0)
    q = critic_min_q_fn(obs_t, action)                    # frozen critics; grad -> residual via action
    q_mean = tf.reduce_mean(q)
    anchor = anchor_coef * tf.reduce_mean(tf.square(delta))
    loss = -q_weight * q_mean + anchor
    tel = {"q_mean": float(q_mean.numpy()), "anchor_loss": float(anchor.numpy()),
           "delta_sq_mean": float(tf.reduce_mean(tf.square(delta)).numpy())}
    return loss, delta, action, tel
