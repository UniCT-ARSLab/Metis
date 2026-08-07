"""M6.2 Phase C1: gated bounded-settling residual head around the FROZEN clone.

    d_obs   = ||obs[14:17]||                      (target position error in obs-space; ~2*meters)
    gate(d) = clip((outer - d)/(outer - inner), 0, 1)     # 0 for d>=outer (approach), 1 for d<=inner
    action  = clip(clone(obs) + gate(d) * delta_max * tanh(residual(obs)), -1, 1)

gate is EXACTLY 0 outside the near-target zone, so the approach is provably identical to the clone
(|action-clone| == 0 there); the correction magnitude is hard-bounded by delta_max. The residual's
output layer is zero-initialized -> at init the policy IS the clone. Only the residual is trainable.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# obs-space gate boundaries (workspace_scale=0.5 -> obs[14:17] ~= 2 * position_error_m).
GATE_OUTER = 0.20      # ~10 cm: gate starts opening
GATE_INNER = 0.08      # ~4 cm: gate fully open


def build_settling_residual(obs_dim, action_size, hidden=(128, 128)):
    inputs = keras.Input(shape=(obs_dim,))
    x = inputs
    for h in hidden:
        x = keras.layers.Dense(h, activation="relu")(x)
    out = keras.layers.Dense(action_size, activation="linear",
                             kernel_initializer="zeros", bias_initializer="zeros")(x)
    return keras.Model(inputs, out, name="settling_residual")


def d_obs_of(obs):
    """Near-target distance signal in obs-space (batched or single)."""
    o = tf.convert_to_tensor(obs, tf.float32)
    return tf.norm(o[..., 14:17], axis=-1)


def gate(d_obs, *, outer=GATE_OUTER, inner=GATE_INNER):
    return tf.clip_by_value((outer - d_obs) / (outer - inner), 0.0, 1.0)


def settling_correction(residual_model, obs, *, delta_max, outer=GATE_OUTER, inner=GATE_INNER,
                        training=False):
    """gate(d_obs) * delta_max * tanh(residual(obs)) -- the applied bounded correction (>=... = 0 in
    approach). |correction| <= delta_max everywhere by construction."""
    o = tf.convert_to_tensor(obs, tf.float32)
    g = gate(d_obs_of(o), outer=outer, inner=inner)[:, None]
    return g * float(delta_max) * tf.tanh(residual_model(o, training=training))


def settling_action(clone_out, residual_model, obs, *, delta_max, outer=GATE_OUTER, inner=GATE_INNER,
                    training=False):
    """action = clip(clone + correction, -1, 1). `clone_out` is the frozen clone output in [-1,1]."""
    corr = settling_correction(residual_model, obs, delta_max=delta_max, outer=outer, inner=inner,
                               training=training)
    return tf.clip_by_value(tf.convert_to_tensor(clone_out, tf.float32) + corr, -1.0, 1.0), corr
