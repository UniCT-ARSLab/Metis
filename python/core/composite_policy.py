"""Self-contained composite policy: a frozen base actor + a distance-gated bounded residual, fused
into ONE Keras model that emits the final action. Loads via tf.keras.models.load_model with NO custom
code at the call site (the gate/combine layer is a registered Metis serializable), so run.py never
rebuilds the residual or knows anything task-specific.

    action = clip(base(obs) + gate(||obs[err]||) * delta_max * tanh(residual(obs)), low, high)
    gate(d) = clip((outer - d) / (outer - inner), 0, 1)          # 0 outside near-target

The same fusion turns a residual-PPO checkpoint into a standard, deployable policy.keras (see
build_residual_export_policy): the effective action equals the runtime ResidualActionAdapter, so the
generic evaluation/best-checkpoint machinery and run.py can consume it with no residual awareness. The
action bounds are carried in the layer (NOT hardcoded to [-1,1]) so parity holds for any continuous task.
"""
import keras
import tensorflow as tf


def _as_jsonable_bound(b):
    """Normalize an action bound to a JSON-serializable scalar or list (for get_config round-trip)."""
    if hasattr(b, "tolist"):          # numpy array / scalar
        return b.tolist()
    if isinstance(b, (list, tuple)):
        return [float(x) for x in b]
    return float(b)


@keras.saving.register_keras_serializable(package="metis")
class BoundedGateCombine(keras.layers.Layer):
    """inputs = [obs, base_out, residual_out] -> clip(base + gate(d)*delta*tanh(residual), low, high).
    d = ||obs[err_start : err_start+err_size]|| (the target position-error slice of the observation).
    action_low/action_high are per-component bounds (scalars broadcast, or length-action_size lists);
    clip_base=True first clips base_out to [low, high] (matches ResidualActionAdapter.base_action, which
    clips the base action per-component before adding the gated correction)."""

    def __init__(self, delta_max, gate_outer, gate_inner, err_start=14, err_size=3, clip_base=False,
                 action_low=-1.0, action_high=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta_max = float(delta_max)
        self.gate_outer = float(gate_outer)
        self.gate_inner = float(gate_inner)
        self.err_start = int(err_start)
        self.err_size = int(err_size)
        self.clip_base = bool(clip_base)
        # Kept as plain Python (JSON-serializable in get_config); materialized to tensors in build().
        self.action_low = _as_jsonable_bound(action_low)
        self.action_high = _as_jsonable_bound(action_high)

    def build(self, input_shape):
        self._low = tf.constant(self.action_low, dtype=tf.float32)
        self._high = tf.constant(self.action_high, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        obs, base_out, residual_out = inputs
        if self.clip_base:
            base_out = tf.clip_by_value(base_out, self._low, self._high)
        err = obs[:, self.err_start:self.err_start + self.err_size]
        d = tf.norm(err, axis=-1, keepdims=True)
        g = tf.clip_by_value((self.gate_outer - d) / (self.gate_outer - self.gate_inner), 0.0, 1.0)
        corr = g * self.delta_max * tf.tanh(residual_out)
        return tf.clip_by_value(base_out + corr, self._low, self._high)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(delta_max=self.delta_max, gate_outer=self.gate_outer, gate_inner=self.gate_inner,
                   err_start=self.err_start, err_size=self.err_size, clip_base=self.clip_base,
                   action_low=self.action_low, action_high=self.action_high)
        return cfg


def build_composite_policy(base_model, residual_model, *, delta_max, gate_outer, gate_inner,
                           err_start=14, err_size=3, clip_base=False, action_low=-1.0, action_high=1.0,
                           freeze=True, name="composite_policy"):
    """Fuse a frozen base + a residual into one action-emitting Keras model.

    freeze=True (default) marks both sub-models non-trainable -- correct for an offline packaging tool
    that owns fresh sub-models. Pass freeze=False when a sub-model SHARES layers with a live trainable
    network (e.g. exporting the current residual-PPO actor): mutating `.trainable` would then freeze the
    live network. Saving serializes the current weights regardless of the trainable flag.
    action_low/action_high are the env action bounds the final (and base) clip use."""
    if freeze:
        base_model.trainable = False
        residual_model.trainable = False
    obs_dim = int(base_model.input_shape[-1])
    inp = keras.Input(shape=(obs_dim,), name="obs")
    action = BoundedGateCombine(delta_max, gate_outer, gate_inner, err_start=err_start, err_size=err_size,
                                clip_base=clip_base, action_low=action_low, action_high=action_high,
                                name="gate_combine")([inp, base_model(inp), residual_model(inp)])
    return keras.Model(inp, action, name=name)


def build_residual_export_policy(base_model, actor_model, *, delta_max, gate_outer, gate_inner,
                                 err_start, err_size, action_low=-1.0, action_high=1.0,
                                 name="residual_effective_policy"):
    """Fuse the frozen base with the residual-PPO actor's `continuous_mean` head into ONE runnable
    policy.keras whose effective action == the runtime ResidualActionAdapter, for ANY action bounds.

    The output is the SAME multi-head shape run.py's PPO continuous decode expects -- a list
    [effective_action, value] -- so run.py takes output[-2] and CLIPS it to the env bounds (an identity,
    since the fused action is already inside them). Emitting a single tensor would instead route run.py's
    continuous path through scale_action_numpy (a [-1,1]->bounds remap), which double-transforms for any
    bounds != [-1, 1]. The residual/value heads are VIEWS over the live actor (shared weights, reflect the
    current policy); freeze=False keeps the live actor trainable; clip_base matches the adapter's base clip."""
    obs_dim = int(base_model.input_shape[-1])
    inp = keras.Input(shape=(obs_dim,), name="obs")
    latent = keras.Model(actor_model.inputs, actor_model.get_layer("continuous_mean").output,
                         name="residual_latent")
    value_view = keras.Model(actor_model.inputs, actor_model.outputs[-1], name="residual_value")
    action = BoundedGateCombine(delta_max, gate_outer, gate_inner, err_start=err_start, err_size=err_size,
                                clip_base=True, action_low=action_low, action_high=action_high,
                                name="gate_combine")([inp, base_model(inp), latent(inp)])
    return keras.Model(inp, [action, value_view(inp)], name=name)
