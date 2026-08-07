import tensorflow as tf
from tensorflow import keras


def build_greedy_action_fn(model, obs_dim, device="/CPU:0"):
    """Compile a batched argmax-Q call for an async collector thread.

    Collectors call this once per env step, which makes dispatch overhead the whole
    cost: at obs_dim=5 the arithmetic is ~70k FLOPs. Two things make the eager call
    expensive, and both are fixed here.

    `input_signature` fixes the shape so the graph is traced once instead of being
    re-examined per call, and it folds the argmax into the graph. Batch is dynamic so
    one graph serves both single-agent (batch 1) and multi-agent (batch N) collectors.

    The device scope pins the *ops* to where the collector's weights already live.
    Without it TF's soft placement runs the matmuls on the GPU and copies the
    CPU-resident weights across on every single call -- measured at 3.4ms/call
    versus 0.6ms once pinned. It also matters for correctness under XLA, which
    errors outright on the CPU-variable/GPU-op split rather than paying for it.

    Eager dispatch holds the GIL and is not thread-safe under concurrent collectors,
    so this is what lets collector threads overlap safely: measured aggregate
    throughput at 4 threads went from 279 to 2183 calls/s.
    """
    @tf.function(input_signature=[tf.TensorSpec([None, obs_dim], tf.float32)])
    def greedy_action(obs_batch):
        with tf.device(device):
            q_values = model(obs_batch, training=False)
            return tf.argmax(q_values, axis=1, output_type=tf.int32)

    return greedy_action


def build_actor_forward_fn(model, obs_dim, device="/CPU:0"):
    """Trace a deterministic actor forward for an async collector thread.

    Same thread-safety reason as build_greedy_action_fn: an eager `actor(obs)` called
    from several collector threads corrupts the output under concurrency. DDPG/TD3 add
    exploration noise in numpy after this, so only the forward pass needs tracing.
    Batch is dynamic so one graph serves both single- and multi-agent collectors.
    """
    @tf.function(input_signature=[tf.TensorSpec([None, obs_dim], tf.float32)])
    def actor_forward(obs_batch):
        with tf.device(device):
            return model(obs_batch, training=False)

    return actor_forward


## Hidden-layer widths for every network built here. Kept as process-wide state set once at
## startup rather than threaded through ~20 construction sites, with an explicit per-call override
## still available.
##
## Defaults follow each algorithm's reference architecture, so a run with no --network-layers
## reproduces the published setup instead of a Metis-specific one:
##
##   SAC          256, 256    Haarnoja et al. 2018
##   TD3 / DDPG   400, 300    Fujimoto et al. 2018; Lillicrap et al. 2015
##   PPO           64,  64    Schulman et al. 2017 (MuJoCo MLP)
##   DQN           64,  64    no MLP architecture in Mnih et al. 2015 (it is convolutional);
##                            this is the common vector-observation choice, matching SB3
##
## LEGACY is what every Metis checkpoint built before this change used. Pass
## `--network-layers 256 256 128` to resume, evaluate or export one of those from a TF
## checkpoint; `policy.keras` bundles carry their own architecture and load either way.
LEGACY_NETWORK_LAYERS = (256, 256, 128)
PAPER_NETWORK_LAYERS = {
    "dqn": (64, 64),
    "ppo": (64, 64),
    "ddpg": (400, 300),
    "ddpg_bc": (400, 300),
    "ddpgfd": (400, 300),
    "td3": (400, 300),
    "td3_bc": (400, 300),
    "sac": (256, 256),
}
DEFAULT_NETWORK_LAYERS = LEGACY_NETWORK_LAYERS
_network_layers = DEFAULT_NETWORK_LAYERS


def default_network_layers(algorithm):
    """Reference widths for `algorithm`, falling back to the legacy stack for unknown names."""
    return PAPER_NETWORK_LAYERS.get(str(algorithm), LEGACY_NETWORK_LAYERS)


def set_network_layers(layers):
    """Set the process-wide hidden-layer widths. Call once, before any network is built."""
    global _network_layers
    if not layers:
        return
    widths = tuple(int(width) for width in layers)
    if any(width <= 0 for width in widths):
        raise ValueError(f"network layer widths must be positive, got {layers!r}")
    _network_layers = widths


def network_layers():
    return _network_layers


def _mlp(x, layers=None, name_prefix=None):
    """Stack the configured hidden layers. Unnamed by default so historical checkpoints, whose
    layer names Keras auto-generated, still match."""
    for index, width in enumerate(layers or _network_layers):
        if name_prefix:
            x = keras.layers.Dense(width, activation="relu", name=f"{name_prefix}_d{index}")(x)
        else:
            x = keras.layers.Dense(width, activation="relu")(x)
    return x


def build_shared_q_network(obs_dim, num_actions, layers=None):
    inputs = keras.Input(shape=(obs_dim,))
    x = _mlp(inputs, layers)
    outputs = keras.layers.Dense(num_actions)(x)
    return keras.Model(inputs, outputs)


def build_continuous_actor(obs_dim, action_size, layers=None):
    inputs = keras.Input(shape=(obs_dim,))
    x = _mlp(inputs, layers)
    outputs = keras.layers.Dense(action_size, activation="tanh")(x)
    return keras.Model(inputs, outputs)


def build_continuous_critic(obs_dim, action_size, layers=None):
    obs_input = keras.Input(shape=(obs_dim,))
    action_input = keras.Input(shape=(action_size,))
    x = keras.layers.Concatenate()([obs_input, action_input])
    x = _mlp(x, layers)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model([obs_input, action_input], outputs)


def build_sac_actor(obs_dim, action_size, layers=None):
    inputs = keras.Input(shape=(obs_dim,))
    x = _mlp(inputs, layers)
    mean = keras.layers.Dense(action_size, name="mean")(x)
    log_std = keras.layers.Dense(action_size, name="log_std")(x)
    return keras.Model(inputs, [mean, log_std])


def build_hybrid_actor_critic(obs_dim, discrete_sizes, continuous_size, *,
                              continuous_activation="tanh", separate_value_tower=False,
                              layers=None):
    """Actor-critic. Defaults (tanh continuous head, SHARED actor/value trunk) are unchanged.

    continuous_activation: the continuous-mean head activation. "tanh" bounds it to [-1,1] (standard);
        "linear" leaves it unbounded -- needed for a RESIDUAL policy where a later tanh squash is applied
        by the action adapter (a tanh head would double-squash and cap the residual at ~0.762*delta_max).
    separate_value_tower: when True the value head gets its OWN trunk, so the value loss cannot flow into
        the actor trunk / policy (required isolation for a protected residual policy). Default False keeps
        the shared trunk (unchanged behaviour + parameter count).
    """
    inputs = keras.Input(shape=(obs_dim,))

    def _trunk(src, prefix=None):
        # Default (prefix=None) keeps the original layer names, so existing standard checkpoints still
        # load. The separate value tower is named "value_*" so a caller can split actor/value variables.
        return _mlp(src, layers, name_prefix=prefix)

    x = _trunk(inputs)                                          # actor (and, by default, value) trunk

    outputs = []
    for idx, size in enumerate(discrete_sizes):
        outputs.append(keras.layers.Dense(int(size), name=f"discrete_{idx}_logits")(x))

    if continuous_size > 0:
        outputs.append(
            keras.layers.Dense(int(continuous_size), activation=continuous_activation, name="continuous_mean")(x)
        )

    value_in = _trunk(inputs, "value") if separate_value_tower else x   # isolated, name-tagged value trunk
    outputs.append(keras.layers.Dense(1, name="value")(value_in))
    return keras.Model(inputs, outputs)
