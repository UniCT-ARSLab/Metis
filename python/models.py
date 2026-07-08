from tensorflow import keras


def build_shared_q_network(obs_dim, num_actions):
    inputs = keras.Input(shape=(obs_dim,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_actions)(x)
    return keras.Model(inputs, outputs)


def build_continuous_actor(obs_dim, action_size):
    inputs = keras.Input(shape=(obs_dim,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(action_size, activation="tanh")(x)
    return keras.Model(inputs, outputs)


def build_continuous_critic(obs_dim, action_size):
    obs_input = keras.Input(shape=(obs_dim,))
    action_input = keras.Input(shape=(action_size,))
    x = keras.layers.Concatenate()([obs_input, action_input])
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model([obs_input, action_input], outputs)


def build_hybrid_actor_critic(obs_dim, discrete_sizes, continuous_size):
    inputs = keras.Input(shape=(obs_dim,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = []
    for idx, size in enumerate(discrete_sizes):
        outputs.append(keras.layers.Dense(int(size), name=f"discrete_{idx}_logits")(x))

    if continuous_size > 0:
        outputs.append(
            keras.layers.Dense(int(continuous_size), activation="tanh", name="continuous_mean")(x)
        )

    outputs.append(keras.layers.Dense(1, name="value")(x))
    return keras.Model(inputs, outputs)
