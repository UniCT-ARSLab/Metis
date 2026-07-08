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
