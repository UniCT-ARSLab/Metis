from tensorflow import keras


def build_shared_q_network(obs_dim, num_actions):
    inputs = keras.Input(shape=(obs_dim,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_actions)(x)
    return keras.Model(inputs, outputs)
