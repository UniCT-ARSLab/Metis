from collections import OrderedDict

import numpy as np
from gymnasium import spaces


def _component_bound(value, size, default):
    array = np.asarray(
        value if isinstance(value, (list, tuple, np.ndarray)) else [value],
        dtype=np.float32,
    )
    if array.size == 0:
        array = np.asarray([default], dtype=np.float32)
    if array.size == 1:
        return np.full((size,), float(array[0]), dtype=np.float32)
    if array.size != size:
        raise ValueError(f"Action bound has size {array.size}, expected {size}")
    return array.astype(np.float32)


class SB3ActionCodec:
    """Map one Metis agent action to the action space consumed by SB3.

    Discrete and continuous spaces pass through unchanged. A hybrid Metis Dict is
    represented as a Box: continuous values retain their declared bounds, while each
    discrete component receives one latent logit per choice and is decoded with
    argmax. This keeps the bridge contract generic without pretending SB3 natively
    implements a mixed categorical/Gaussian distribution.
    """

    def __init__(self, action_type, native_space, action_space_spec=None):
        self.action_type = str(action_type)
        self.native_space = native_space
        self.action_space_spec = OrderedDict(action_space_spec or {})
        self.components = []

        if self.action_type == "discrete":
            self.policy_space = spaces.Discrete(int(native_space.n))
            self.latent_size = 1
            return
        if self.action_type == "continuous":
            self.policy_space = spaces.Box(
                low=np.asarray(native_space.low, dtype=np.float32),
                high=np.asarray(native_space.high, dtype=np.float32),
                dtype=np.float32,
            )
            self.latent_size = int(np.prod(self.policy_space.shape))
            return
        if self.action_type != "hybrid":
            raise ValueError(f"Unsupported Metis action_type={self.action_type!r}")
        if not self.action_space_spec:
            raise ValueError("A hybrid Metis action requires action_space_spec metadata")

        lows = []
        highs = []
        offset = 0
        for name, raw_component in self.action_space_spec.items():
            component = dict(raw_component)
            component_type = str(
                component.get("action_type", component.get("type", "discrete"))
            )
            size = int(component.get("size", 1))
            if size < 1:
                raise ValueError(f"Action component {name!r} must have size >= 1")

            if component_type == "discrete":
                latent_size = size
                lows.extend([-1.0] * latent_size)
                highs.extend([1.0] * latent_size)
            elif component_type == "continuous":
                latent_size = size
                lows.extend(
                    _component_bound(component.get("low", -1.0), size, -1.0).tolist()
                )
                highs.extend(
                    _component_bound(component.get("high", 1.0), size, 1.0).tolist()
                )
            else:
                raise ValueError(
                    f"Unsupported action component type {component_type!r} for {name!r}"
                )

            self.components.append(
                {
                    "name": str(name),
                    "type": component_type,
                    "size": size,
                    "slice": slice(offset, offset + latent_size),
                }
            )
            offset += latent_size

        self.latent_size = offset
        self.policy_space = spaces.Box(
            low=np.asarray(lows, dtype=np.float32),
            high=np.asarray(highs, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def uses_hybrid_encoding(self):
        return self.action_type == "hybrid"

    def decode(self, policy_action):
        if self.action_type == "discrete":
            return int(np.asarray(policy_action).reshape(()))
        if self.action_type == "continuous":
            action = np.asarray(policy_action, dtype=np.float32).reshape(
                self.policy_space.shape
            )
            return np.clip(action, self.policy_space.low, self.policy_space.high)

        latent = np.asarray(policy_action, dtype=np.float32).reshape(self.latent_size)
        latent = np.clip(latent, self.policy_space.low, self.policy_space.high)
        decoded = {}
        for component in self.components:
            values = latent[component["slice"]]
            if component["type"] == "discrete":
                decoded[component["name"]] = int(np.argmax(values))
            elif component["size"] == 1:
                decoded[component["name"]] = float(values[0])
            else:
                decoded[component["name"]] = values.astype(np.float32).tolist()
        return decoded

    def describe(self):
        if not self.uses_hybrid_encoding:
            return self.action_type
        parts = [
            f"{component['name']}:{component['type']}[{component['size']}]"
            for component in self.components
        ]
        return f"hybrid_box(latent={self.latent_size}, components={parts})"
