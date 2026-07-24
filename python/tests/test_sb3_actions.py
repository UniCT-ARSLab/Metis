import unittest

import numpy as np
from gymnasium import spaces

from backends.sb3_actions import SB3ActionCodec


class SB3ActionCodecTests(unittest.TestCase):
    def test_hybrid_box_decodes_discrete_logits_and_continuous_values(self):
        codec = SB3ActionCodec(
            "hybrid",
            spaces.Dict({}),
            {
                "movement": {
                    "action_type": "continuous",
                    "size": 2,
                    "low": [-1.0, -0.5],
                    "high": [1.0, 0.5],
                },
                "weapon": {
                    "action_type": "discrete",
                    "size": 3,
                },
            },
        )

        decoded = codec.decode(np.asarray([0.75, -0.25, -0.5, 0.9, 0.1]))

        self.assertEqual(codec.policy_space.shape, (5,))
        self.assertEqual(decoded["movement"], [0.75, -0.25])
        self.assertEqual(decoded["weapon"], 1)

    def test_hybrid_codec_clips_before_decoding(self):
        codec = SB3ActionCodec(
            "hybrid",
            spaces.Dict({}),
            {
                "throttle": {
                    "action_type": "continuous",
                    "size": 1,
                    "low": 0.0,
                    "high": 1.0,
                },
                "fire": {
                    "action_type": "discrete",
                    "size": 2,
                },
            },
        )

        decoded = codec.decode(np.asarray([4.0, 3.0, -4.0]))

        self.assertEqual(decoded, {"throttle": 1.0, "fire": 0})

    def test_plain_spaces_keep_their_native_shape(self):
        discrete = SB3ActionCodec("discrete", spaces.Discrete(4))
        continuous = SB3ActionCodec(
            "continuous",
            spaces.Box(
                low=np.asarray([-2.0, -1.0], dtype=np.float32),
                high=np.asarray([2.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        )

        self.assertEqual(discrete.decode(np.int64(3)), 3)
        np.testing.assert_allclose(
            continuous.decode(np.asarray([3.0, -2.0])),
            np.asarray([2.0, -1.0]),
        )


if __name__ == "__main__":
    unittest.main()
