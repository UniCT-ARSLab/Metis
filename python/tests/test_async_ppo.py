import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

import numpy as np
import tensorflow as tf

from core.models import build_hybrid_actor_critic
from algorithms.ppo import build_action_metadata, run_async_ppo


class FakeHybridEnv:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.agent_ids = ["Agent"]
        self.reset_count = 0
        self.step_count = 0
        self.configs = []

    def configure(self, **config):
        self.configs.append(config)

    def reset(self, seed=None):
        self.reset_count += 1
        self.step_count = 0
        return np.zeros((self.obs_dim,), dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.assert_valid_action(action)
        self.step_count += 1
        obs = np.full((self.obs_dim,), self.step_count / 10.0, dtype=np.float32)
        terminated = self.step_count >= 2
        return obs, 1.0, terminated, False, {"agent_info": {}}

    @staticmethod
    def assert_valid_action(action):
        if set(action) != {"fire", "drive"}:
            raise AssertionError(f"Unexpected hybrid action: {action}")


class AsyncPPOTests(unittest.TestCase):
    def test_two_workers_train_versioned_on_policy_generations(self):
        obs_dim = 3
        action_meta = build_action_metadata({
            "fire": {"action_type": "discrete", "size": 2},
            "drive": {"action_type": "continuous", "size": 1, "low": -1.0, "high": 1.0},
        })
        model = build_hybrid_actor_critic(
            obs_dim=obs_dim,
            discrete_sizes=action_meta["discrete_sizes"],
            continuous_size=action_meta["continuous_size"],
        )
        log_std = tf.Variable(np.asarray([-0.5], dtype=np.float32), trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        checkpoint = tf.train.Checkpoint(
            model=model,
            log_std=log_std,
            optimizer=optimizer,
            episode=tf.Variable(0, dtype=tf.int64),
        )
        envs = [FakeHybridEnv(obs_dim), FakeHybridEnv(obs_dim)]
        args = SimpleNamespace(
            collector_mode="async",
            opponent_pool=False,
            async_queue_capacity=16,
            async_policy_sync_steps=10,
            async_policy_publish_updates=10,
            async_updates_per_step=1,
            async_update_basis="transitions",
            async_update_every=4,
            async_max_updates_per_env_step=1,
            async_drain_max_events=64,
            async_replay_save=True,
            num_episodes=2,
            max_steps_per_episode=2,
            physics_frames_per_step=1,
            best_final_drain_timeout=0.0,
            episode_seed_multiplier=1000,
            multi_agent=False,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_epochs=1,
            batch_size=4,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            checkpoint_every=0,
            log_format="compact",
        )

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)
            completed = run_async_ppo(
                args,
                envs,
                model,
                log_std,
                optimizer,
                action_meta,
                obs_dim,
                checkpoint,
                manager,
                # The loop polls the tracker every iteration, so the fake needs the
                # whole surface apply_ready_best_checkpoint touches, not just
                # should_evaluate.
                SimpleNamespace(
                    should_evaluate=lambda _episode: False,
                    poll_ready=lambda timeout=0.0: None,
                    enabled=False,
                ),
                start_episode=0,
            )

        self.assertEqual(completed, 2)
        self.assertEqual(int(checkpoint.episode.numpy()), 2)
        self.assertEqual([env.reset_count for env in envs], [2, 2])
        self.assertTrue(all(config["max_steps"] == 2 for env in envs for config in env.configs))


if __name__ == "__main__":
    unittest.main()
