"""Checkpointable actor averaging for evaluation, without changing collection policy."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.policy_artifact import PolicyArtifactSaver


def add_actor_ema_arguments(group):
    group.add_argument("--actor-ema-decay", type=float, default=0.0, help="Evaluation/export actor EMA decay in (0, 1); 0 disables it. Raw actor, optimizer and EMA state remain in training checkpoints. Single-policy SAC/DDPG/TD3-family only.")


def validate_actor_ema_options(args):
    decay = float(getattr(args, "actor_ema_decay", 0.0))
    if not np.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("--actor-ema-decay must be finite and in [0, 1)")
    if decay and getattr(args, "multi_policy", False):
        raise ValueError("Actor EMA currently supports single-policy training only")


def validate_actor_ema_checkpoint(args, checkpoint_path):
    if checkpoint_path and not getattr(args, "actor_ema_decay", 0.0):
        if any(name.startswith("actor_ema/") for name, _shape in tf.train.list_variables(str(checkpoint_path))):
            raise ValueError("Resuming an actor-EMA checkpoint requires its --actor-ema-decay; use a new warm-start run to remove EMA")


class ActorEMA(tf.Module):
    def __init__(self, actor, decay):
        super().__init__(name="actor_ema")
        if not 0 < decay < 1:
            raise ValueError("Actor EMA decay must be in (0, 1)")
        self._actor = self._no_dependency(actor)
        self._export_model = self._no_dependency(tf.keras.models.clone_model(actor))
        self.configured_decay = float(decay)
        self.decay = tf.Variable(decay, dtype=tf.float64, trainable=False)
        self.initialized = tf.Variable(False, trainable=False)
        self.updates = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.shadow = [tf.Variable(tf.convert_to_tensor(weight), trainable=False, name=f"weight_{index}") for index, weight in enumerate(actor.weights)]

    def reset(self):
        for shadow, weight in zip(self.shadow, self._actor.weights):
            shadow.assign(weight)
        self.updates.assign(0)
        self.initialized.assign(True)

    def prepare(self, reset=False):
        if abs(float(self.decay.numpy()) - self.configured_decay) > 1e-12:
            raise ValueError("Checkpoint actor EMA decay differs from --actor-ema-decay; use a new warm-start run to change it")
        if reset or not bool(self.initialized.numpy()):
            self.reset()

    def update(self):
        tf.debugging.assert_equal(self.initialized, True, message="Prepare actor EMA after restore/BC and before learning")
        for shadow, weight in zip(self.shadow, self._actor.weights):
            if shadow.dtype.is_floating:
                decay = tf.cast(self.decay, shadow.dtype)
                shadow.assign(decay * shadow + (1 - decay) * tf.cast(weight, shadow.dtype))
            else:
                shadow.assign(weight)
        self.updates.assign_add(1)

    def after_restore(self, checkpoint_path):
        if any(name.startswith("actor_ema/") for name, _shape in tf.train.list_variables(str(checkpoint_path))):
            self.prepare()
        else:
            self.reset()

    def export_model(self):
        self._export_model.set_weights([weight.numpy() for weight in self.shadow])
        return self._export_model


class EMAPolicyArtifactSaver:
    def __init__(self, ema, directory, metadata):
        self.ema = ema
        self.directory = directory
        self.metadata = {**metadata, "inference": {**metadata.get("inference", {}), "policy_weights": "ema"}}

    def save(self, episode):
        return PolicyArtifactSaver(self.ema.export_model(), self.directory, self.metadata).save(episode)

    def export(self, destination):
        self.ema.export_model().save(str(destination))
        Path(destination).with_suffix(".json").write_text(json.dumps(self.metadata, indent=2) + "\n")


def configure_actor_ema(args, checkpoint, tracker, metadata, *, reset=False):
    ema = getattr(checkpoint, "actor_ema", None)
    if ema is None:
        return
    ema.prepare(reset=reset)
    args._actor_ema = ema
    saver = EMAPolicyArtifactSaver(ema, args.checkpoint_dir, metadata)
    args.policy_artifact = saver
    tracker.set_policy_artifact_exporter(saver.export, retain_training_checkpoint=True)
    print(f"Actor EMA enabled: decay={ema.configured_decay:g}; frozen evaluations/exports use EMA, collectors use raw actor", flush=True)
