"""Supervised expert data for PPO, kept separate from on-policy rollouts."""

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf


def validate_options(args):
    sources = bool(args.demo_path or args.demo_bc_path)
    for name in ("demo_bc_epochs", "demo_max_transitions", "ppo_demo_decay_updates"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.demo_bc_batch_size <= 0:
        raise ValueError("--demo-bc-batch-size must be positive")
    for name in ("ppo_demo_coef", "ppo_demo_final_coef"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.demo_bc_learning_rate is not None and (not np.isfinite(args.demo_bc_learning_rate) or args.demo_bc_learning_rate <= 0):
        raise ValueError("--demo-bc-learning-rate must be finite and positive")
    if args.ppo_demo_final_coef > args.ppo_demo_coef:
        raise ValueError("--ppo-demo-final-coef cannot exceed --ppo-demo-coef")
    if args.ppo_demo_final_coef and not args.ppo_demo_decay_updates:
        raise ValueError("--ppo-demo-final-coef requires --ppo-demo-decay-updates")
    if (args.demo_bc_epochs > 0 or args.ppo_demo_coef > 0) and not sources:
        raise ValueError("PPO imitation requires --demo-path or --demo-bc-path")
    if sources and not (args.demo_bc_epochs > 0 or args.ppo_demo_coef > 0):
        raise ValueError("Enable --demo-bc-epochs or --ppo-demo-coef to use PPO demonstrations")
    if args.demo_validation_path and not (sources and args.demo_bc_epochs > 0):
        raise ValueError("--demo-validation-path requires PPO behavior cloning")
    if args.stop_after_bc and (not sources or args.demo_bc_epochs <= 0):
        raise ValueError("--stop-after-bc requires demonstration sources and --demo-bc-epochs > 0")
    if sources and (getattr(args, "multi_policy", False) or getattr(args, "policy_mode", "standard") != "standard"):
        raise ValueError("PPO demonstrations currently support standard single-policy training only; residual targets and per-policy routing are not implemented")


def _action_block(data, key, count, width, path):
    block = np.asarray(data[key], dtype=np.float64)
    if width == 1 and block.ndim == 1:
        block = block[:, None]
    if block.shape != (count, width) or not np.isfinite(block).all():
        raise ValueError(f"Demo {path!r} {key} must be finite with shape {(count, width)}; got {block.shape}")
    return block


def load_demonstrations(paths, obs_dim, action_meta, max_transitions=0):
    parts = {"obs": [], "discrete_actions": [], "continuous_actions": []}
    discrete_count = len(action_meta["discrete"])
    continuous_size = action_meta["continuous_size"]
    remaining = int(max_transitions)
    expected_type = "hybrid" if discrete_count and continuous_size else "continuous" if continuous_size else "discrete"
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "obs" not in data:
                raise ValueError(f"Demo {path!r} is missing obs")
            obs = np.asarray(data["obs"], dtype=np.float32)
            if obs.ndim != 2 or obs.shape[1] != obs_dim or not np.isfinite(obs).all():
                raise ValueError(f"Demo {path!r} obs must be finite with shape (N, {obs_dim})")
            count = len(obs)
            if "action_type" in data and (np.asarray(data["action_type"]).size != 1 or str(np.asarray(data["action_type"]).reshape(-1)[0]) != expected_type):
                raise ValueError(f"Demo {path!r} action_type does not match {expected_type}")
            blocks = []
            for key, width in (("discrete_actions", discrete_count), ("continuous_actions", continuous_size)):
                if not width:
                    block = np.empty((count, 0), dtype=np.float64)
                else:
                    source = key if key in data else "actions" if expected_type != "hybrid" else key
                    if source not in data:
                        raise ValueError(f"Demo {path!r} is missing {key}; hybrid demos need explicit discrete_actions and continuous_actions")
                    block = _action_block(data, source, count, width, path)
                blocks.append(block)
            discrete, continuous = blocks
            for index, component in enumerate(action_meta["discrete"]):
                labels = discrete[:, index]
                if np.any(labels != np.floor(labels)) or np.any(labels < 0) or np.any(labels >= component["size"]):
                    raise ValueError(f"Demo {path!r} has invalid discrete labels for {component['name']}")
            if continuous_size and (np.any(continuous < action_meta["continuous_low"]) or np.any(continuous > action_meta["continuous_high"])):
                raise ValueError(f"Demo {path!r} continuous actions exceed the scenario bounds")
            if remaining > 0:
                count = min(count, remaining)
                remaining -= count
            parts["obs"].append(obs[:count])
            parts["discrete_actions"].append(discrete[:count].astype(np.int32))
            parts["continuous_actions"].append(continuous[:count].astype(np.float32))
        if max_transitions > 0 and remaining == 0:
            break
    if not parts["obs"]:
        return None
    result = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    if not len(result["obs"]):
        raise ValueError("PPO demonstration dataset is empty")
    for value in result.values():
        value.setflags(write=False)
    return result


def imitation_loss(model, obs, discrete_actions, continuous_actions, action_meta, training=True):
    outputs = model(obs, training=training)
    terms = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=discrete_actions[:, index], logits=outputs[index])) for index in range(len(action_meta["discrete"]))]
    if action_meta["continuous_size"]:
        means = outputs[len(action_meta["discrete"])]
        terms.append(tf.reduce_mean(tf.square(means - continuous_actions)))
    return tf.add_n(terms)


def coefficient(args, updates):
    initial = tf.constant(args.ppo_demo_coef, dtype=tf.float32)
    if not args.ppo_demo_decay_updates:
        return initial
    fraction = tf.minimum(tf.cast(updates, tf.float32) / float(args.ppo_demo_decay_updates), 1.0)
    return initial + fraction * (float(args.ppo_demo_final_coef) - initial)


def sample_batch(data, batch_size, updates, seed):
    # Index the private stream by the checkpointed PPO optimizer counter, including after recovery.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) % (2**32), int(updates), 0x50504F]))
    indices = rng.integers(len(data["obs"]), size=batch_size)
    return tuple(tf.convert_to_tensor(data[key][indices]) for key in ("obs", "discrete_actions", "continuous_actions"))


def evaluate_imitation(model, data, action_meta, batch_size):
    count = len(data["obs"])
    total = 0.0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = [tf.convert_to_tensor(data[key][start:stop]) for key in ("obs", "discrete_actions", "continuous_actions")]
        total += float(imitation_loss(model, *batch, action_meta, training=False).numpy()) * (stop - start)
    return total / count


def pretrain(model, data, action_meta, args, validation=None):
    optimizer = tf.keras.optimizers.Adam(args.demo_bc_learning_rate or args.learning_rate)
    rng = np.random.default_rng(np.random.SeedSequence([int(args.env_seed_base) % (2**32), 0x4243]))

    @tf.function(reduce_retracing=True)
    def step(obs, discrete, continuous):
        with tf.GradientTape() as tape:
            loss = imitation_loss(model, obs, discrete, continuous, action_meta)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients([(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables) if gradient is not None])
        return loss

    best_loss, best_weights, best_epoch = None, None, None
    for epoch in range(args.demo_bc_epochs):
        order = rng.permutation(len(data["obs"]))
        for start in range(0, len(order), args.demo_bc_batch_size):
            indices = order[start:start + args.demo_bc_batch_size]
            step(*(tf.convert_to_tensor(data[key][indices]) for key in ("obs", "discrete_actions", "continuous_actions")))
        loss = evaluate_imitation(model, data, action_meta, args.demo_bc_batch_size)
        validation_loss = evaluate_imitation(model, validation, action_meta, args.demo_bc_batch_size) if validation is not None else None
        measured = loss if validation_loss is None else validation_loss
        if not np.isfinite(measured):
            raise ValueError("PPO behavior cloning produced a nonfinite loss")
        if best_loss is None or measured < best_loss:
            best_loss, best_weights, best_epoch = measured, model.get_weights(), epoch + 1
        suffix = "" if validation_loss is None else f" demo_bc_validation_loss={validation_loss:.6f}"
        print(f"demo_bc_epoch={epoch + 1}/{args.demo_bc_epochs} demo_bc_loss={loss:.6f}{suffix}", flush=True)
    if validation is not None:
        model.set_weights(best_weights)
        print(f"PPO BC restored validation-best epoch={best_epoch} loss={best_loss:.6f}", flush=True)
    path = Path(args.bc_actor_weights_path or str(Path(args.weights_path).with_name(Path(args.weights_path).stem.removesuffix('.weights') + '_bc.weights.h5')))
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(path)
    print(f"Saved BC-pretrained PPO model weights: {path}", flush=True)
    return {"best_epoch": best_epoch, "best_loss": best_loss, "weights_path": str(path)}


def training_contract(args, data, action_meta):
    if args.ppo_demo_coef <= 0:
        return None
    digest = hashlib.sha256()
    for key in ("obs", "discrete_actions", "continuous_actions"):
        array = data[key]
        digest.update(key.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return {
        "version": 1,
        "dataset_digest": digest.hexdigest(),
        "demonstrations": len(data["obs"]),
        "discrete_sizes": action_meta["discrete_sizes"],
        "discrete_names": [component["name"] for component in action_meta["discrete"]],
        "continuous_components": [{"name": component["name"], "size": component["size"]} for component in action_meta["continuous"]],
        "continuous_low": action_meta["continuous_low"].tolist(),
        "continuous_high": action_meta["continuous_high"].tolist(),
        "coefficient": args.ppo_demo_coef,
        "final_coefficient": args.ppo_demo_final_coef,
        "decay_optimizer_updates": args.ppo_demo_decay_updates,
        "batch_size": args.demo_bc_batch_size,
        "seed": args.env_seed_base,
    }
