# Metis for Godot

Metis connects a Godot simulation to reinforcement-learning tools built around Gymnasium and TensorFlow/Keras. The add-on provides the runtime nodes used to define agents, observations, actions, rewards, events, progress, and the TCP bridge. It also includes the Inspector helpers and the maintained URDF/STL integration used by the robotics examples.

## Enable the add-on

After copying the `addons/metis` directory into a Godot project, open **Project > Project Settings > Plugins** and enable **Metis**.

### Node palette

In **Create New Node**, the framework's Node-based runtime types live under the `Metis` branch. Their normal inheritance provides smaller subtrees for observation sources, reward components, scenario rewards, event sources, and progress providers. This keeps the main Node list readable while preserving every public class name used by existing scenes and scripts.

`URDFIKController` is part of the main `Metis` branch. Spatial URDF components appear under `Node3D > MetisRobot3D`, including `GodotRobot` and the reusable `URDFRobotArmAgentBody`. The serializable model appears in the resource picker as `MetisURDFResource > URDFRobot`. These parallel Metis-labelled roots are necessary because Godot uses single inheritance: moving a robot or resource under the plain `Metis` Node would remove its transforms or resource serialization.

Other types that require a more specific engine base remain in the corresponding native branch. For example, `TargetSamplingRegion3D` stays under `Area3D`, while URDF rigid bodies and joints retain their physics or joint type.

A release downloaded from the Godot Asset Library carries a matching Python wheel. The first time the packaged add-on is enabled, Metis offers to create a project-local runtime in:

```text
res://.metis/venv
```

Nothing is installed globally. The setup window lets you choose:

- native training on CPU;
- native training with NVIDIA CUDA on Linux;
- native training with TensorFlow Metal on Apple Silicon;
- an existing Python interpreter instead of a managed environment.

Dashboard, policy export, and the limited Stable-Baselines3 comparison backend are independent checkboxes. SB3 is installed alongside the selected native compute profile and is selected during training with `--backend sb3`.

The setup is always explicit and can be reopened with **Tools > Metis Runtime Setup...**. Logs and status are stored under `res://.metis`. Before installing, Metis verifies the release payload, Python version, host platform, and matching Godot/Python package versions. The final diagnostic imports the selected runtime dependencies and requires a visible TensorFlow GPU for CUDA or Metal profiles. A failed validation leaves the project unchanged apart from its local log and status files.

## Training from the editor

The Train wizard reads its option schema from the selected native Python trainer. SAC exposes **Protect demonstrations** to retain complete expert transitions in replay; PPO exposes expert dataset, BC pretraining and auxiliary imitation controls while keeping labels outside on-policy rollouts.

Native algorithms also expose optional solved threshold, solved patience and plateau LR factor. SAC and the DDPG/TD3 family additionally expose an evaluation/export actor EMA field. **All options** includes certification episode count, confidence, held-out seed, cumulative recovery and the remaining supported flags. The schema cache is invalidated when shared control or EMA sources change.

Solved stopping requires an independent test of the pinned candidate after qualification. A failed or incomplete test continues training. Certification is currently single-agent/single-policy only and is not a physical robot-safety guarantee. EMA and SAC protected demonstrations are single-policy features, and PPO imitation does not combine with residual or independent multi-policy mode. These additions are disabled by default.

`ObservationMarginReward` is available among the reward components. It adds a linear or quadratic centering cost around a scalar observation target, optionally gated by another observation, without changing success thresholds.

## Command line

The editor is a convenience layer, not a replacement for the terminal. Once the runtime is ready, its CLI can be used directly:

```bash
.metis/venv/bin/metis doctor
.metis/venv/bin/metis train --help
.metis/venv/bin/metis run --help
.metis/venv/bin/metis record --help
.metis/venv/bin/metis export --help
```

On Windows, use `.metis\venv\Scripts\metis.exe`.

## Source checkouts

The Metis repository uses this directory as the canonical Godot add-on source. Its demo project references the add-on directly, while Python is run from `python/`. Release archives are generated from those same sources, so fixes are not maintained in a second copied tree.

When working from the repository, the complete manuals live under `docs/`. In a standalone installed project, use the documentation and version notes attached to the same Metis release.

## Bundled integrations

Metis includes maintained copies of:

- Godot URDF, under `integrations/urdf`;
- STL-IO, under `integrations/stl`.

They are part of the add-on because Metis relies on local URDF path, mimic-joint, and bounded-joint behavior. Their upstream notices and licenses are preserved beside the source.
