# Metis for Godot

Metis connects a Godot simulation to reinforcement-learning tools built around
Gymnasium and TensorFlow/Keras. The add-on provides the runtime nodes used to define
agents, observations, actions, rewards, events, progress, and the TCP bridge. It also
includes the Inspector helpers and the maintained URDF/STL integration used by the
robotics examples.

## Enable the add-on

After copying the `addons/metis` directory into a Godot project, open
**Project > Project Settings > Plugins** and enable **Metis**.

A release downloaded from the Godot Asset Library carries a matching Python wheel.
The first time the packaged add-on is enabled, Metis offers to create a project-local
runtime in:

```text
res://.metis/venv
```

Nothing is installed globally. The setup window lets you choose:

- native training on CPU;
- native training with NVIDIA CUDA on Linux;
- native training with TensorFlow Metal on Apple Silicon;
- an existing Python interpreter instead of a managed environment.

Dashboard, policy export, and the limited Stable-Baselines3 comparison backend are
independent checkboxes. SB3 is installed alongside the selected native compute
profile and is selected during training with `--backend sb3`.

The setup is always explicit and can be reopened with
**Tools > Metis Runtime Setup...**. Logs and status are stored under `res://.metis`.
Before installing, Metis verifies the release payload, Python version, host platform,
and matching Godot/Python package versions. The final diagnostic imports the selected
runtime dependencies and requires a visible TensorFlow GPU for CUDA or Metal
profiles. A failed validation leaves the project unchanged apart from its local log
and status files.

## Command line

The editor is a convenience layer, not a replacement for the terminal. Once the
runtime is ready, its CLI can be used directly:

```bash
.metis/venv/bin/metis doctor
.metis/venv/bin/metis train --help
.metis/venv/bin/metis run --help
.metis/venv/bin/metis record --help
.metis/venv/bin/metis export --help
```

On Windows, use `.metis\venv\Scripts\metis.exe`.

## Source checkouts

The Metis repository uses this directory as the canonical Godot add-on source. Its
demo project references the add-on directly, while Python is run from `python/`.
Release archives are generated from those same sources, so fixes are not maintained
in a second copied tree.

When working from the repository, the complete manuals live under `docs/`. In a
standalone installed project, use the documentation and version notes attached to the
same Metis release.

## Bundled integrations

Metis includes maintained copies of:

- Godot URDF, under `integrations/urdf`;
- STL-IO, under `integrations/stl`.

They are part of the add-on because Metis relies on local URDF path, mimic-joint, and
bounded-joint behavior. Their upstream notices and licenses are preserved beside the
source.
