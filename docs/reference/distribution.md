# Distribution and installation

Metis is developed as one repository and distributed as two coordinated pieces:

- a Godot add-on under `addons/metis`;
- a Python package named `metis-rl`.

They share a version and a wire protocol. The release builder refuses to package
different versions together.

## Source layout

The repository keeps one canonical copy of each side:

```text
godot/
  addons/metis/   Godot runtime, editor integration, URDF and STL support
  agents/         reusable example agents
  scenarios/      example training environments
  tests/          Godot regression tests

python/
  algorithms/     native TensorFlow/Keras learners
  backends/       optional backend adapters
  core/           models, replay, collection, checkpoints and health
  envs/           Gymnasium bridge and Godot process management
  train.py        source-checkout training entry point
  run.py          source-checkout inference entry point
  recorder.py     source-checkout demonstration recorder
  export.py       source-checkout policy exporter
```

The demo project consumes `godot/addons/metis` directly. The add-on is not copied into
another development directory, and the Python wheel is generated from `python/`
instead of being checked in.

## Installing from a release

Copy the release's `addons/metis` directory into a Godot project, then enable
**Metis** in **Project > Project Settings > Plugins**.

The packaged add-on includes:

- all Godot runtime and editor scripts;
- the maintained Godot URDF and STL integrations;
- the matching `metis_rl` wheel;
- compute profiles for CPU, Linux CUDA, and Apple Silicon Metal;
- optional SB3, dashboard, and policy-export components;
- third-party notices and licenses.

When the packaged add-on is enabled without a matching runtime, it opens
**Metis Runtime Setup**. The same window is always available from the **Tools** menu.
Metis does not install packages silently.

Before creating an environment, the installer checks:

- Python is 3.11 or newer;
- the selected profile is valid for the host (`linux_cuda` on Linux and
  `macos_metal` on Apple Silicon);
- the wheel and every requirements file match the SHA-256 values in the release
  manifest;
- the Godot and Python package versions are identical.

Installation stops before `pip` runs if any of these checks fails. This catches
incomplete downloads, mixed release files, and selecting a profile for the wrong
machine.

## Project-local Python runtime

The managed setup creates:

```text
.metis/
  venv/          isolated Python environment
  runtime.json   selected interpreter, Metis version and profile
  setup.log      complete setup and diagnostic output
  setup_status.json
```

`.metis` belongs to the local machine and should not be committed. A project can
instead point Metis at an existing interpreter. The setup validates that interpreter
with `metis doctor` before marking it ready. For a packaged release, an existing
environment must contain the exact `metis-rl` version bundled with the add-on.

Available profiles are:

| Profile | Intended host |
|---|---|
| `cpu` | portable native TensorFlow installation |
| `linux_cuda` | Linux with a supported NVIDIA driver |
| `macos_metal` | Apple Silicon with the pinned TensorFlow Metal pair |

Dashboard, policy export tools, and Stable-Baselines3 are independent optional
components. Dashboard and policy export are selected by default in the setup window
because they are part of the normal Metis workflow, but they can be deselected for a
smaller runtime. SB3 remains opt-in and does not replace the compute profile: enabling
it adds the limited PyTorch comparison backend beside the native TensorFlow/Keras
backend. Training selects it later with `--backend sb3`.

A separate SB3-only environment may still be useful for reproducible backend
benchmarks, but it is an isolation choice rather than an installation requirement.

The final validation imports every package required by the chosen profile, checks the
Godot executable, and verifies the installed Metis version. CUDA and Metal profiles
are marked ready only if TensorFlow can actually see a GPU; installing the package
without a working driver or Metal device is not treated as a successful accelerated
setup.

The setup window reports one of four states:

| State | Meaning |
|---|---|
| `running` | a validation or installation step is active |
| `ready` | all selected capabilities passed their checks |
| `error` | setup failed; the exact command output is in `setup.log` |
| `cancelled` | the user stopped an active setup |

An unexpected helper-process exit is converted into a persistent `error` state, so
the editor never waits indefinitely on a stale `running` file.

Deleting `.metis` removes the managed environment without touching system Python.
Reopen the setup window to recreate it or switch profile.

## Terminal workflow

The command line remains the complete interface. A managed release runtime exposes:

```bash
.metis/venv/bin/metis train --help
.metis/venv/bin/metis run --help
.metis/venv/bin/metis record --help
.metis/venv/bin/metis export --help
.metis/venv/bin/metis doctor
```

The `metis-train`, `metis-run`, `metis-record`, `metis-export`, and `metis-doctor`
aliases are installed as well. Windows places these launchers in
`.metis\venv\Scripts`.

From a source checkout, existing commands such as
`python/.venv/bin/python python/train.py` remain supported and exercise the same
modules. This is the preferred development workflow because edits take effect without
rebuilding a wheel.

## Building a release

From the repository root:

```bash
python/.venv/bin/python packaging/build_release.py --dist-dir dist
```

The builder:

1. checks the versions in `plugin.cfg` and `pyproject.toml`;
2. builds the Python wheel from canonical source;
3. stages the Godot add-on and generated Python payload;
4. records and validates SHA-256 hashes for the wheel and requirement profiles;
5. validates required files and license notices;
6. writes a deterministic ZIP and SHA-256 checksums.

The output contains the wheel separately for normal Python installation and a
`metis-godot-VERSION.zip` ready for the Godot Asset Library. Build products stay out
of Git.

Use a release tag for public builds. A practical release sequence is:

1. update the shared version;
2. run Python and Godot tests;
3. build the release twice and compare checksums;
4. test the ZIP in an empty Godot project;
5. publish the tag, wheel, ZIP, and checksums together.

The repository includes that isolated project check:

```bash
python/.venv/bin/python packaging/smoke_test_release.py \
  dist/metis-godot-VERSION.zip \
  --godot-bin /path/to/Godot
```

## Why URDF and STL are included

The robotics runtime depends on behavior that differs from the original upstream
add-ons, including project-local URDF paths, mimic joints, bounded joint control, and
the XArm linkage used by the examples. Shipping those sources inside Metis makes a
release reproducible and prevents an Asset Library install from silently selecting an
incompatible dependency version.

Upstream attribution remains in:

- `addons/metis/integrations/urdf/UPSTREAM.md`;
- `addons/metis/integrations/urdf/LICENSE`;
- `addons/metis/integrations/stl/UPSTREAM.md`;
- `addons/metis/integrations/stl/license.txt`.
