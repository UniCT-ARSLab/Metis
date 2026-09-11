# Godot framework tests

The test scripts directly under this directory cover reusable Metis runtime, editor, reward, observation, scenario-contract, and URDF behavior.

Some integration tests load an example scene or robot as a fixture. They remain framework tests when their assertions target a reusable API rather than the example task's reward, curriculum, geometry, or success behavior.

Task-specific tests belong under ignored `tests/local/`. They remain runnable inside the Godot project with:

```bash
godot --headless --path godot --script tests/local/<test>.gd
```

Run `test_metis_runtime_setup.gd` with `--editor` because it exercises Godot editor APIs:

```bash
godot --headless --editor --path godot --script tests/test_metis_runtime_setup.gd
```
