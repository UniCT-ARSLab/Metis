# Godot framework tests

The test scripts directly under this directory cover reusable Metis runtime, editor, reward, observation, scenario-contract, and URDF behavior.

Some integration tests load an example scene or robot as a fixture. They remain framework tests when their assertions target a reusable API rather than the example task's reward, curriculum, geometry, or success behavior.

`test_observation_margin_reward.gd` checks centering, both band edges, behavior outside the band, linear/quadratic shapes, weights, observation gating and invalid inputs for the generic margin reward.

`test_training_control_options.gd` requires `--editor` and checks that native training dialogs expose solved/LR controls, certification settings, supported actor EMA fields and PPO demo inputs from the actual Python schema without launching training.

Task-specific tests belong under ignored `tests/local/`. They remain runnable inside the Godot project with:

```bash
godot --headless --path godot --script tests/local/<test>.gd
```

Run `test_metis_runtime_setup.gd` with `--editor` because it exercises Godot editor APIs:

```bash
godot --headless --editor --path godot --script tests/test_metis_runtime_setup.gd
```
