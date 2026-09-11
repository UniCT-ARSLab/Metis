# Local Godot development tools

`tools/local/` is reserved for task-specific probes, demo recorders, data generators, and temporary diagnostics. Its contents are ignored by Git but stay inside the Godot project so they can use `res://` resources.

Run a local script from the repository root with:

```bash
godot --headless --path godot --script tools/local/<script>.gd
```

Reusable runtime code belongs under `addons/metis/`. A tool intended to become part of the supported framework should expose a stable command or editor action and include focused tests and documentation before moving out of `tools/local/`.
