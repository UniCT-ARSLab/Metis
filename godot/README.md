# Godot URDF add-on

Metis includes a copy of the Godot URDF add-on for Godot 4.6. It parses URDF files with
Godot's native `XMLParser` and generates visual meshes, collision bodies, joints, and a
`GodotRobot` runtime controller.

## Importing a robot

Drag a `.urdf` file into the project and let the editor import it, then instantiate the
generated resource in a scene. For runtime loading, attach `urdf_loader.gd` to a
`Node3D`; the generated robot nodes become its children and can be extended by your own
controller or Metis agent adapter.

Both the importer's `package_folder` option and `URDFLoader.urdf_file_path` use
project-local paths such as:

```text
res://assets/urdf/xarm/xarm_fixed.urdf
res://assets/urdf
```

The second path is the root used to resolve references such as
`package://xarm/meshes/base.stl`. Absolute paths inside the Godot project are
automatically localized to `res://`, keeping scenes and import settings portable
between machines. Absolute paths outside the project remain supported when needed.

The add-on also contains a basic wheeled-robot controller. Configure the drive type and
wheel joints, then use the corresponding input actions to test the imported model.

## Metis extensions

This copy adds URDF `mimic` support. A follower joint reads its source, multiplier, and
offset from:

```xml
<mimic joint="source_joint" multiplier="1.0" offset="0.0"/>
```

Mimic joints are not independent actuators. `GodotRobot` synchronizes their target,
position, and velocity from the source joint in both kinematic and physics-motor modes.
The parser also reports missing sources and dependency cycles.

Run the regression test with:

```bash
/path/to/Godot --headless --path godot \
  --script res://tests/test_urdf_mimic.gd
```

## Implementation notes

Generated collision shapes are children of `RigidBody3D` links connected by
`Generic6DOFJoint3D` nodes. This representation is flatter than the source XML. The
custom editor dock presents the original robot hierarchy when you need to inspect it.

URDF files that reference STL meshes can use the bundled `godot-stl-io` add-on.

The original add-on and demo projects are available from:

- [Godot URDF](https://github.com/brean/godot_urdf)
- [Godot URDF demos](https://github.com/brean/godot_urdf_demo)
- [godot-stl-io](https://github.com/onze/godot-stl-io)
