extends Node3D
## Standalone teleop test for URDFIKController.
## Instances the OpenArm URDF, spawns a Target the TCP should follow, and lets
## you move the target with the keyboard. Watch whether the end-effector tracks.
##
## Keys: W/S = +/-X, A/D = +/-Y, Q/E = +/-Z, arrows = pitch/yaw of the target,
## R = reset target. Hold Shift for fine steps.

const URDF_PATH := "res://assets/urdf/openarm/openarm_bimanual.urdf"
var JOINTS := PackedStringArray([
	"openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
	"openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
	"openarm_right_joint7"])
const TCP_LINK := "openarm_right_hand"
const TCP_OFFSET := Vector3(0, 0.08, 0)
const MAX_SPEED := 0.5

var _robot: Node = null
var _ik: URDFIKController = null
var _target: Node3D = null
var _move_speed := 0.25  # m/s
var _rot_speed := 1.2    # rad/s
var _t := 0.0


func _ready() -> void:
	# light + camera
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-50, -30, 0)
	add_child(light)
	var cam := Camera3D.new()
	cam.position = Vector3(1.2, 0.9, 1.2)
	cam.look_at_from_position(cam.position, Vector3(0, 0.6, -0.2), Vector3.UP)
	add_child(cam)

	# robot
	var packed: PackedScene = load(URDF_PATH)
	if packed == null:
		push_error("Could not load URDF %s" % URDF_PATH)
		return
	_robot = packed.instantiate()
	_robot.control_mode = 1  # KINEMATIC: integrate joint velocity -> pose, no falling
	add_child(_robot)

	# target marker
	_target = Node3D.new()
	var mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.02
	sphere.height = 0.04
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1, 0.2, 0.2)
	mat.emission_enabled = true
	mat.emission = Color(1, 0.2, 0.2)
	mesh.mesh = sphere
	mesh.material_override = mat
	_target.add_child(mesh)
	add_child(_target)
	_reset_target()

	# IK controller
	_ik = URDFIKController.new()
	_ik.robot_path = _robot.get_path()
	_ik.joint_names = JOINTS
	_ik.tcp_link_name = TCP_LINK
	_ik.tcp_local_offset = TCP_OFFSET
	_ik.max_joint_speed = MAX_SPEED
	_ik.track_orientation = false  # start position-only; press T to toggle auto-orientation
	_ik.active = true
	add_child(_ik)
	_ik.set_target(_target)
	print("[teleop] ready. move the red target with W/A/S/D/Q/E + arrows.")


func _physics_process(delta: float) -> void:
	if _target == null:
		return
	var step_scale: float = 0.25 if Input.is_key_pressed(KEY_SHIFT) else 1.0
	var d: float = _move_speed * delta * step_scale
	if Input.is_key_pressed(KEY_W): _target.position.x += d
	if Input.is_key_pressed(KEY_S): _target.position.x -= d
	if Input.is_key_pressed(KEY_A): _target.position.y += d
	if Input.is_key_pressed(KEY_D): _target.position.y -= d
	if Input.is_key_pressed(KEY_Q): _target.position.z += d
	if Input.is_key_pressed(KEY_E): _target.position.z -= d
	var r: float = _rot_speed * delta * step_scale
	if Input.is_key_pressed(KEY_UP): _target.rotate_x(r)      # pitch
	if Input.is_key_pressed(KEY_DOWN): _target.rotate_x(-r)
	if Input.is_key_pressed(KEY_LEFT): _target.rotate_y(r)    # yaw
	if Input.is_key_pressed(KEY_RIGHT): _target.rotate_y(-r)
	if Input.is_key_pressed(KEY_Z): _target.rotate_z(r)      # roll
	if Input.is_key_pressed(KEY_X): _target.rotate_z(-r)
	if Input.is_key_pressed(KEY_R): _reset_target()

	# telemetry every ~0.5s
	_t += delta
	if _t >= 0.5 and _robot != null and _ik != null:
		_t = 0.0
		var tcp: Node3D = _robot.get_link_node(TCP_LINK)
		if tcp != null:
			var p_tcp: Vector3 = tcp.global_transform * TCP_OFFSET
			var err: float = (_target.global_position - p_tcp).length()
			var q_tcp := tcp.global_transform.basis.get_rotation_quaternion()
			var q_tgt := _target.global_transform.basis.get_rotation_quaternion()
			var ang_err := rad_to_deg(q_tcp.angle_to(q_tgt))
			print("[teleop] pos_err=%.3fm orient_err=%.1fdeg" % [err, ang_err])


func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if (event as InputEventKey).keycode == KEY_T:
			_ik.track_orientation = not _ik.track_orientation
			if _ik.track_orientation:
				# snap the target orientation to the current EE pose so there is no
				# jump; you then nudge it a little with the arrows / Z / X.
				var tcp: Node3D = _robot.get_link_node(TCP_LINK)
				if tcp != null:
					_target.global_transform.basis = tcp.global_transform.basis.orthonormalized()
			print("[teleop] auto-orientation = ", _ik.track_orientation)


func _reset_target() -> void:
	# small reachable offset from the rest TCP (~0,0.08,0.16) to validate tracking
	_target.global_position = Vector3(0.10, 0.20, 0.25)
	_target.global_rotation = Vector3.ZERO


func _freeze_base(root: Node) -> void:
	# keep the arm anchored: freeze the first RigidBody3D (the base link)
	var stack: Array = [root]
	while not stack.is_empty():
		var n: Node = stack.pop_back()
		if n is RigidBody3D:
			(n as RigidBody3D).freeze = true
			print("[teleop] froze base body: ", n.name)
			return
		for c in n.get_children():
			stack.append(c)
