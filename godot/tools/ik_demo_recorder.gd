extends Node3D
## IK demo recorder. Instances the real OpenArm agent (so observations/rewards
## match training exactly), drives it with the IK toward a Target you teleoperate,
## and logs (obs, action, reward, next_obs, done) per step. Demos are written as
## JSON and converted to Metis npz by python/tools/convert_ik_demos.py.
##
## Keys: W/A/S/D/Q/E move the target, arrows + Z/X rotate it, T toggles auto-
## orientation, SPACE starts/stops recording the current demo (saved on stop),
## N resets the arm to home for a fresh demo, R resets the target.
##
## Run headless with --auto to self-drive to a fixed target and save one demo
## (pipeline smoke test).

const AGENT_SCENE := "res://agents/RobotArms/openarm/openarm_agent.tscn"
var JOINTS := PackedStringArray([
	"openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
	"openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
	"openarm_right_joint7"])
const TCP_LINK := "openarm_right_hand"
const TCP_OFFSET := Vector3(0, 0.08, 0)
const MAX_SPEED := 0.5
const OUT_DIR := "user://ik_demos"

var _agent: Node = null       # urdf_robot_arm_agent
var _metis: Node = null       # metis Agent (obs/reward)
var _robot: Node = null
var _ik: URDFIKController = null
var _target: Node3D = null

var _recording := false
var _pending := {}            # {obs, action} awaiting its next_obs/reward/done
var _demo := {"obs": [], "actions": [], "rewards": [], "next_obs": [], "dones": []}
var _demo_count := 0
var _auto := false
var _auto_frames := 0


func _ready() -> void:
	_auto = "--auto" in OS.get_cmdline_user_args() or "--auto" in OS.get_cmdline_args()
	DirAccess.make_dir_recursive_absolute(OUT_DIR)

	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-50, -30, 0)
	add_child(light)
	var cam := Camera3D.new()
	cam.position = Vector3(1.1, 0.8, 1.1)
	cam.look_at_from_position(cam.position, Vector3(0, 0.35, 0.1), Vector3.UP)
	add_child(cam)

	var packed: PackedScene = load(AGENT_SCENE)
	_agent = packed.instantiate()
	add_child(_agent)
	_robot = _agent.get_node_or_null("openarm")
	_metis = _agent.get_node_or_null("Agent")
	if _robot == null or _metis == null:
		push_error("recorder: missing robot/Agent node under the agent scene")
		return
	_robot.control_mode = 1  # KINEMATIC

	_target = Node3D.new()
	var mesh := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.02; sphere.height = 0.04
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1, 0.2, 0.2); mat.emission_enabled = true; mat.emission = Color(1, 0.2, 0.2)
	mesh.mesh = sphere; mesh.material_override = mat
	_target.add_child(mesh)
	add_child(_target)
	_reset_target()

	# route the agent's reach target to our teleop target
	if "target" in _agent:
		_agent.target = _target
	if "target_pose" in _agent:
		_agent.target_pose = _target

	_ik = URDFIKController.new()
	_ik.robot_path = _robot.get_path()
	_ik.joint_names = JOINTS
	_ik.tcp_link_name = TCP_LINK
	_ik.tcp_local_offset = TCP_OFFSET
	_ik.max_joint_speed = MAX_SPEED
	_ik.track_orientation = false
	_ik.active = false  # we drive it manually from _physics_process
	add_child(_ik)
	_ik.set_target(_target)

	var obs: Array = _metis.get_observation_vector()
	print("[rec] ready. obs_size=%d (expect 27). auto=%s" % [obs.size(), str(_auto)])
	if _auto:
		_target.global_position = Vector3(0.10, 0.20, 0.25)
		_recording = true
		print("[rec] AUTO recording to fixed target...")


func _physics_process(_delta: float) -> void:
	if _robot == null or _metis == null:
		return
	if not _auto:
		_handle_teleop(_delta)

	# env loop: obs_t -> action_t -> apply -> (next frame) reward/next_obs/done
	var obs_now: Array = _metis.get_observation_vector()
	if _recording and not _pending.is_empty():
		var reward: float = _metis.get_reward()
		var done: bool = _agent.has_succeeded() if _agent.has_method("has_succeeded") else false
		_demo["obs"].append(_pending["obs"])
		_demo["actions"].append(_pending["action"])
		_demo["rewards"].append(reward)
		_demo["next_obs"].append(obs_now)
		_demo["dones"].append(done)

	_ik.solve(false)  # compute only
	var action := _ik.get_normalized_action()
	_agent.apply_action(action)  # applies velocities + updates previous_action
	if _recording:
		_pending = {"obs": obs_now, "action": Array(action)}
	else:
		_pending = {}

	if _auto:
		_auto_frames += 1
		if _auto_frames >= 180:  # ~3s
			_stop_and_save()
			get_tree().quit()


func _handle_teleop(delta: float) -> void:
	var step_scale: float = 0.25 if Input.is_key_pressed(KEY_SHIFT) else 1.0
	var d: float = 0.25 * delta * step_scale
	if Input.is_key_pressed(KEY_W): _target.position.x += d
	if Input.is_key_pressed(KEY_S): _target.position.x -= d
	if Input.is_key_pressed(KEY_A): _target.position.y += d
	if Input.is_key_pressed(KEY_D): _target.position.y -= d
	if Input.is_key_pressed(KEY_Q): _target.position.z += d
	if Input.is_key_pressed(KEY_E): _target.position.z -= d
	var r: float = 1.2 * delta * step_scale
	if Input.is_key_pressed(KEY_UP): _target.rotate_x(r)
	if Input.is_key_pressed(KEY_DOWN): _target.rotate_x(-r)
	if Input.is_key_pressed(KEY_LEFT): _target.rotate_y(r)
	if Input.is_key_pressed(KEY_RIGHT): _target.rotate_y(-r)
	if Input.is_key_pressed(KEY_Z): _target.rotate_z(r)
	if Input.is_key_pressed(KEY_X): _target.rotate_z(-r)


func _unhandled_key_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	match (event as InputEventKey).keycode:
		KEY_T:
			_ik.track_orientation = not _ik.track_orientation
			if _ik.track_orientation:
				var tcp: Node3D = _robot.get_link_node(TCP_LINK)
				if tcp != null:
					_target.global_transform.basis = tcp.global_transform.basis.orthonormalized()
			print("[rec] auto-orientation = ", _ik.track_orientation)
		KEY_SPACE:
			if _recording:
				_stop_and_save()
			else:
				_recording = true
				_pending = {}
				print("[rec] recording started")
		KEY_N:
			_reset_arm_home()
		KEY_R:
			_reset_target()


func _stop_and_save() -> void:
	_recording = false
	var count: int = _demo["obs"].size()
	if count == 0:
		print("[rec] nothing to save")
		return
	# mark the final transition terminal
	_demo["dones"][count - 1] = true
	var path := "%s/ik_demo_%03d.json" % [OUT_DIR, _demo_count]
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(JSON.stringify(_demo))
	f.close()
	print("[rec] saved %s (%d transitions)" % [path, count])
	_demo_count += 1
	_demo = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "dones": []}
	_pending = {}


func _reset_target() -> void:
	_target.global_position = Vector3(0.10, 0.20, 0.25)
	_target.global_rotation = Vector3.ZERO


func _reset_arm_home() -> void:
	if _agent.has_method("reset_to_home"):
		_agent.reset_to_home()
	elif _robot.has_method("reset_joint_positions"):
		_robot.reset_joint_positions({})
	print("[rec] arm reset")
