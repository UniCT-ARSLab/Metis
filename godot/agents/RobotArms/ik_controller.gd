extends Node
class_name URDFIKController
## Jacobian-transpose IK for a URDF arm loaded via the metis GodotRobot.
##
## Drives the arm so its TCP follows a Target Node3D. Outputs per-joint VELOCITY
## commands, capped to `max_joint_speed`, so a recorded command maps 1:1 to the
## RL action space used by urdf_robot_arm_agent (action = velocity / max_speed).
##
## Wiring: add as a child anywhere; set robot_path (GodotRobot), joint_names
## (base->tip, same order the agent controls), tcp_link_name/offset, target_path
## (a Node3D you move with keyboard/joystick). Enable `active` only while
## teleoperating / recording demos -- keep it OFF during RL training.

@export var robot_path: NodePath
@export var joint_names: PackedStringArray
@export var tcp_link_name: String
@export var tcp_local_offset := Vector3.ZERO
@export var target_path: NodePath
@export var track_orientation := true
## "pointing" aligns only the EE approach axis with the target (roll stays free,
## so no 180-degree roll flip); "full" matches the complete target orientation.
@export_enum("pointing", "full") var orientation_mode := "pointing"
## EE-local axis that should point at/along the target (openarm gripper: +Y).
@export var pointing_axis := Vector3(0, 1, 0)
@export_range(0.0, 40.0, 0.1) var position_gain := 4.0   ## P-gain on the DLS step -> velocity
@export_range(0.01, 0.5, 0.005) var damping := 0.08      ## DLS damping lambda (higher = smoother/slower near singularities)
@export_range(0.0, 20.0, 0.1) var orientation_gain := 0.5   ## weight of the orientation task vs position
@export_range(0.01, 5.0, 0.01) var max_joint_speed := 0.5   ## MUST equal the agent's per-joint max speed
@export var position_deadzone := 0.003                       ## m; stop translating within this
@export_range(0.0, 1.0, 0.005) var orientation_deadzone := 0.03  ## rad
@export var limit_margin := 0.02                             ## rad of headroom before a joint limit
@export var active := false

var _robot: Node = null
var _target: Node3D = null
var _last_action: PackedFloat32Array = PackedFloat32Array()


func _ready() -> void:
	_robot = get_node_or_null(robot_path)
	_target = get_node_or_null(target_path)
	_last_action.resize(joint_names.size())


func set_target(node: Node3D) -> void:
	_target = node


func set_robot(node: Node) -> void:
	_robot = node


func _physics_process(_delta: float) -> void:
	if active:
		solve()


## One IK step: compute joint velocities toward the target, cache the normalized
## action, and (when apply=true) drive the robot. Pass apply=false to only compute
## the action -- e.g. when a recorder feeds it through agent.apply_action instead.
func solve(apply: bool = true) -> PackedFloat32Array:
	var n := joint_names.size()
	if _last_action.size() != n:
		_last_action.resize(n)
	if _robot == null or _target == null:
		return _last_action
	var tcp_node: Node3D = _robot.get_link_node(tcp_link_name)
	if tcp_node == null:
		return _last_action

	var tcp_xform: Transform3D = tcp_node.global_transform
	var p_tcp: Vector3 = tcp_xform * tcp_local_offset
	var err_pos: Vector3 = _target.global_position - p_tcp
	var err_rot := Vector3.ZERO
	if track_orientation:
		if orientation_mode == "pointing":
			err_rot = _pointing_error(tcp_xform.basis, _target.global_transform.basis)
		else:
			err_rot = _orientation_error(
				tcp_xform.basis.get_rotation_quaternion(),
				_target.global_transform.basis.get_rotation_quaternion())

	if (err_pos.length() < position_deadzone
			and (not track_orientation or err_rot.length() < orientation_deadzone)):
		for i in range(n):
			if apply:
				_robot.set_joint_target_velocity(joint_names[i], 0.0)
			_last_action[i] = 0.0
		return _last_action

	# Geometric Jacobian columns per joint, read from the CHILD LINK (moves with
	# the arm; the URDF axis is invariant in the child frame). lin = axis x lever,
	# ang = the rotation axis itself.
	var lin: Array[Vector3] = []
	var ang: Array[Vector3] = []
	var jdatas: Array = []
	for i in range(n):
		var jnode = _robot.get_joint_node(joint_names[i])
		var child_link: Node3D = null
		if jnode != null and jnode.joint != null:
			child_link = _robot.get_link_node(jnode.joint.child)
		if child_link != null:
			var axis_world: Vector3 = (child_link.global_transform.basis * jnode.joint.axis).normalized()
			var p_joint: Vector3 = child_link.global_transform.origin
			lin.append(axis_world.cross(p_tcp - p_joint))
			ang.append(axis_world)
			jdatas.append(jnode.joint)
		else:
			lin.append(Vector3.ZERO)
			ang.append(Vector3.ZERO)
			jdatas.append(null)

	# Task rows: 3 (position) or 6 (position + orientation), and the error vector.
	var m := 6 if track_orientation else 3
	var e := PackedFloat32Array()
	e.resize(m)
	e[0] = err_pos.x; e[1] = err_pos.y; e[2] = err_pos.z
	if track_orientation:
		e[3] = err_rot.x * orientation_gain
		e[4] = err_rot.y * orientation_gain
		e[5] = err_rot.z * orientation_gain
	var rows: Array = []  # each: PackedFloat32Array of length n (a Jacobian row)
	for r in range(m):
		var row := PackedFloat32Array()
		row.resize(n)
		for i in range(n):
			match r:
				0: row[i] = lin[i].x
				1: row[i] = lin[i].y
				2: row[i] = lin[i].z
				3: row[i] = ang[i].x
				4: row[i] = ang[i].y
				_: row[i] = ang[i].z
		rows.append(row)

	# damped least squares: solve (J Jt + lambda^2 I) x = e, then dq = Jt x
	var lam2 := damping * damping
	var amat: Array = []
	for r in range(m):
		var arow := PackedFloat32Array()
		arow.resize(m)
		var rr: PackedFloat32Array = rows[r]
		for s in range(m):
			var rs: PackedFloat32Array = rows[s]
			var acc := 0.0
			for i in range(n):
				acc += rr[i] * rs[i]
			arow[s] = acc + (lam2 if r == s else 0.0)
		amat.append(arow)
	var x := _solve_linear(amat, e, m)

	for i in range(n):
		var v := 0.0
		for r in range(m):
			var rr: PackedFloat32Array = rows[r]
			v += rr[i] * x[r]
		v *= position_gain
		v = _limit_clamped(joint_names[i], jdatas[i], v)
		v = clampf(v, -max_joint_speed, max_joint_speed)
		if apply:
			_robot.set_joint_target_velocity(joint_names[i], v)
		_last_action[i] = (v / max_joint_speed) if max_joint_speed > 0.0 else 0.0
	return _last_action


## Solve the m x m system a_mat x = b (Gauss elimination, partial pivot). DLS
## damping keeps a_mat well-conditioned so this stays stable.
func _solve_linear(a_mat: Array, b: PackedFloat32Array, m: int) -> PackedFloat32Array:
	var aug: Array = []
	for r in range(m):
		var row := PackedFloat32Array()
		row.append_array(a_mat[r])
		row.append(b[r])
		aug.append(row)
	for col in range(m):
		var piv := col
		var best := absf((aug[col] as PackedFloat32Array)[col])
		for r in range(col + 1, m):
			var val := absf((aug[r] as PackedFloat32Array)[col])
			if val > best:
				best = val
				piv = r
		if best < 1e-9:
			continue
		if piv != col:
			var tmp = aug[col]
			aug[col] = aug[piv]
			aug[piv] = tmp
		var pivrow: PackedFloat32Array = aug[col]
		var pv: float = pivrow[col]
		for r in range(m):
			if r == col:
				continue
			var rr: PackedFloat32Array = aug[r]
			var factor: float = rr[col] / pv
			for c in range(col, m + 1):
				rr[c] -= factor * pivrow[c]
	var xout := PackedFloat32Array()
	xout.resize(m)
	for r in range(m):
		var rr: PackedFloat32Array = aug[r]
		xout[r] = (rr[m] / rr[r]) if absf(rr[r]) > 1e-9 else 0.0
	return xout


## The last normalized command in the RL action space [-1, 1] per joint.
func get_normalized_action() -> PackedFloat32Array:
	return _last_action.duplicate()


## Rotation vector that aligns the EE approach axis with the target's, ignoring
## roll about that axis (so a target rolled 180 degrees does not flip the wrist).
func _pointing_error(ee_basis: Basis, tgt_basis: Basis) -> Vector3:
	var ee_point := (ee_basis * pointing_axis).normalized()
	var tgt_point := (tgt_basis * pointing_axis).normalized()
	var cross := ee_point.cross(tgt_point)
	var s := cross.length()
	if s < 1e-6:
		return Vector3.ZERO  # already aligned (or exactly anti-aligned)
	var angle := atan2(s, ee_point.dot(tgt_point))
	return cross / s * angle


func _orientation_error(q_from: Quaternion, q_to: Quaternion) -> Vector3:
	var q_err := (q_to * q_from.inverse()).normalized()
	if q_err.w < 0.0:
		q_err = -q_err  # shortest path
	var w := clampf(q_err.w, -1.0, 1.0)
	var s := sqrt(maxf(1.0 - w * w, 0.0))
	if s < 1e-5:
		return Vector3.ZERO
	var angle := 2.0 * acos(w)
	return Vector3(q_err.x, q_err.y, q_err.z) / s * angle


func _limit_clamped(jname: String, jdata, v: float) -> float:
	if jdata == null or jdata.limit == null:
		return v
	var lower := minf(jdata.limit.lower, jdata.limit.upper)
	var upper := maxf(jdata.limit.lower, jdata.limit.upper)
	var pos: float = _robot.get_joint_position(jname)
	if v > 0.0 and pos >= upper - limit_margin:
		return 0.0
	if v < 0.0 and pos <= lower + limit_margin:
		return 0.0
	return v
