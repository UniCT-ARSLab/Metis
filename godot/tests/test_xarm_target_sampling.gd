extends SceneTree


func _initialize() -> void:
	var packed := load("res://scenarios/robotarms/XarmScenario.tscn") as PackedScene
	if not packed:
		push_error("XArm target sampling test: scenario could not be loaded.")
		quit(1)
		return

	var scenario := packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame
	await physics_frame

	var spawn_pool: Array[Marker3D] = scenario.call("_target_spawn_pool")
	var lower := spawn_pool[0].global_position
	var upper := lower
	for marker in spawn_pool:
		lower = lower.min(marker.global_position)
		upper = upper.max(marker.global_position)

	var rng := RandomNumberGenerator.new()
	rng.seed = 12345
	var found_non_marker_position := false
	var passed := spawn_pool.size() > 1
	for _sample_index in range(100):
		var sampled: Vector3 = scenario.call("_sample_target_position", rng, spawn_pool)
		passed = passed and sampled.x >= lower.x and sampled.x <= upper.x
		passed = passed and sampled.y >= lower.y and sampled.y <= upper.y
		passed = passed and sampled.z >= lower.z and sampled.z <= upper.z
		var matches_marker := false
		for marker in spawn_pool:
			if sampled.is_equal_approx(marker.global_position):
				matches_marker = true
				break
		if not matches_marker:
			found_non_marker_position = true

	passed = passed and found_non_marker_position
	if not passed:
		push_error(
			"XArm target sampling test failed: markers=%d bounds=%s..%s continuous=%s" % [
				spawn_pool.size(), lower, upper, found_non_marker_position])
		scenario.free()
		quit(1)
		return

	print(
		"XArm target sampling test passed: markers=%d bounds=%s..%s" % [
			spawn_pool.size(), lower, upper])
	scenario.free()
	quit(0)
