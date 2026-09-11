extends SceneTree


func _initialize() -> void:
	var packed := load("res://scenarios/breakout/breakout_scenario.tscn") as PackedScene
	if packed == null:
		push_error("Scenario spec contract test could not load Breakout.")
		quit(1)
		return

	var scenario := packed.instantiate()
	root.add_child(scenario)
	await process_frame
	await physics_frame

	var controller: ScenarioController = scenario.get_node("ScenarioController")
	var response := controller.configure({
		"max_steps": 17,
		"physics_frames_per_step": 2,
		"strict_physics_frames": true,
	})
	var spec := controller.get_spec()
	var passed := bool(spec.get("ok", false))
	passed = passed and int(spec.get("max_steps", -1)) == 17
	passed = passed and int(spec.get("physics_frames_per_step", -1)) == 2
	passed = passed and bool(spec.get("strict_physics_frames", false))
	passed = passed and bool(response.get("strict_physics_frames", false))
	passed = passed and not spec.get("agents", []).is_empty()
	if not spec.get("agents", []).is_empty():
		var agent_spec: Dictionary = spec["agents"][0]
		passed = passed and int(agent_spec.get("obs_dim", -1)) == (
			agent_spec.get("observation_scalar_names", []) as Array).size()
		passed = passed and (
			agent_spec.get("observation_names", []) as Array).size() == 3
		passed = passed and (
			agent_spec.get("observation_layout", []) as Array).size() == 3
		passed = passed and not (
			agent_spec.get("action_names", []) as Array).is_empty()

	if not passed:
		push_error("Scenario spec contract test failed: spec=%s response=%s" % [
			spec,
			response,
		])
		scenario.free()
		quit(1)
		return

	print("Scenario spec contract test passed")
	scenario.free()
	quit(0)
