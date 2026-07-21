extends SceneTree


func _initialize() -> void:
	var packed := load("res://scenarios/cars/cars_scenario.tscn") as PackedScene
	if not packed:
		push_error("Cars path finish test: scenario could not be loaded.")
		quit(1)
		return

	var scenario := packed.instantiate() as Node3D
	var bridge := scenario.get_node("BridgeServer")
	scenario.remove_child(bridge)
	bridge.free()
	root.add_child(scenario)
	await process_frame

	var provider: Path3DProgressProvider = scenario.get_node(
		"ScenarioController/ProgressProvider")
	var path: Path3D = scenario.get_node("RoadGenerator")
	var finish: Area3D = scenario.get_node("Environment/EndRace")
	var car: Car = scenario.get_node("Agents/Car")
	car.set_physics_process(false)

	var curve := path.curve
	var last_point := path.to_global(curve.get_point_position(curve.point_count - 1))
	var previous_point := path.to_global(curve.get_point_position(curve.point_count - 2))
	var original_y := car.global_position.y

	car.global_position = Vector3(
		finish.global_position.x, original_y, finish.global_position.z)
	var finish_progress := provider.measure_progress(car)
	car.global_position = Vector3(previous_point.x, original_y, previous_point.z)
	var previous_progress := provider.measure_progress(car)

	var finish_planar_error := Vector2(
		last_point.x - finish.global_position.x,
		last_point.z - finish.global_position.z).length()
	var passed := (
		curve.point_count == 9
		and finish_planar_error < 0.001
		and finish_progress > 0.999
		and previous_progress < 0.99)
	if not passed:
		push_error(
			"Cars path finish test failed: points=%d error=%f previous=%f finish=%f" % [
				curve.point_count,
				finish_planar_error,
				previous_progress,
				finish_progress,
			])
		scenario.free()
		quit(1)
		return

	scenario.free()
	print("Cars path finish alignment test passed")
	quit(0)
