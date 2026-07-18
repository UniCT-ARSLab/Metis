extends SceneTree


func _initialize() -> void:
	var body := CharacterBody2D.new()
	body.velocity = Vector2(3.0, -4.0)

	var agent := Agent.new()
	body.add_child(agent)

	var observation_system := ObservationSystem.new()
	agent.add_child(observation_system)

	var velocity_source := PropertyObservationSource.new()
	velocity_source.observation_name = "velocity"
	velocity_source.property_path = NodePath("velocity")
	observation_system.add_child(velocity_source)
	velocity_source.register_observations(agent, body)

	var velocity_x_source := PropertyObservationSource.new()
	velocity_x_source.observation_name = "velocity_x"
	velocity_x_source.property_path = NodePath("velocity:x")
	velocity_x_source.normalize_numeric = true
	velocity_x_source.input_min = -10.0
	velocity_x_source.input_max = 10.0
	velocity_x_source.output_min = -1.0
	velocity_x_source.output_max = 1.0
	observation_system.add_child(velocity_x_source)
	velocity_x_source.register_observations(agent, body)

	var observations := agent.get_observations()
	var vector_value: Variant = observations.get("velocity", null)
	var x_value := float(observations.get("velocity_x", -100.0))
	var passed: bool = vector_value is Vector2 and vector_value == Vector2(3.0, -4.0)
	passed = passed and is_equal_approx(x_value, 0.3)
	passed = passed and agent.get_observation_vector().size() == 3

	body.free()
	if not passed:
		push_error("PropertyObservationSource test failed: %s" % observations)
		quit(1)
		return
	print("PropertyObservationSource test passed")
	quit(0)
