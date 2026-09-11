extends SceneTree


func _initialize() -> void:
	var bases := {}
	for entry in ProjectSettings.get_global_class_list():
		bases[str(entry.get("class", ""))] = str(entry.get("base", ""))

	var expected := {
		"Metis": "Node",
		"Agent": "Metis",
		"BridgeServer": "Metis",
		"ActionSpace": "Metis",
		"ObservationSystem": "Metis",
		"ObservationSource": "Metis",
		"MethodObservationSource": "ObservationSource",
		"RewardSystem": "Metis",
		"RewardComponent": "Metis",
		"StepPenaltyReward": "RewardComponent",
		"ScenarioRewardSystem": "Metis",
		"ScenarioRewardComponent": "Metis",
		"ProgressDeltaScenarioReward": "ScenarioRewardComponent",
		"ScenarioEventSystem": "Metis",
		"ScenarioEventSource": "Metis",
		"AreaReachedEventSource": "ScenarioEventSource",
		"ProgressProvider": "Metis",
		"MethodProgressProvider": "ProgressProvider",
		"URDFIKController": "Metis",
		# Spatial and resource integrations use parallel Metis-labelled roots so
		# their required Godot engine behavior is preserved.
		"MetisRobot3D": "Node3D",
		"GodotRobot": "MetisRobot3D",
		"URDFRobotArmAgentBody": "MetisRobot3D",
		"MetisURDFResource": "Resource",
		"URDFRobot": "MetisURDFResource",
		# Physical editor nodes keep their engine base instead of pretending to be
		# plain Nodes merely to fit under the Metis branch.
		"TargetSamplingRegion3D": "Area3D",
	}

	var failures: Array[String] = []
	for type_name in expected:
		var actual := str(bases.get(type_name, "<missing>"))
		if actual != expected[type_name]:
			failures.append("%s: expected %s, got %s" % [
				type_name,
				expected[type_name],
				actual,
			])

	if not failures.is_empty():
		push_error("Metis node hierarchy test failed: %s" % "; ".join(failures))
		quit(1)
		return

	print("Metis node hierarchy test passed")
	quit(0)
