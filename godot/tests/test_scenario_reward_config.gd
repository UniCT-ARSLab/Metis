extends SceneTree


func _initialize() -> void:
	var component = load(
		"res://addons/metis/runtime/agent/scenario_reward_components/ScenarioRewardComponent.gd"
	).new()
	root.add_child(component)
	component.weight = 1.0
	component.enabled = true
	component.scenario_config_properties = PackedStringArray(["weight"])

	component.apply_scenario_config({
		"weight": 2.5,
		"enabled": false,
		"max_steps": 10,
	})

	if not is_equal_approx(component.weight, 2.5):
		push_error("Whitelisted scenario reward configuration was not applied")
		quit(1)
		return
	if not component.enabled:
		push_error("Non-whitelisted scenario reward property was overwritten")
		quit(1)
		return

	print("Scenario reward configuration test passed")
	quit(0)
